#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shot_detect.py — 镜头边界检测 / 转场分类 / 节奏统计 / 镜头计数（cine-eval F1~F4）。

F1 镜头边界: 相邻帧 H-S 二维直方图(50x60 bin)相关性 corr < 0.5 → 边界。
             先每 --step 帧粗扫提速，再对低相关区间回到逐帧精度定位。
F2 转场分类: 边界点前后 --window 帧窗口内分析帧间差异/亮度模式，
             分 hard_cut / dissolve / fade，判不准标 unknown。
F3 节奏统计: 镜头数、平均时长、时长标准差、最短/最长 + 模式分类。
F4 镜头计数: detected_shot_count = 边界数 + 1。

时间戳 = 帧号 / fps（fps 从 cv2.CAP_PROP_FPS 读取）。

用法:
    python shot_detect.py video.mp4
    python shot_detect.py video.mp4 --step 3 --window 40 --output shots.json
    python shot_detect.py video.mp4 --expect-shots 5   # 直接输出 D4 规则分
"""
import argparse
import json
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CORR_THRESHOLD = 0.5  # F1: 直方图相关性 < 0.5 判镜头边界
FADE_BLACK = 0.10     # F2 fade: 近黑亮度阈值（0~1）
FADE_DELTA = 0.15     # F2 fade: 亮度下降/上升的最小幅度


def hs_hist(frame_bgr):
    """F1 的 H-S 二维直方图（H 50 bin / S 60 bin，归一化）。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def read_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def read_range(cap, lo, hi):
    """顺序读取 [lo, hi] 闭区间帧，返回 {帧号: 帧图像}。"""
    frames = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for idx in range(lo, hi + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames[idx] = frame
    return frames


def coarse_scan(cap, total, step):
    """每 step 帧采样，算相邻样本的 H-S 相关性。

    返回 (样本帧号表 indices, corrs)；corrs[i] 对应样本对 (indices[i], indices[i+1])。
    """
    indices = list(range(0, max(total - 1, 1), step))
    if not indices or indices[-1] != total - 1:
        indices.append(total - 1)
    corrs, prev_hist = [], None
    for idx in indices:
        frame = read_frame_at(cap, idx)
        if frame is None:
            continue
        h = hs_hist(frame)
        if prev_hist is not None:
            corrs.append(float(cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL)))
        prev_hist = h
    return indices, corrs


def group_events(indices, corrs, gap_tol=2):
    """把 corr < 阈值的相邻样本对归并为边界事件（允许中间隔 gap_tol 个正常样本对，
    以覆盖淡出到黑、再淡入这类相关性两次跌落的情况）。返回粗粒度帧区间列表。"""
    low = [i for i, c in enumerate(corrs) if c < CORR_THRESHOLD]
    groups = []
    for i in low:
        if groups and i - groups[-1][1] <= gap_tol + 1:
            groups[-1][1] = i
        else:
            groups.append([i, i])
    return [(indices[s], indices[min(e + 1, len(indices) - 1)]) for s, e in groups]


def refine_boundary(cap, span, total):
    """在粗扫区间 ±2 帧内逐帧算相关性，把边界定位到单帧精度。

    返回新镜头的起始帧号（corr 最小的相邻帧对中的后一帧）。
    """
    lo = max(0, span[0] - 2)
    hi = min(total - 1, span[1] + 2)
    frames = read_range(cap, lo, hi)
    idxs = sorted(frames)
    best_frame, best_corr = None, 2.0
    for a, b in zip(idxs[:-1], idxs[1:]):
        if b != a + 1:
            continue
        c = float(cv2.compareHist(hs_hist(frames[a]), hs_hist(frames[b]), cv2.HISTCMP_CORREL))
        if c < best_corr:
            best_corr, best_frame = c, b
    return best_frame if best_frame is not None else span[1]


def classify_transition(cap, boundary, total, window=30):
    """F2: 分析边界点前后 window 帧内的帧间差异 / 亮度模式。

    - fade:      窗口内最低亮度近黑，且亮度降/升幅度 > 0.15（淡出至黑或从黑淡入）
    - hard_cut:  相关性在切点单帧突刺（<0.5），相邻两侧 > 0.8
    - dissolve:  连续 3~40 帧中等差异（corr < 0.7）且无近黑
    - unknown:   以上均不满足
    """
    lo = max(0, boundary - window)
    hi = min(total - 1, boundary + window)
    frames = read_range(cap, lo, hi)
    idxs = sorted(frames)
    if len(idxs) < 3:
        return "unknown"
    brightness = [float(np.mean(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY))) / 255.0
                  for i in idxs]
    corrs = [float(cv2.compareHist(hs_hist(frames[a]), hs_hist(frames[b]), cv2.HISTCMP_CORREL))
             for a, b in zip(idxs[:-1], idxs[1:]) if b == a + 1]
    # boundary 在窗口中的下标；corrs[j-1] 即切点相邻对
    j = idxs.index(boundary) if boundary in idxs else len(idxs) // 2

    # --- fade 检测: 亮度线性降至近黑 / 从近黑升起 ---
    min_b = min(brightness)
    pre, post = brightness[:j], brightness[j:]
    drop = (max(pre) - min_b) if pre else 0.0
    rise = (max(post) - min_b) if post else 0.0
    if min_b < FADE_BLACK and max(drop, rise) > FADE_DELTA:
        return "fade"

    if corrs:
        # --- hard_cut 检测: 单帧突刺 ---
        cut = j - 1
        left_ok = cut - 1 < 0 or corrs[cut - 1] > 0.8
        right_ok = cut + 1 >= len(corrs) or corrs[cut + 1] > 0.8
        if 0 <= cut < len(corrs) and corrs[cut] < CORR_THRESHOLD and left_ok and right_ok:
            return "hard_cut"
        # --- dissolve 检测: 连续多帧中等差异 ---
        run = longest = 0
        for c in corrs:
            run = run + 1 if c < 0.7 else 0
            longest = max(longest, run)
        if 3 <= longest <= 40:
            return "dissolve"
    return "unknown"


def rhythm_stats(shots):
    """F3: 切换节奏统计与模式分类。"""
    durs = [s["duration"] for s in shots]
    n = len(durs)
    if n == 0:
        return {"shot_count": 0, "pattern": "unknown"}
    mean = float(np.mean(durs))
    std = float(np.std(durs))
    cv = std / mean if mean > 0 else 0.0  # 变异系数，衡量"标准差大小"
    if n <= 2 and max(durs) > 15:
        pattern = "long_take"      # 仅 1~2 个镜头且单镜头 > 15s
    elif mean < 2 and cv < 0.5:
        pattern = "fast_cutting"   # 平均时长 < 2s 且标准差小
    elif 2 <= mean <= 6 and cv <= 0.5:
        pattern = "regular"        # 平均时长 2~6s
    else:
        pattern = "variable"       # 标准差大 → 变速节奏
    return {
        "shot_count": n,
        "mean_duration": round(mean, 3),
        "std_duration": round(std, 3),
        "min_duration": round(min(durs), 3),
        "max_duration": round(max(durs), 3),
        "duration_cv": round(cv, 3),
        "pattern": pattern,
    }


def d4_score(detected_count, expected_count):
    """F4: 实际镜头数与指令指定数的差值映射得分（纯规则，无需 LLM）。

    差 0→5；差 1→4；差 2→3；差 3~4→2；差 ≥5→1。
    """
    diff = abs(int(expected_count) - int(detected_count))
    score = {0: 5.0, 1: 4.0, 2: 3.0}.get(diff, 2.0 if diff <= 4 else 1.0)
    return {"expected": int(expected_count), "detected": int(detected_count),
            "diff": diff, "score": score}


def merge_adjacent_fades(cap, transitions, window):
    """F2 分类后处理：相邻两个 fade 边界间隔 <= window 帧时，视为同一次
    "淡出到黑再淡入" 被检出两次，合并为一个边界（取区间内最暗帧，即真正黑场点）。"""
    merged = []
    for tr in transitions:
        if (merged and tr["type"] == "fade" and merged[-1]["type"] == "fade"
                and tr["frame"] - merged[-1]["frame"] <= window):
            lo, hi = merged[-1]["frame"], tr["frame"]
            frames = read_range(cap, lo, hi)
            if frames:
                merged[-1]["frame"] = min(
                    frames, key=lambda i: np.mean(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)))
        else:
            merged.append(dict(tr))
    return merged


def detect_shots(video_path, step=2, window=30, expect_shots=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("视频没有可读帧")

    # F1: 粗扫 → 事件归并 → 逐帧精定位
    indices, corrs = coarse_scan(cap, total, step)
    events = group_events(indices, corrs)
    boundaries = sorted({refine_boundary(cap, span, total) for span in events})
    boundaries = [b for b in boundaries if 0 < b < total]

    # F2: 逐边界分类转场类型；随后合并相邻 fade（同一次淡出+淡入被检出两次）
    transitions = [{
        "frame": b,
        "type": classify_transition(cap, b, total, window),
    } for b in boundaries]
    transitions = merge_adjacent_fades(cap, transitions, window)
    cap.release()
    boundaries = [t["frame"] for t in transitions]
    for t in transitions:
        t["time"] = round(t["frame"] / fps, 3)

    # 由边界构建镜头列表（end_frame 含）
    shots = []
    starts = [0] + boundaries
    ends = [b - 1 for b in boundaries] + [total - 1]
    for i, (s, e) in enumerate(zip(starts, ends)):
        shots.append({
            "index": i,
            "start_frame": s,
            "end_frame": e,
            "start_time": round(s / fps, 3),
            "end_time": round(e / fps, 3),
            "duration": round((e - s + 1) / fps, 3),
        })

    result = {
        "video": video_path,
        "fps": fps,
        "total_frames": total,
        "duration": round(total / fps, 3),
        "shots": shots,
        "transitions": transitions,
        "rhythm": rhythm_stats(shots),
        "detected_shot_count": len(shots),  # F4: 边界数 + 1
    }
    if expect_shots is not None:
        # D4 纯规则打分：实际镜头数 vs 指令指定数
        result["d4_shot_count"] = d4_score(len(shots), expect_shots)
    return result


def main():
    p = argparse.ArgumentParser(description="镜头边界检测(F1)/转场分类(F2)/节奏统计(F3)/镜头计数(F4)")
    p.add_argument("video", help="输入视频路径")
    p.add_argument("--step", type=int, default=2, help="粗扫采样步长（帧），默认 2；边界仍回到逐帧精度")
    p.add_argument("--window", type=int, default=30, help="F2 转场分析的前后窗口帧数，默认 30")
    p.add_argument("--expect-shots", type=int, default=None,
                   help="可选：指令⑤指定的镜头数；传入后直接输出 D4 规则分")
    p.add_argument("--output", help="可选：把结果 JSON 写入该文件")
    args = p.parse_args()

    try:
        result = detect_shots(args.video, step=args.step, window=args.window,
                              expect_shots=args.expect_shots)
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
