#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""visual_analysis.py — 视觉客观分析（cine-eval F5~F9）。

F5 亮度/色温/对比度/明暗比/调性/光质/光源方向（3x3 网格最亮区）
F6 K-means 调色板（k=5，每秒 1 帧采样像素，随机子采样防慢）+ 主色调 hue
F7 相邻镜头中间帧 H-S 直方图相关性（需 --shots 传入 shot_detect.py 的 JSON）
F8 8x8 Laplacian 方差清晰度网格 → 焦点区域 + 全画面清晰度分布是否均匀
F9 Farneback 稠密光流（320px 宽、每 --stride 帧一对）→ 方向/归一化速度/连续性

输出单个 JSON，键按公式分组:
lighting / palette / shot_color_continuity / focus / camera_motion

用法:
    python visual_analysis.py video.mp4
    python visual_analysis.py video.mp4 --shots shots.json --output visual.json
"""
import argparse
import json
import sys
from collections import Counter

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# F9 归一化速度档位阈值（速度 = 全帧平均光流位移 / 画面对角线）
SPEED_BINS = [(0.001, "static"), (0.005, "very_slow"), (0.02, "slow"),
              (0.05, "medium"), (float("inf"), "fast")]
STATIC_SPEED = 0.001  # 低于该归一化速度视为静止
DIRECTION_NAMES_3x3 = [["top_left", "top", "top_right"],
                       ["left", "center", "right"],
                       ["bottom_left", "bottom", "bottom_right"]]


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total


def speed_level(speed):
    for thr, name in SPEED_BINS:
        if speed < thr:
            return name
    return "fast"


def hs_hist(frame_bgr):
    """F1/F7 共用的 H-S 二维直方图（H 50 bin / S 60 bin，归一化）。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def iter_sampled_frames(cap, total, fps, interval):
    """每 interval 秒采 1 帧，yield (时间秒, 帧图像)。"""
    n = max(int((total / fps) / interval), 1)
    for i in range(n):
        fidx = min(int(round(i * interval * fps)), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if ok:
            yield i * interval, frame


# ---------------- F5 亮度/色温/对比度/光源方向 ----------------

def analyze_lighting(video_path, interval=1.0):
    cap, fps, total = open_video(video_path)
    per_frame = []
    grid_sum = np.zeros((3, 3))
    sums = dict(brightness=0.0, r_b=0.0, g_contrast=0.0, l_contrast=0.0, ratio=0.0)
    for t, frame in iter_sampled_frames(cap, total, fps, interval):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2].astype(np.float64)
        b, g, r = cv2.split(frame.astype(np.float64))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        brightness = float(v.mean() / 255.0)                    # F5 平均亮度
        r_b_ratio = float(r.mean() / (b.mean() + 1e-6))         # F5 色温 R/B 比
        temperature = "warm" if r_b_ratio > 1.3 else ("cool" if r_b_ratio < 0.8 else "neutral")
        g_contrast = float((gray.max() - gray.min()) / 255.0)   # 全局对比度
        l_contrast = float(gray.std() / 255.0)                  # 局部对比度
        flat = np.sort(gray, axis=None)
        k = max(int(flat.size * 0.2), 1)
        light_ratio = float(flat[-k:].mean() / (flat[:k].mean() + 1e-6))  # 明暗比
        tonality = "high_key" if brightness > 0.6 else ("low_key" if brightness < 0.4 else "mid_key")

        # 光源方向: 3x3 网格平均亮度，最亮区域即大致光源方向
        gh, gw = gray.shape[0] // 3, gray.shape[1] // 3
        grid = np.array([[gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].mean()
                          for j in range(3)] for i in range(3)])
        grid_sum += grid

        sums["brightness"] += brightness
        sums["r_b"] += r_b_ratio
        sums["g_contrast"] += g_contrast
        sums["l_contrast"] += l_contrast
        sums["ratio"] += light_ratio
        per_frame.append({"time": round(t, 2), "brightness": round(brightness, 4),
                          "r_b_ratio": round(r_b_ratio, 3), "color_temperature": temperature,
                          "tonality": tonality})
    cap.release()
    if not per_frame:
        raise RuntimeError("未能采到任何帧")

    n = len(per_frame)
    brightness = sums["brightness"] / n
    r_b_ratio = sums["r_b"] / n
    light_ratio = sums["ratio"] / n
    dir_idx = np.unravel_index(grid_sum.argmax(), grid_sum.shape)
    return {
        "mean_brightness": round(brightness, 4),
        "color_temperature": "warm" if r_b_ratio > 1.3 else ("cool" if r_b_ratio < 0.8 else "neutral"),
        "r_b_ratio": round(r_b_ratio, 3),
        "global_contrast": round(sums["g_contrast"] / n, 4),
        "local_contrast": round(sums["l_contrast"] / n, 4),
        "light_dark_ratio": round(light_ratio, 3),
        "tonality": "high_key" if brightness > 0.6 else ("low_key" if brightness < 0.4 else "mid_key"),
        "light_quality": "hard" if light_ratio > 5 else ("soft" if light_ratio < 3 else "mixed"),
        "light_direction": DIRECTION_NAMES_3x3[dir_idx[0]][dir_idx[1]],
        "per_frame": per_frame,
    }


# ---------------- F6 调色板与主色调 ----------------

def analyze_palette(video_path, interval=1.0, k=5, max_pixels=30000, seed=42):
    cap, fps, total = open_video(video_path)
    rng = np.random.default_rng(seed)
    pixels, hues = [], []
    for t, frame in iter_sampled_frames(cap, total, fps, interval):
        h, w = frame.shape[:2]
        scale = 160.0 / max(h, w)  # 缩小后再采样像素，控制计算量
        small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame
        pixels.append(small.reshape(-1, 3))
        hues.append(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 0].ravel())
    cap.release()
    if not pixels:
        raise RuntimeError("未能采到任何帧")

    px = np.concatenate(pixels).astype(np.float32)
    if len(px) > max_pixels:  # 随机子采样防慢
        px = px[rng.choice(len(px), max_pixels, replace=False)]
    k_eff = min(k, len(px))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, labels, centers = cv2.kmeans(px, k_eff, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k_eff).astype(np.float64)
    palette = []
    for i in np.argsort(-counts):
        cb, cg, cr = (float(v) for v in centers[i])  # OpenCV 内部为 BGR
        palette.append({"rgb": [round(cr), round(cg), round(cb)],
                        "hex": "#%02x%02x%02x" % (round(cr), round(cg), round(cb)),
                        "weight": round(float(counts[i] / counts.sum()), 4)})

    hue_hist = np.bincount(np.concatenate(hues).astype(np.int64), minlength=180)
    dominant_hue = int(hue_hist.argmax())
    return {
        "k": k_eff,
        "palette": palette,
        "dominant_hue": dominant_hue,                # OpenCV hue 0~179
        "dominant_hue_deg": dominant_hue * 2,        # 换算成色相角度 0~358
        "dominant_hue_share": round(float(hue_hist[dominant_hue] / hue_hist.sum()), 4),
    }


def per_shot_dominant_hue(video_path, shots):
    """有 --shots 时附加：每个镜头中间帧的主 hue，便于对比镜头间色调差异。"""
    cap, _, _ = open_video(video_path)
    out = []
    for s in shots:
        mid = (s["start_frame"] + s["end_frame"]) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if not ok:
            continue
        h = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 0].ravel()
        hist = np.bincount(h.astype(np.int64), minlength=180)
        out.append({"shot_index": s["index"], "mid_frame": mid,
                    "dominant_hue": int(hist.argmax()),
                    "dominant_hue_deg": int(hist.argmax()) * 2})
    cap.release()
    return out


# ---------------- F7 镜头间色彩连续性 ----------------

def analyze_shot_continuity(video_path, shots):
    cap, _, _ = open_video(video_path)
    mid_frames, exposures = [], []
    for s in shots:
        mid = (s["start_frame"] + s["end_frame"]) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        mid_frames.append(frame if ok else None)
        exposures.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) / 255.0 if ok else None)
    cap.release()

    counts = {"continuous": 0, "acceptable": 0, "inconsistent": 0}
    pairs = []
    for i in range(len(shots) - 1):
        fa, fb = mid_frames[i], mid_frames[i + 1]
        if fa is None or fb is None:
            continue
        corr = float(cv2.compareHist(hs_hist(fa), hs_hist(fb), cv2.HISTCMP_CORREL))
        exp_diff = abs(exposures[i] - exposures[i + 1])
        # F7: >0.8 连续；0.5~0.8 可接受；<0.5 不一致（曝光差 >0.1 佐证）
        assessment = "continuous" if corr > 0.8 else ("acceptable" if corr >= 0.5 else "inconsistent")
        counts[assessment] += 1
        pairs.append({"shots": [shots[i]["index"], shots[i + 1]["index"]],
                      "corr": round(corr, 4),
                      "exposure_diff": round(exp_diff, 4),
                      "exposure_jump": bool(exp_diff > 0.1),
                      "assessment": assessment})
    return {"pairs": pairs, "summary": counts}


# ---------------- F8 清晰度与焦点 ----------------

def analyze_focus(video_path, interval=1.0, grid_n=8):
    cap, fps, total = open_video(video_path)
    per_frame, cells = [], []
    for t, frame in iter_sampled_frames(cap, total, fps, interval):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grid = np.zeros((grid_n, grid_n))
        for i, row in enumerate(np.array_split(gray, grid_n, axis=0)):
            for j, cell in enumerate(np.array_split(row, grid_n, axis=1)):
                if cell.size:
                    grid[i, j] = cv2.Laplacian(cell, cv2.CV_64F).var()  # F8 清晰度
        fr, fc = (int(v) for v in np.unravel_index(grid.argmax(), grid.shape))
        sharp_cv = float(grid.std() / (grid.mean() + 1e-9))
        per_frame.append({"time": round(t, 2), "focus_cell": [fr, fc],
                          "sharpness_cv": round(sharp_cv, 3),
                          "mean_sharpness": round(float(grid.mean()), 2)})
        cells.append((fr, fc))
    cap.release()
    if not per_frame:
        raise RuntimeError("未能采到任何帧")

    mode_cell = Counter(cells).most_common(1)[0][0]
    mean_cv = float(np.mean([f["sharpness_cv"] for f in per_frame]))
    uniform = mean_cv < 0.5  # 全画面清晰度分布均匀 → 景深深；反之可能浅景深
    return {
        "grid": f"{grid_n}x{grid_n}",
        "focus_region": {"cell": [mode_cell[0], mode_cell[1]],
                         "row": mode_cell[0], "col": mode_cell[1]},
        "sharpness_cv": round(mean_cv, 3),
        "uniform": uniform,
        "assessment": "even_sharpness(景深较深)" if uniform else "uneven_sharpness(可能浅景深)",
        "per_frame": per_frame,
    }


# ---------------- F9 运镜参数（稠密光流） ----------------

def flow_pairs(video_path, stride=2, width=320):
    """每 stride 帧取一对，缩放到 width px 宽后算 Farneback 稠密光流。

    返回 [{time, speed, angle, direction}, ...]；speed 为归一化速度。
    """
    cap, fps, _ = open_video(video_path)
    pairs = []
    prev_gray = None
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % stride == 0:
            h, w = frame.shape[:2]
            dim = (width, max(1, round(h * width / w)))
            gray = cv2.cvtColor(cv2.resize(frame, dim, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                fx = float(flow[..., 0].mean())   # F9: 全帧光流取均值
                fy = float(flow[..., 1].mean())
                diag = float(np.hypot(*dim))
                speed = float(np.hypot(fx, fy) / diag)
                angle = float(np.degrees(np.arctan2(fy, fx)))
                # F9 方向判定: |fx| > 2|fy| 水平；|fy| > 2|fx| 垂直；否则对角
                if speed < STATIC_SPEED:
                    direction = "static"
                elif abs(fx) > 2 * abs(fy):
                    direction = "horizontal"
                elif abs(fy) > 2 * abs(fx):
                    direction = "vertical"
                else:
                    direction = "diagonal"
                pairs.append({"time": round(fidx / fps, 3), "speed": speed,
                              "angle": angle, "direction": direction})
            prev_gray = gray
        fidx += 1
    cap.release()
    return pairs


def summarize_pairs(pairs):
    """对一组光流对做 F9 聚合：平均速度/档位/主方向/连续性。"""
    if not pairs:
        return {"pair_count": 0}
    # 连续性: 相邻光流对方向角差 > 90° 记一次突变（双方都非静止才记，静止帧角向是噪声）
    abrupt = 0
    for a, b in zip(pairs[:-1], pairs[1:]):
        if a["speed"] >= STATIC_SPEED and b["speed"] >= STATIC_SPEED:
            d = abs(b["angle"] - a["angle"]) % 360.0
            if d > 180:
                d = 360 - d
            if d > 90:
                abrupt += 1
    transitions = len(pairs) - 1
    continuity = 1.0 - abrupt / transitions if transitions > 0 else 1.0
    mean_speed = float(np.mean([p["speed"] for p in pairs]))
    dir_counts = Counter(p["direction"] for p in pairs)
    return {
        "pair_count": len(pairs),
        "mean_normalized_speed": round(mean_speed, 6),
        "speed_level": speed_level(mean_speed),
        "dominant_direction": dir_counts.most_common(1)[0][0],
        "direction_distribution": dict(dir_counts),
        "continuity": round(continuity, 4),
        "abrupt_count": abrupt,
    }


def analyze_motion(video_path, stride=2, width=320, shots=None):
    pairs = flow_pairs(video_path, stride=stride, width=width)
    result = {"stride": stride, "flow_width": width}
    result.update(summarize_pairs(pairs))
    if shots:  # 有镜头列表时附加逐镜头运镜，便于逐镜头与摄影指令④对齐
        per_shot = []
        for s in shots:
            sp = [p for p in pairs if s["start_time"] <= p["time"] < s["end_time"]]
            summary = summarize_pairs(sp)
            summary["shot_index"] = s["index"]
            summary["start_time"] = s["start_time"]
            summary["end_time"] = s["end_time"]
            per_shot.append(summary)
        result["per_shot"] = per_shot
    return result


# ---------------- 主流程 ----------------

def load_shots(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shots = data.get("shots") if isinstance(data, dict) else data
    if not shots:
        raise RuntimeError(f"{path} 中没有镜头列表（期望 shot_detect.py 的输出）")
    return shots


def main():
    p = argparse.ArgumentParser(description="视觉客观分析：F5 光影 / F6 调色板 / F7 镜头间色彩连续性 / F8 焦点 / F9 运镜")
    p.add_argument("video", help="输入视频路径")
    p.add_argument("--shots", help="可选：shot_detect.py 的 JSON 输出文件（启用 F7 及逐镜头统计）")
    p.add_argument("--interval", type=float, default=1.0, help="F5/F6/F8 采样间隔秒数，默认 1.0")
    p.add_argument("--stride", type=int, default=2, help="F9 光流帧对步长（帧），默认 2")
    p.add_argument("--output", help="可选：把结果 JSON 写入该文件")
    args = p.parse_args()

    try:
        shots = load_shots(args.shots) if args.shots else None
        result = {
            "video": args.video,
            "lighting": analyze_lighting(args.video, interval=args.interval),          # F5
            "palette": analyze_palette(args.video, interval=args.interval),            # F6
            "shot_color_continuity": (analyze_shot_continuity(args.video, shots)       # F7
                                      if shots else None),
            "focus": analyze_focus(args.video, interval=args.interval),                # F8
            "camera_motion": analyze_motion(args.video, stride=args.stride, shots=shots),  # F9
        }
        if shots:
            result["palette"]["per_shot_dominant_hue"] = per_shot_dominant_hue(args.video, shots)
        else:
            result["note"] = "未提供 --shots：跳过 F7 镜头间连续性及逐镜头统计"
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
