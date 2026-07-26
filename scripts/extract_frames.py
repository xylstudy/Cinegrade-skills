#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract_frames.py — 视频抽帧（cine-eval skill 抽帧约定实现）。

均匀抽帧：默认每秒 1 帧、上限 30 帧；帧缩放到最长边 640px 存 JPEG，同时写
manifest.json（每帧的文件名 + 时间戳秒）。摘要 JSON 打印到 stdout（--output 可写文件）。

镜头关键帧（--shots）：传入 shot_detect.py 的 JSON 输出后，按抽帧约定为每个
镜头抽取代表帧——中间帧必取，时长 >10 秒的长镜头加取 1/3、2/3 处帧。文件名
带镜头号与位置，manifest.json 中记入 "shot_keyframes" 段，供 PD4/C4/D3 等
跨镜头维度对比使用。

用法:
    python extract_frames.py video.mp4 --outdir frames/
    python extract_frames.py video.mp4 --outdir frames/ --interval 0.5 --max-frames 60
    python extract_frames.py video.mp4 --outdir frames/ --shots shots.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LONGEST_EDGE = 640       # 输出帧最长边像素
LONG_SHOT_SEC = 10.0     # 镜头时长超过该秒数 → 加取 1/3、2/3 处帧


def _resize_and_save(frame, path):
    """缩放到最长边 LONGEST_EDGE（只缩小不放大）并存 JPEG。"""
    h, w = frame.shape[:2]
    scale = LONGEST_EDGE / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])


def extract_frames(video_path, outdir, interval=1.0, max_frames=30):
    """按 interval 秒均匀抽帧，存 JPEG + manifest.json，返回 (manifest路径, manifest)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0  # 个别容器读不到 fps 时回退
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if total > 0 else 0.0

    # 候选时间点: 0, interval, 2*interval, ... < duration
    times = np.arange(0.0, max(duration, 1e-6), interval)
    if len(times) == 0:
        times = np.array([0.0])
    if len(times) > max_frames:
        # 超上限 → 在全片范围内重新均匀取 max_frames 个时间点
        times = np.linspace(0.0, max(duration - 1.0 / fps, 0.0), max_frames)

    os.makedirs(outdir, exist_ok=True)
    entries = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        fname = f"frame_{i:03d}_{t:06.2f}s.jpg"
        _resize_and_save(frame, os.path.join(outdir, fname))
        entries.append({"index": i, "file": fname, "timestamp": round(float(t), 3)})
    cap.release()

    manifest = {
        "video": os.path.abspath(video_path),
        "fps": fps,
        "total_frames": total,
        "duration": round(duration, 3),
        "interval": interval,
        "max_frames": max_frames,
        "frame_count": len(entries),
        "frames": entries,
    }
    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest_path, manifest


def load_shots(path):
    """读 shot_detect.py 的 JSON 输出，返回镜头列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shots = data.get("shots") if isinstance(data, dict) else data
    if not shots:
        raise RuntimeError(f"{path} 中没有镜头列表（期望 shot_detect.py 的输出）")
    return shots


def extract_shot_keyframes(video_path, shots, outdir):
    """按抽帧约定取镜头关键帧：中间帧必取；时长 >10s 的长镜头加取 1/3、2/3 处。

    返回关键帧条目列表（含 shot_index / position / file / timestamp）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(outdir, exist_ok=True)

    entries = []
    for shot in shots:
        idx = shot.get("index", len(entries))
        s_frame, e_frame = int(shot["start_frame"]), int(shot["end_frame"])
        span = e_frame - s_frame
        duration = span / fps if fps else 0.0
        # 位置: (名称, 相对偏移比例)；长镜头取三点，否则只取中间帧
        positions = [("mid", 0.5)]
        if duration > LONG_SHOT_SEC:
            positions = [("third_1", 1.0 / 3.0), ("mid", 0.5), ("third_2", 2.0 / 3.0)]
        for name, ratio in positions:
            fidx = min(max(s_frame + int(round(span * ratio)), 0), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            t = fidx / fps
            fname = f"shot_{idx:03d}_{name}_{t:06.2f}s.jpg"
            _resize_and_save(frame, os.path.join(outdir, fname))
            entries.append({"shot_index": idx, "position": name, "file": fname,
                            "timestamp": round(float(t), 3)})
    cap.release()
    return entries


def main():
    p = argparse.ArgumentParser(description="视频抽帧：均匀抽帧（默认每秒 1 帧、上限 30 帧）+ 可选镜头关键帧（--shots）")
    p.add_argument("video", help="输入视频路径")
    p.add_argument("--outdir", required=True, help="帧图片与 manifest.json 的输出目录")
    p.add_argument("--interval", type=float, default=1.0, help="抽帧间隔秒数，默认 1.0")
    p.add_argument("--max-frames", type=int, default=30, help="帧数上限，默认 30")
    p.add_argument("--shots", help="可选：shot_detect.py 的 JSON 输出文件，启用镜头关键帧抽取")
    p.add_argument("--output", help="可选：把 stdout 摘要 JSON 同时写入该文件")
    args = p.parse_args()

    try:
        manifest_path, manifest = extract_frames(args.video, args.outdir,
                                                 args.interval, args.max_frames)
        keyframes = []
        if args.shots:
            keyframes = extract_shot_keyframes(args.video, load_shots(args.shots), args.outdir)
            manifest["shot_keyframes"] = keyframes
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)

    summary = {
        "manifest_path": os.path.abspath(manifest_path),
        "frame_count": manifest["frame_count"],
        "shot_keyframe_count": len(keyframes),
        "outdir": os.path.abspath(args.outdir),
        "duration": manifest["duration"],
        "fps": manifest["fps"],
        "interval": args.interval,
        "max_frames": args.max_frames,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
