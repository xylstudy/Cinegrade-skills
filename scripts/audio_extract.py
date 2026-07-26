#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audio_extract.py — 音频提取与基础统计（cine-eval F11）。

用 subprocess 调系统 ffmpeg 把视频音频转成 16kHz 单声道 wav，
再用标准库 wave + numpy 计算 RMS 音量与静音段占比
（帧级 RMS 低于全局 RMS 的 10% 记静音，20ms 一帧）。

ffmpeg 不存在 / 提取失败时打印明确错误并以非零码退出。

用法:
    python audio_extract.py video.mp4
    python audio_extract.py video.mp4 --outdir audio/ --output audio.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

FRAME_MS = 20          # 静音判定的帧长（毫秒）
SILENCE_RATIO = 0.1    # 帧 RMS < 全局 RMS * 0.1 → 静音帧


def fail(msg, code=1):
    """打印明确错误（stderr 给人看，stdout JSON 给程序读）并以非零码退出。"""
    print(f"错误: {msg}", file=sys.stderr)
    print(json.dumps({"error": msg}, ensure_ascii=False))
    sys.exit(code)


def extract_wav(video_path, wav_path):
    """ffmpeg -i video -vn -ac 1 -ar 16000 out.wav（16bit PCM）。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        fail("未找到 ffmpeg。请先安装 ffmpeg 并加入 PATH"
             "（Windows 可用 winget install ffmpeg 或下载 builds；音频维度可选，跳过不影响其他维度）。", 3)
    cmd = [ffmpeg, "-y", "-i", video_path, "-vn",
           "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", wav_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        fail("无法执行 ffmpeg（已找到路径但启动失败），请检查安装。", 3)
    except subprocess.TimeoutExpired:
        fail("ffmpeg 执行超时（300s）。", 4)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        hint = ""
        if any("does not contain any stream" in l or "Stream map" in l for l in tail):
            hint = "（该视频可能没有音轨）"
        fail(f"ffmpeg 提取音频失败{hint}: " + " | ".join(tail), 4)
    return wav_path


def analyze_wav(wav_path, frame_ms=FRAME_MS):
    """读 16bit PCM wav，算全局 RMS、时长、静音段占比（frame_ms 毫秒一帧）。"""
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sampwidth != 2:
        raise RuntimeError(f"不支持的采样位宽: {sampwidth * 8}bit（期望 16bit PCM）")
    data = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    x = data.astype(np.float64) / 32768.0  # 归一化到 [-1, 1]

    duration = len(x) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0

    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(x) // frame_len
    if rms <= 1e-8:
        silence_ratio = 1.0  # 全程数字静音
    elif n_frames == 0:
        silence_ratio = 0.0
    else:
        frames = x[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        silence_ratio = float(np.mean(frame_rms < SILENCE_RATIO * rms))
    return {
        "duration": round(duration, 3),
        "rms": round(rms, 6),
        "silence_ratio": round(silence_ratio, 4),
        "sample_rate": sr,
        "frame_ms": frame_ms,
    }


def main():
    p = argparse.ArgumentParser(description="ffmpeg 提取 16kHz 单声道 wav + RMS/静音比统计（F11）")
    p.add_argument("video", help="输入视频路径")
    p.add_argument("--outdir", help="wav 输出目录；不传则写到系统临时目录")
    p.add_argument("--output", help="可选：把结果 JSON 写入该文件")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        fail(f"视频文件不存在: {args.video}", 2)
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.video))[0]
        wav_path = os.path.join(args.outdir, f"{stem}_16k_mono.wav")
    else:
        fd, wav_path = tempfile.mkstemp(prefix="cineeval_", suffix=".wav")
        os.close(fd)

    extract_wav(args.video, wav_path)
    try:
        stats = analyze_wav(wav_path)
    except Exception as e:
        fail(f"wav 分析失败: {e}", 5)

    result = {"duration": stats["duration"], "rms": stats["rms"],
              "silence_ratio": stats["silence_ratio"], "wav_path": os.path.abspath(wav_path)}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
