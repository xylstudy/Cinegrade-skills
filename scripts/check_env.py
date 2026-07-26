#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_env.py — cine-eval 运行环境自检。

检查核心依赖（Python / OpenCV / NumPy）与可选依赖（ffmpeg，仅音频维度需要），
打印 JSON 结果。核心依赖缺失时以退出码 1 结束；仅缺 ffmpeg 时退出码 0
（S1~S4 标记 N/A 即可，不影响其他维度）。

用法:
    python check_env.py
"""
import json
import shutil
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

INSTALL_HINT = "pip install opencv-python numpy"


def check_import(name, import_as=None):
    """尝试导入模块并取版本号。"""
    try:
        mod = __import__(import_as or name)
        return {"ok": True, "version": getattr(mod, "__version__", "unknown")}
    except ImportError:
        return {"ok": False, "install": INSTALL_HINT}


def check_opencv_funcs():
    """确认脚本用到的 OpenCV 关键函数可用。"""
    try:
        import cv2
        funcs = ["VideoCapture", "calcHist", "compareHist", "kmeans",
                 "Laplacian", "calcOpticalFlowFarneback"]
        missing = [f for f in funcs if not hasattr(cv2, f)]
        return {"ok": not missing, "missing": missing}
    except ImportError:
        return {"ok": False, "missing": ["cv2 未安装"]}


def check_ffmpeg():
    """ffmpeg 仅需在 PATH 中（音频维度用，命令行调用）。"""
    exe = shutil.which("ffmpeg")
    if not exe:
        return {"ok": False,
                "note": "未找到 ffmpeg：S1~S4 声音维度将标记 N/A，其余维度不受影响。"
                        "安装：Windows `winget install ffmpeg`；macOS `brew install ffmpeg`；"
                        "Debian/Ubuntu `apt install ffmpeg`"}
    try:
        proc = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10)
        first = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
        return {"ok": True, "path": exe, "version": first.strip()}
    except Exception as e:
        return {"ok": False, "path": exe, "note": f"ffmpeg 存在但无法执行: {e}"}


def main():
    cv2_status = check_import("opencv", "cv2")
    np_status = check_import("numpy")
    core_ok = cv2_status["ok"] and np_status["ok"] and check_opencv_funcs()["ok"]

    result = {
        "python": {"version": sys.version.split()[0]},
        "opencv": cv2_status,
        "opencv_functions": check_opencv_funcs(),
        "numpy": np_status,
        "ffmpeg": check_ffmpeg(),
        "core_ok": core_ok,
        "audio_available": check_ffmpeg()["ok"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not core_ok:
        print(f"\n核心依赖缺失，请先执行: {INSTALL_HINT}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
