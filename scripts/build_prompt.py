#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_prompt.py — 按维度生成具体评估 Prompt（cine-eval）。

把 SKILL.md「LLM 评估要点」的三层结构（全局契约 / 路径模板 / 维度规格）固化：
内置 23 维注册表（路径、指令模块、规则数据键、输入材料、检查点），
读取指令原文、脚本产物 JSON 与抽帧 manifest，输出该维度的最终具体 Prompt。

每个维度一次调用，不同维度产出的 Prompt 在以下方面各不相同：
- 路径模板（纯LLM 激发型 / 混合 融合型 / 规则为主 抑制型）
- 指令模块摘录（只含该维度相关的 ①~⑦ 原文）
- 嵌入的规则数据（只含该维度相关 JSON 键）
- 输入材料清单（均匀帧 / 镜头关键帧 / 转场窗口帧 / 音频 / 无）
- 维度专属检查点

用法:
    python build_prompt.py A1 --script script.txt
    python build_prompt.py B1 --script script.txt --visual visual.json --shots shots.json \
        --manifest frames/manifest.json
    python build_prompt.py --all --script script.txt --outdir prompts/
    python build_prompt.py S2 --script script.txt --audio audio.json --json
"""
import argparse
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 23 维注册表（SKILL.md「23 维总表与评分路径」+「各角色检查点」的机器可读版）
# path: pure_llm / mixed / rule_first
# modules: 七模块指令键（scene/character/narrative/camera/editing/sound/mood）
# rule_data: [(来源文件参数名, JSON 内 dotted 键), ...]
# materials: uniform / shot_keyframes / shot_pairs / transition_window /
#            audio / audio_uniform / audio_character / stats_only / none
# ---------------------------------------------------------------------------
DIMENSIONS = {
    "A1": {"role": "导演", "name": "叙事完整性", "path": "pure_llm",
           "modules": ["narrative", "mood"], "rule_data": [], "materials": "uniform",
           "checkpoints": "① 用 1~2 句话描述片段中发生了什么事件；② 事件是否有可理解的"
           "起因/铺垫；③ 是否有明确的结果或走向；④ 没看过原片的陌生观众仅凭此片段能否"
           "看懂基本事件流；⑤ 与③叙事指令逐项对齐：要求的事件是否都呈现、有无遗漏或多余"},
    "A2": {"role": "导演", "name": "场景逻辑一致性", "path": "pure_llm",
           "modules": ["scene", "narrative"], "rule_data": [], "materials": "uniform",
           "checkpoints": "① 物品是否凭空出现或消失（逐帧核对主要道具）；② 空间关系"
           "（左右、前后、室内外）前后是否一致；③ 与①场景设定/③叙事指令中的空间描述对齐"},
    "A3": {"role": "导演", "name": "情绪节奏弧", "path": "pure_llm",
           "modules": ["editing", "mood"], "rule_data": [], "materials": "uniform",
           "checkpoints": "① 先用 3 个形容词描述主观感受；② 全片情绪是否一致；③ 情绪"
           "有无起伏弧线（铺垫→推进→收束）；④ 与⑦情绪风格、⑤剪辑指令的节奏要求对齐"},
    "PD1": {"role": "美术指导", "name": "场景与道具设计", "path": "pure_llm",
            "modules": ["scene"], "rule_data": [], "materials": "uniform",
            "checkpoints": "① 空间布局是否合理且服务叙事；② 道具丰富度与陈设细节；"
            "③ 类型感（能否一眼看出时代/题材）；④ 与①场景设定对齐"},
    "PD2": {"role": "美术指导", "name": "色彩方案", "path": "mixed",
            "modules": ["scene", "mood"],
            "rule_data": [("visual", "palette"), ("visual", "lighting.color_temperature"),
                          ("visual", "lighting.r_b_ratio")],
            "materials": "uniform",
            "checkpoints": "① 调色板主色与占比是否服务于⑦指定情绪；② 色温（暖/冷/中性）"
            "与情绪意图是否匹配；③ 配色与①场景设定的时代/氛围是否协调"},
    "PD3": {"role": "美术指导", "name": "风格自洽", "path": "pure_llm",
            "modules": ["scene", "mood"], "rule_data": [], "materials": "uniform",
            "checkpoints": "① 场景/道具/服装/色彩是否在同一风格体系内；② 有无风格突兀"
            "的元素（如写实场景中出现卡通物体）；③ 与⑦情绪风格对齐"},
    "PD4": {"role": "美术指导", "name": "场景跨镜头一致", "path": "rule_first",
            "modules": ["scene"],
            "rule_data": [("visual", "shot_color_continuity")],
            "materials": "shot_keyframes",
            "checkpoints": "核验规则标记的每个不一致镜头对：并排看两镜头关键帧，判断不一致"
            "是否属于合理叙事切换（闪回/换场/时间跳跃）；不合理的每处 -1 并给出时间戳"},
    "B1": {"role": "摄影师", "name": "运镜轨迹", "path": "mixed",
           "modules": ["camera"],
           "rule_data": [("visual", "camera_motion")],
           "materials": "uniform",
           "checkpoints": "把光流数据描述的运镜（类型/方向/速度/连续性）与④摄影指定逐项"
           "对比：① 运镜类型（推/拉/摇/移/跟/固定）是否一致；② 方向是否一致；③ 速度档位"
           "是否一致；④ 有无未指定的运镜或抖动"},
    "B2": {"role": "摄影师", "name": "焦点与景深", "path": "mixed",
           "modules": ["camera"],
           "rule_data": [("visual", "focus")],
           "materials": "uniform",
           "checkpoints": "① 焦点区域（清晰度最高的网格）是否落在④指定的主体上；② 虚实"
           "关系（浅景深/深景深）与④是否一致；③ 指定的焦点转移（Rack Focus）有无执行、"
           "时机是否对"},
    "B3": {"role": "摄影师", "name": "构图质量", "path": "pure_llm",
           "modules": ["camera"], "rule_data": [], "materials": "uniform",
           "checkpoints": "① 主体位置与画面权重是否服务叙事；② 层次感（前/中/后景）；"
           "③ 视觉平衡与引导性；④ 与④摄影指定的构图要求对齐。注意：好构图可以打破规则，"
           "以视觉意图为准，不要机械套三分法"},
    "B4": {"role": "摄影师", "name": "光影氛围", "path": "mixed",
           "modules": ["camera", "mood"],
           "rule_data": [("visual", "lighting")],
           "materials": "uniform",
           "checkpoints": "① 调性（高/低/中间调）与⑦情绪是否匹配；② 色温与情绪；③ 明暗"
           "对比（光质硬/柔）是否服务叙事；④ 光源方向是否有逻辑（窗外光/灯光/月光）；"
           "⑤ 与④摄影指定的光影要求对齐"},
    "C1": {"role": "演员指导", "name": "行为自然度", "path": "pure_llm",
           "modules": ["character", "narrative"], "rule_data": [], "materials": "uniform",
           "checkpoints": "① 动作起止是否自然、有无机械感；② 动作速度是否符合人体规律；"
           "③ 有无反物理现象（穿模/悬浮/关节异常/重心错误）；④ 与②角色描述的行为设定对齐"},
    "C2": {"role": "演员指导", "name": "情绪表达", "path": "pure_llm",
           "modules": ["character", "mood"], "rule_data": [], "materials": "uniform",
           "checkpoints": "① 表情是否传递了②/⑦指定的情绪、是否到位；② 情绪有无层次变化"
           "还是全程单一；③ 肢体语言是否配合表情；④ 情绪转变是否突兀"},
    "C3": {"role": "演员指导", "name": "空间互动", "path": "pure_llm",
           "modules": ["character", "narrative"], "rule_data": [], "materials": "uniform",
           "checkpoints": "① 角色间距离与场景情绪是否匹配；② 站位有无层次（前后/遮挡"
           "关系）；③ 互动（对视/递物/肢体接触）是否自然；④ 有无穿模或位置瞬移"},
    "C4": {"role": "演员指导", "name": "演员跨镜头一致", "path": "pure_llm",
           "modules": ["character"], "rule_data": [], "materials": "shot_keyframes",
           "checkpoints": "并排对比各镜头关键帧中的同一角色：① 外观/发型/服装是否连续；"
           "② 伤妆/污渍等状态是否连续；③ 与②角色描述是否一致。合理的换装/时间跳跃除外，"
           "需注明判断依据"},
    "D1": {"role": "剪辑师", "name": "转场方式", "path": "rule_first",
           "modules": ["editing"],
           "rule_data": [("shots", "transitions")],
           "materials": "transition_window",
           "checkpoints": "核验规则分类的每个转场类型与⑤剪辑指令是否一致；规则标 unknown "
           "或疑似匹配剪辑/跳切的，单独看前后窗口帧判断并说明"},
    "D2": {"role": "剪辑师", "name": "切换节奏", "path": "rule_first",
           "modules": ["editing"],
           "rule_data": [("shots", "rhythm")],
           "materials": "stats_only",
           "checkpoints": "核验节奏统计（镜头数/平均时长/标准差/模式分类）与⑤剪辑指令的"
           "节奏要求是否对齐；指令写「长镜头/一镜到底」而检出多镜头，或指令写「快切」而"
           "平均时长远超 2s，均属矛盾"},
    "D3": {"role": "剪辑师", "name": "轴线规则", "path": "pure_llm",
           "modules": ["scene", "editing"], "rule_data": [], "materials": "shot_pairs",
           "checkpoints": "逐对检查相邻镜头关键帧：① 转场后人物视线方向/运动方向/空间"
           "方位是否一致（180° 轴线）；② 若越轴，是否造成空间理解混乱；③ 越轴是否明显"
           "是故意艺术选择（需说明依据）"},
    "D4": {"role": "剪辑师", "name": "镜头数量", "path": "rule_first",
           "modules": ["editing"],
           "rule_data": [("shots", "detected_shot_count"), ("shots", "d4_shot_count")],
           "materials": "none",
           "checkpoints": "纯规则维度：shot_detect.py --expect-shots 已直接给出分数时，"
           "无需 LLM 调用"},
    "S1": {"role": "声音设计师", "name": "对白质量", "path": "pure_llm",
           "modules": ["sound"], "rule_data": [], "materials": "audio_character",
           "checkpoints": "① 对白是否清晰可懂、有无爆音/断续；② 语气与角色情绪是否匹配；"
           "③ 结合人物画面帧看口型与对白是否同步；④ 与⑥声音指令的对白要求对齐"},
    "S2": {"role": "声音设计师", "name": "声音设计", "path": "pure_llm",
           "modules": ["sound"],
           "rule_data": [("audio", "rms"), ("audio", "silence_ratio")],
           "materials": "audio",
           "checkpoints": "① 环境音与场景是否匹配（街道/室内/野外各有声景）；② 对白/音效/"
           "音乐的层次是否平衡、有无互相掩盖；③ 静音占比是否异常（全程近乎静音而指令要求"
           "环境音即不合格）；④ 与⑥声音指令对齐"},
    "S3": {"role": "声音设计师", "name": "声画同步", "path": "pure_llm",
           "modules": ["sound"], "rule_data": [], "materials": "audio_uniform",
           "checkpoints": "① 主要声音事件（脚步/关门/撞击/对白起点）与画面事件时间是否"
           "对齐（容差约 200ms）；② 有无整体提前/滞后；③ 与⑥声音指令对齐"},
    "S4": {"role": "声音设计师", "name": "音乐情绪", "path": "pure_llm",
           "modules": ["sound", "mood"], "rule_data": [], "materials": "audio",
           "checkpoints": "① 配乐风格与⑦情绪风格是否匹配；② 音乐情绪随叙事的动态变化"
           "（铺垫/高潮/收束）；③ 有无音乐与画面情绪打架的段落；④ 与⑥声音指令对齐"},
}

# 七模块指令键 → 中文章节名（解析【章节名】标题用）
MODULE_HEADINGS = {
    "scene": "场景设定", "character": "角色描述", "narrative": "叙事指令",
    "camera": "摄影指定", "editing": "剪辑指令", "sound": "声音指令", "mood": "情绪风格",
}

ROLE_OF = {c: d["role"] for c, d in DIMENSIONS.items()}

GLOBAL_CONTRACT = """【全局契约】
- 只输出 JSON，analysis 必须先于 score 给出
- evidence 必须引用具体时间戳（帧图看文件名中的秒数，音频引用秒级时间点），禁止无证据断言
- score 按 1~5 锚定：5 优秀(可直接成片) / 4 良好(细微偏差) / 3 合格(明显可感知偏差) / 2 不合格(核心指令未执行) / 1 失败(完全偏离)
- 若本维度不适用（如片段无角色、无声音），输出 {"score": null, "na_reason": "原因"}，不要强行打分"""

AUDIO_NOTE = "\n注意：若你没有音频理解能力，不要猜测，直接输出 {\"score\": null, \"na_reason\": \"评估模型无音频能力\"}。"


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------

def parse_modules(script_text):
    """把七模块指令原文切分为 {模块键: 文本}。识别【场景设定】等标题；
    识别不到标题时返回空 dict（调用方回退为全文）。"""
    names = "|".join(MODULE_HEADINGS.values())
    parts = re.split(r"【\s*(%s)\s*】" % names, script_text)
    modules = {}
    # re.split 保留捕获组: [前导, 标题1, 正文1, 标题2, 正文2, ...]
    for i in range(1, len(parts) - 1, 2):
        heading, body = parts[i].strip(), parts[i + 1].strip()
        for key, cn in MODULE_HEADINGS.items():
            if cn == heading:
                modules[key] = body
    return modules


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick(data, dotted):
    """按 'a.b.c' 从嵌套 dict 取值；缺失返回 None。"""
    cur = data
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------------
# 材料解析
# ---------------------------------------------------------------------------

def resolve_materials(kind, manifest, shots, audio):
    """返回 (材料说明, 文件清单, 是否需要音频)。"""
    frames = (manifest or {}).get("frames", [])
    keyframes = (manifest or {}).get("shot_keyframes", [])

    if kind == "none":
        return "本维度为纯规则维度，无需查看任何材料。", [], False
    if kind == "stats_only":
        return "本维度以规则统计数据为准，帧图可不看。", [], False
    if kind == "uniform":
        files = [f["file"] for f in frames]
        return ("以下为全片均匀抽帧（文件名含时间戳秒）。", files, False)
    if kind in ("shot_keyframes", "shot_pairs"):
        if not keyframes:
            return ("需要镜头关键帧但 manifest 中没有（extract_frames.py 请带 --shots 运行）。",
                    [], False)
        files = [f["file"] for f in keyframes]
        note = ("以下为各镜头关键帧（文件名含镜头号与时间戳）。"
                "请按镜头号并排对比。" if kind == "shot_keyframes"
                else "以下为各镜头关键帧。请按相邻镜头对（0-1, 1-2, ...）逐对对比。")
        return (note, files, False)
    if kind == "transition_window":
        transitions = (shots or {}).get("transitions", [])
        if not transitions:
            return "未检出转场点（单镜头片段），本维度通常标 N/A。", [], False
        times = [t.get("time", 0) for t in transitions]
        near = [f["file"] for f in frames
                if any(abs(f["timestamp"] - t) <= 1.0 for t in times)]
        return (f"检出 {len(times)} 个转场点（秒）：{times}。以下为各转场点 ±1s 内的帧"
                "（文件名含时间戳），请逐点查看前后变化。", near, False)
    wav = (audio or {}).get("wav_path")
    audio_note = f"音频文件：{wav}" if wav else "音频文件（由调用方提供）"
    if kind == "audio":
        return f"{audio_note}。请听完整段音频后评估。", [], True
    if kind == "audio_uniform":
        files = [f["file"] for f in frames]
        return (f"{audio_note}，另附全片均匀帧用于声画对照。", files, True)
    if kind == "audio_character":
        files = [f["file"] for f in frames]
        return (f"{audio_note}，另附全片均匀帧（重点看有人物画面处的口型）。", files, True)
    raise ValueError(f"未知材料类型: {kind}")


# ---------------------------------------------------------------------------
# Prompt 生成
# ---------------------------------------------------------------------------

def build_prompt(code, modules, data_sources, manifest, shots, audio):
    """生成一个维度的具体 Prompt。返回 (prompt_text, meta)。"""
    spec = DIMENSIONS[code]
    warnings = []

    # 指令模块摘录
    module_lines = []
    for key in spec["modules"]:
        text = modules.get(key)
        if text:
            module_lines.append(f"〈{MODULE_HEADINGS[key]}〉{text}")
        else:
            module_lines.append(f"〈{MODULE_HEADINGS[key]}〉（指令中未找到该模块原文）")
            warnings.append(f"指令缺少模块: {MODULE_HEADINGS[key]}")
    module_block = "\n".join(module_lines)

    # 规则数据摘录
    rule_parts = []
    for src, dotted in spec["rule_data"]:
        data = data_sources.get(src)
        val = pick(data, dotted) if data else None
        if val is None:
            warnings.append(f"规则数据缺失: {src}:{dotted}")
            continue
        rule_parts.append(json.dumps({dotted: val}, ensure_ascii=False))
    rule_block = "\n".join(rule_parts) if rule_parts else ""

    # 材料
    mat_note, mat_files, needs_audio = resolve_materials(
        spec["materials"], manifest, shots, audio)
    mat_block = mat_note
    if mat_files:
        mat_block += "\n" + "\n".join(f"- {f}" for f in mat_files)

    header = (f"你是一位专业的{spec['role']}，正在评估一段 AI 生成视频是否准确执行了电影指令。\n"
              f"【评估维度】{code} {spec['name']}\n"
              f"【相关指令原文】\n{module_block}\n")

    if spec["path"] == "pure_llm":
        aux = ""
        if rule_block:  # 纯LLM维度的规则数据是可选辅助参考（如 S2 的音量/静音比）
            aux = f"【辅助数据】（可选参考的客观统计）\n{rule_block}\n"
        body = (f"{aux}"
                f"【材料】\n{mat_block}\n"
                f"请逐步分析：\n{spec['checkpoints']}\n"
                '只输出 JSON：{"analysis": "逐步分析...", "evidence": "时间戳证据", "score": N}')
        if needs_audio:
            body += AUDIO_NOTE
    elif spec["path"] == "mixed":
        body = (f"【客观测量数据】（算法对原视频的客观测量，不含主观判断）\n{rule_block or '（缺失，按纯LLM路径评估并在 analysis 注明）'}\n"
                f"【材料】\n{mat_block}\n"
                f"1. 先依据客观数据与指令的匹配程度给出规则分 rule_score（1~5），写明映射依据；"
                "若上方数据已直接给出分数（如 F4/F7），直接采用\n"
                f"2. 再看材料，按检查点独立给出 LLM 分 llm_score（1~5）：\n{spec['checkpoints']}\n"
                "3. 两者冲突时以材料为准复核数据是否失效（如检测错误），并在 analysis 说明\n"
                '只输出 JSON：{"analysis": "...", "evidence": "...", "rule_score": N, '
                '"llm_score": N, "score": round(rule_score*0.4 + llm_score*0.6)}')
    else:  # rule_first
        body = (f"【客观结论】（规则产出的基础分/客观事实）\n{rule_block or '（缺失：请先运行对应 scripts/ 再评估）'}\n"
                f"【核验材料】\n{mat_block}\n"
                "你的任务只是核验客观结论与材料是否矛盾：\n"
                f"核验要点：{spec['checkpoints']}\n"
                "- 不矛盾 → 直接采用规则分，evidence 说明核验了什么\n"
                "- 矛盾 → ±1 微调，evidence 必须给出矛盾处的时间戳\n"
                "禁止抛开规则结论凭主观印象重新打分。\n"
                '只输出 JSON：{"analysis": "...", "evidence": "...", "score": N}')

    prompt = f"{header}\n{body}\n\n{GLOBAL_CONTRACT}"

    # D4 特判：规则分已存在则无需 LLM
    skip_llm = False
    if code == "D4" and shots and pick(shots, "d4_shot_count.score") is not None:
        skip_llm = True
        prompt = (f"维度 D4 镜头数量为纯规则维度，规则已直接出分：\n"
                  f"{json.dumps(pick(shots, 'd4_shot_count'), ensure_ascii=False)}\n"
                  f"无需 LLM 调用，直接采用该分数并在报告中注明「规则分」。")

    meta = {"dimension": code, "role": spec["role"], "name": spec["name"],
            "path": spec["path"], "requires_audio": needs_audio,
            "skip_llm": skip_llm, "material_files": mat_files, "warnings": warnings}
    return prompt, meta


def main():
    p = argparse.ArgumentParser(description="按维度生成具体评估 Prompt（23 维注册表 + 三层组装）")
    p.add_argument("dimension", nargs="?", help="维度代号（A1~S4）；--all 时省略")
    p.add_argument("--all", action="store_true", help="生成全部 23 个维度的 Prompt")
    p.add_argument("--script", help="七模块指令原文文件（UTF-8）")
    p.add_argument("--visual", help="visual_analysis.py 输出 JSON")
    p.add_argument("--shots", help="shot_detect.py 输出 JSON")
    p.add_argument("--audio", help="audio_extract.py 输出 JSON")
    p.add_argument("--manifest", help="extract_frames.py 的 manifest.json")
    p.add_argument("--outdir", help="--all 模式：每个维度写一个 <维度>.txt 到该目录")
    p.add_argument("--json", action="store_true",
                   help="输出 JSON（含 prompt + 元数据），供编排程序路由模型使用")
    args = p.parse_args()

    if not args.all and not args.dimension:
        p.error("需要维度代号或 --all")

    try:
        modules = parse_modules(open(args.script, encoding="utf-8").read()) if args.script else {}
        data_sources = {k: load_json(v) for k, v in
                        (("visual", args.visual), ("shots", args.shots), ("audio", args.audio)) if v}
        manifest = load_json(args.manifest) if args.manifest else None
        shots = data_sources.get("shots")
        audio = data_sources.get("audio")

        codes = list(DIMENSIONS) if args.all else [args.dimension.upper()]
        results = []
        for code in codes:
            if code not in DIMENSIONS:
                raise RuntimeError(f"未知维度代号: {code}（可选: {', '.join(DIMENSIONS)}）")
            prompt, meta = build_prompt(code, modules, data_sources, manifest, shots, audio)
            results.append({"prompt": prompt, **meta})

        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            for r in results:
                with open(os.path.join(args.outdir, f"{r['dimension']}.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(r["prompt"])
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)

    if args.json:
        print(json.dumps(results if args.all else results[0], ensure_ascii=False, indent=2))
    else:
        for i, r in enumerate(results):
            if i:
                print("\n" + "=" * 70 + "\n")
            print(r["prompt"])
        for r in results:
            for w in r["warnings"]:
                print(f"[警告] {r['dimension']}: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
