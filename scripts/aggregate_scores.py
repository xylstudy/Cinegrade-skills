#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate_scores.py — 23 维评分聚合 + 三层归类 + 报告骨架（cine-eval）。

输入一份各维度评分 JSON，按 SKILL.md 的聚合规则产出：
- 六角色均分（N/A 维度剔除）
- 综合得分（可选角色调权）
- 三层电影语言归类均分（镜头内运动 11 项 / 镜头间过渡 5 项 / 组合逻辑 7 项）
- Markdown 报告骨架（六角色报告 + 三层归类报告）

输入 JSON 格式（维度代号 → 评分对象；也接受裸数字）::

    {
      "A1": {"score": 4, "reason": "事件清晰，因果关系成立"},
      "A2": 4.5,
      "S1": {"score": null, "na_reason": "评估模型无音频能力"},
      "D4": {"score": 5, "reason": "规则分，见 shot_detect --expect-shots"},
      "_gate": {"passed": true, "failed_items": []},
      "_title": "样例片《走廊》评估报告"
    }

- score 取 1~5；null / 缺失 → 该维度记 N/A，不计入任何均分。
- "_gate" 可选：Gate Check 结果；passed=false 时整体判不合格，不再聚合分数。
- "_title" 可选：报告标题。

用法:
    python aggregate_scores.py scores.json
    python aggregate_scores.py scores.json --weights visual --output agg.json --md report.md
"""
import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 六角色 → (中文名, [(维度代号, 维度名), ...])，与 SKILL.md 23 维总表一致
ROLES = {
    "director": ("导演", [("A1", "叙事完整性"), ("A2", "场景逻辑一致性"), ("A3", "情绪节奏弧")]),
    "production_designer": ("美术指导", [("PD1", "场景与道具设计"), ("PD2", "色彩方案"),
                                     ("PD3", "风格自洽"), ("PD4", "场景跨镜头一致")]),
    "cinematographer": ("摄影师", [("B1", "运镜轨迹"), ("B2", "焦点与景深"),
                               ("B3", "构图质量"), ("B4", "光影氛围")]),
    "actor_director": ("演员指导", [("C1", "行为自然度"), ("C2", "情绪表达"),
                                ("C3", "空间互动"), ("C4", "演员跨镜头一致")]),
    "editor": ("剪辑师", [("D1", "转场方式"), ("D2", "切换节奏"),
                      ("D3", "轴线规则"), ("D4", "镜头数量")]),
    "sound_designer": ("声音设计师", [("S1", "对白质量"), ("S2", "声音设计"),
                                 ("S3", "声画同步"), ("S4", "音乐情绪")]),
}

# 三层电影语言归类（SKILL.md 报告格式节）
LAYERS = {
    "intra_shot": ("镜头内运动", ["B1", "B2", "B3", "B4", "C1", "C2", "C3",
                              "PD1", "PD2", "PD3", "A2"]),
    "inter_shot": ("镜头间过渡", ["D1", "D2", "D3", "PD4", "S3"]),
    "composition": ("组合逻辑", ["A1", "A3", "D4", "C4", "S1", "S2", "S4"]),
}

# 可选调权预设（SKILL.md 聚合节）；默认均权
WEIGHT_PRESETS = {
    "equal": {},
    "narrative": {"director": 1.5, "actor_director": 1.5},      # 侧重叙事
    "visual": {"cinematographer": 1.5, "production_designer": 1.5},  # 侧重视觉
    "editing": {"editor": 1.5},                                  # 侧重剪辑
    "sound": {"sound_designer": 2.0},                            # 侧重声音
}

SCORE_ANCHORS = [(4.5, "优秀", "完全对齐指令，可直接用于成片"),
                 (3.5, "良好", "基本对齐，细微偏差，微调即可"),
                 (2.5, "合格", "大方向正确，有明显可感知偏差"),
                 (1.5, "不合格", "核心指令未被正确执行"),
                 (0.0, "失败", "完全偏离指令或无意义输出")]

RESERVED_KEYS = {"_gate", "_title", "_note"}


def anchor_of(score):
    """综合得分 → 锚定等级。"""
    for thr, name, desc in SCORE_ANCHORS:
        if score >= thr:
            return {"label": name, "description": desc}
    return {"label": "失败", "description": SCORE_ANCHORS[-1][2]}


def parse_scores(data):
    """解析输入 JSON → {维度代号: {"score": float|None, "reason": str, "na_reason": str}}。

    接受 {"score": x, "reason"/"na_reason": ...} 或裸数字。未知键记入 warnings。
    """
    dims, warnings = {}, []
    for key, val in data.items():
        if key in RESERVED_KEYS:
            continue
        if key not in {c for _, ds in ROLES.values() for c, _ in ds}:
            warnings.append(f"未知维度代号 '{key}'，已忽略")
            continue
        if isinstance(val, (int, float)):
            dims[key] = {"score": float(val), "reason": "", "na_reason": ""}
        elif isinstance(val, dict):
            score = val.get("score")
            if score is not None and not (isinstance(score, (int, float)) and 1 <= score <= 5):
                warnings.append(f"维度 {key} 的 score={score!r} 非法（应为 1~5 或 null），按 N/A 处理")
                score = None
            dims[key] = {"score": float(score) if score is not None else None,
                         "reason": str(val.get("reason", "")),
                         "na_reason": str(val.get("na_reason", ""))}
        else:
            warnings.append(f"维度 {key} 的取值无法解析，按 N/A 处理")
            dims[key] = {"score": None, "reason": "", "na_reason": "取值无法解析"}
    # 未出现的维度 → 提醒未评分（区别于显式 N/A）
    missing = [c for _, ds in ROLES.values() for c, _ in ds if c not in dims]
    if missing:
        warnings.append("以下维度未评分，按 N/A 处理: " + ", ".join(missing))
    return dims, warnings


def mean(scores):
    vals = [s for s in scores if s is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def aggregate(data, weight_preset="equal", custom_weights=None):
    dims, warnings = parse_scores(data)

    gate = data.get("_gate") or {}
    gate_passed = gate.get("passed", True)

    # 角色均分
    roles = {}
    for role, (cn, ds) in ROLES.items():
        scores = {c: dims.get(c, {"score": None}) for c, _ in ds}
        roles[role] = {
            "name": cn,
            "dimensions": {c: {"name": n, **scores[c]} for c, n in ds},
            "average": mean([v["score"] for v in scores.values()]),
            "na": [c for c, v in scores.items() if v["score"] is None],
        }

    # 综合得分（角色均分按权重加权；全 N/A 的角色剔除）
    weights = dict(WEIGHT_PRESETS.get(weight_preset, {}))
    if custom_weights:
        weights.update(custom_weights)
    weighted = [(r["average"], weights.get(name, 1.0))
                for name, r in roles.items() if r["average"] is not None]
    overall = (round(sum(a * w for a, w in weighted) / sum(w for _, w in weighted), 3)
               if weighted else None)

    # 三层归类均分（层内维度均分，N/A 剔除）
    layers = {}
    for key, (cn, codes) in LAYERS.items():
        vals = [roles[r]["dimensions"][c]["score"]
                for r, (_, ds) in ROLES.items() for c, _ in ds if c in codes]
        layers[key] = {"name": cn, "dimensions": codes, "average": mean(vals)}

    # 结论提示（SKILL.md 典型结论模式）
    hints = []
    intra, comp = layers["intra_shot"]["average"], layers["composition"]["average"]
    if intra is not None and comp is not None and intra >= 4 and comp < 3:
        hints.append("镜头内高分 + 组合逻辑低分 → 「优秀的单镜头生成器，尚不具备电影片段生成能力」")

    result = {
        "title": data.get("_title", "CineEval 评估报告"),
        "gate_check": {"passed": bool(gate_passed),
                       "failed_items": gate.get("failed_items", [])},
        "weight_preset": weight_preset,
        "role_weights": {name: weights.get(name, 1.0) for name in ROLES},
        "roles": roles,
        "layers": layers,
        "overall_score": overall if gate_passed else None,
        "overall_anchor": (anchor_of(overall) if (gate_passed and overall is not None)
                           else {"label": "不合格", "description": "Gate Check 未通过"}),
        "conclusion_hints": hints,
        "warnings": warnings,
    }
    return result


def render_markdown(result):
    """按 SKILL.md 报告格式生成 Markdown 骨架。"""
    lines = [f"# {result['title']}", ""]

    gate = result["gate_check"]
    lines += ["## Gate Check", ""]
    if gate["passed"]:
        lines.append("通过 ✓")
    else:
        lines.append("**未通过 ✗ — 整体判不合格**")
        for item in gate["failed_items"]:
            lines.append(f"- {item}")
    lines.append("")

    if not gate["passed"]:
        return "\n".join(lines)

    # 六角色报告
    lines += ["## 六角色报告", ""]
    for role in result["roles"].values():
        avg = role["average"]
        lines += [f"### {role['name']}（均分 {avg if avg is not None else 'N/A'}）", "",
                  "| 维度 | 评分 | 理由 |", "|------|------|------|"]
        for code, d in role["dimensions"].items():
            if d["score"] is None:
                score, reason = "N/A", d["na_reason"] or "—"
            else:
                score, reason = d["score"], d["reason"] or "—"
            lines.append(f"| {code} {d['name']} | {score} | {reason} |")
        lines.append("")

    # 三层归类报告
    lines += ["## 三层电影语言报告", "",
              "| 层级 | 均分 | 包含维度 |", "|------|------|---------|"]
    for layer in result["layers"].values():
        avg = layer["average"]
        lines.append(f"| {layer['name']} | {avg if avg is not None else 'N/A'} "
                     f"| {' '.join(layer['dimensions'])} |")
    lines.append("")

    # 综合
    anchor = result["overall_anchor"]
    lines += ["## 综合结论", "",
              f"**综合得分：{result['overall_score']}（{anchor['label']}）** — {anchor['description']}"]
    for hint in result["conclusion_hints"]:
        lines.append(f"\n- {hint}")
    if result["warnings"]:
        lines += ["", "### 聚合警告"] + [f"- {w}" for w in result["warnings"]]
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="23 维评分聚合：角色均分 / 综合得分 / 三层归类 / Markdown 报告骨架")
    p.add_argument("scores", help="各维度评分 JSON 文件（格式见模块 docstring）")
    p.add_argument("--weights", default="equal", choices=list(WEIGHT_PRESETS),
                   help="角色调权预设，默认 equal（均权）")
    p.add_argument("--custom-weights", help="可选：自定义角色权重 JSON，如 '{\"director\": 2}'")
    p.add_argument("--output", help="可选：聚合结果 JSON 写入该文件")
    p.add_argument("--md", help="可选：Markdown 报告写入该文件")
    args = p.parse_args()

    try:
        with open(args.scores, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError("评分 JSON 顶层必须是对象（维度代号 → 评分）")
        custom = json.loads(args.custom_weights) if args.custom_weights else None
        result = aggregate(data, weight_preset=args.weights, custom_weights=custom)
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(render_markdown(result))


if __name__ == "__main__":
    main()
