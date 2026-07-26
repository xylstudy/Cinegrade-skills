#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_banner.py — 生成 cine-eval README 横幅图（docs/banner.png）。

深色电影感背景 + 胶片齿孔 + 相机光圈 + 六角色标签。
用法: python docs/make_banner.py   （在 skill 根目录运行，Pillow 唯一依赖）
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 500
AMBER = (245, 197, 24)       # 电影金
INK = (13, 17, 23)           # GitHub 深色底
STEEL = (140, 155, 175)
CHIP_BG = (30, 38, 52)

FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
FONT_LATIN = "C:/Windows/Fonts/arialbd.ttf"


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def vertical_gradient(size, top, bottom):
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        c = lerp(top, bottom, y / size[1])
        for x in range(size[0]):
            px[x, y] = c
    return img


def film_strip(draw, y0, height):
    """画一条胶片：深色带 + 两排齿孔 + 中间格窗。"""
    draw.rectangle([0, y0, W, y0 + height], fill=(8, 10, 14))
    hole_w, hole_h, gap = 34, 18, 26
    for row_y in (y0 + 8, y0 + height - 26):
        x = 16
        while x < W - hole_w:
            draw.rounded_rectangle([x, row_y, x + hole_w, row_y + hole_h],
                                   radius=5, fill=(38, 44, 56))
            x += hole_w + gap
    # 中间格窗线
    mid = y0 + height // 2
    draw.line([0, mid - 14, W, mid - 14], fill=(24, 29, 38), width=2)
    draw.line([0, mid + 14, W, mid + 14], fill=(24, 29, 38), width=2)


def draw_aperture(draw, cx, cy, r_out, r_in, blades=6):
    """相机光圈：外环 + 旋转切线叶片。"""
    draw.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out],
                 outline=AMBER, width=6)
    for i in range(blades):
        th = math.radians(i * 360 / blades + 15)
        px, py = cx + r_in * math.cos(th), cy + r_in * math.sin(th)
        tx, ty = -math.sin(th), math.cos(th)          # 切线方向
        L = math.sqrt(max(r_out ** 2 - r_in ** 2, 0))  # 切线段长度（抵外环）
        draw.line([px - tx * L, py - ty * L, px + tx * L, py + ty * L],
                  fill=lerp(AMBER, STEEL, 0.35), width=5)
    draw.ellipse([cx - r_in * 0.55, cy - r_in * 0.55,
                  cx + r_in * 0.55, cy + r_in * 0.55], outline=STEEL, width=3)


def main():
    img = vertical_gradient((W, H), (22, 27, 38), INK)
    d = ImageDraw.Draw(img)

    film_strip(d, 0, 78)
    film_strip(d, H - 78, 78)

    # 装饰：左上角场记板斜纹
    for i in range(6):
        x0 = 60 + i * 42
        d.polygon([(x0, 92), (x0 + 24, 92), (x0 + 10, 118), (x0 - 14, 118)],
                  fill=lerp(AMBER, INK, i / 8))

    f_title = ImageFont.truetype(FONT_LATIN, 108)
    f_sub = ImageFont.truetype(FONT_BOLD, 40)
    f_chip = ImageFont.truetype(FONT_BOLD, 34)

    # 标题（字距拉开）
    x, y = 90, 130
    for ch in "CINE-EVAL":
        d.text((x, y), ch, font=f_title, fill=(240, 244, 250))
        x += int(f_title.getlength(ch)) + 14
    d.rectangle([92, 262, 92 + 520, 268], fill=AMBER)

    d.text((92, 292), "电影视频六角色评估 · CineBench v2", font=f_sub, fill=STEEL)
    d.text((92, 348), "零模型权重 — 多模态 LLM × OpenCV 规则公式", font=f_sub,
           fill=lerp(STEEL, INK, 0.25))

    # 六角色标签
    roles = ["导演", "美术指导", "摄影师", "演员指导", "剪辑师", "声音设计师"]
    cx0, cy0 = 92, 420
    for r in roles:
        tw = f_chip.getlength(r)
        d.rounded_rectangle([cx0, cy0, cx0 + tw + 36, cy0 + 52], radius=12,
                            fill=CHIP_BG, outline=(52, 62, 80), width=2)
        d.text((cx0 + 18, cy0 + 8), r, font=f_chip, fill=AMBER)
        cx0 += tw + 36 + 18

    draw_aperture(d, 1330, 260, 150, 96)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banner.png")
    img.save(out)
    print("written:", out)


if __name__ == "__main__":
    main()
