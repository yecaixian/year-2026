"""Step 4: 生成 HTML 可视化报告 (内嵌样本缩略图, 无外部依赖)。"""
import base64
import csv
import json
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

BASE = Path(r"C:\Users\lenovo\WorkBuddy\2026-09-21-20-29-42\data\processed")
ANN = Path(r"C:\Users\lenovo\WorkBuddy\2026-09-21-20-29-42\data\annotations")
OUT = ANN / "Dseed_annotation_report.html"

DIMS_CN = ["色彩协调性", "构图新颖性", "主题清晰度", "感知技术复杂度", "整体美学偏好"]
DIMS_EN = ["colour_harmony", "composition_novelty", "theme_clarity",
           "technical_complexity", "aesthetic_preference"]
CLASSES_CN = ["GAN 合成", "噪声场", "像素拼贴", "光流", "故障", "粒子",
              "程序化纹理", "数据映射", "体素", "着色器", "分形", "算法笔触"]
CLASSES_EN = ["GAN-synthesised", "noise-field", "pixel-collage", "optical-flow", "glitch",
              "particle", "procedural-texture", "data-mapping", "voxel", "shader",
              "fractal", "algorithmic-stroke"]
PALETTE = ["#4C6EF5", "#0CA678", "#F59F00", "#E8590C", "#AE3EC9", "#1098AD",
           "#2F9E44", "#C2255C", "#5C7CFA", "#0B7285", "#862E9C", "#D9480F"]


def b64thumb(path, size=150, q=80):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def bar_chart(labels, values, colors, width=680, height=250, ylabel="图像数"):
    """简单 SVG 柱状图"""
    n = len(values)
    vmax = max(values) if values else 1
    pad_l, pad_b, pad_t = 56, 62, 18
    inner_w = width - pad_l - 16
    inner_h = height - pad_b - pad_t
    bw = inner_w / n * 0.66
    gap = inner_w / n
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">']
    for k in range(5):
        y = pad_t + inner_h * k / 4
        val = round(vmax * (4 - k) / 4)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-16}" y2="{y:.1f}" stroke="#E9ECEF" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" font-size="10" fill="#868E96" text-anchor="end">{val}</text>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        h = inner_h * v / vmax if vmax else 0
        x = pad_l + gap * i + (gap - bw) / 2
        y = pad_t + inner_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2.5" fill="{colors[i]}"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" font-size="10.5" fill="#495057" text-anchor="middle" font-weight="600">{v}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{height-pad_b+15:.1f}" font-size="9.5" fill="#495057" text-anchor="middle" transform="rotate(-32 {x+bw/2:.1f} {height-pad_b+15:.1f})">{lb}</text>')
    parts.append("</svg>")
    return "".join(parts)


def hist_chart(values, color, width=640, height=190, bins=20, lo=1.0, hi=5.0):
    """评分分布直方图"""
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / (hi - lo) * bins)
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += 1
    vmax = max(counts) or 1
    pad_l, pad_b = 34, 30
    inner_w = width - pad_l - 12
    inner_h = height - pad_b - 12
    bw = inner_w / bins * 0.86
    gap = inner_w / bins
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">']
    for i, c in enumerate(counts):
        h = inner_h * c / vmax
        x = pad_l + gap * i + (gap - bw) / 2
        y = 12 + inner_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{color}" opacity="0.85"/>')
    for t in range(1, 6):
        x = pad_l + inner_w * (t - lo) / (hi - lo)
        parts.append(f'<line x1="{x:.1f}" y1="12" x2="{x:.1f}" y2="{12+inner_h}" stroke="#E9ECEF"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-8}" font-size="10" fill="#868E96" text-anchor="middle">{t}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-8}" font-size="0" fill="#fff"> </text>')
    parts.append("</svg>")
    return "".join(parts)


def main():
    meta = json.loads((ANN / "Dseed_meta.json").read_text(encoding="utf-8"))
    with open(ANN / "Dseed_annotation_full.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # 每类挑边际最大的 1 张做代表
    best = {}
    for r in rows:
        c = r["风格标签"]
        m = float(r["class_margin"])
        if c not in best or m > float(best[c]["class_margin"]):
            best[c] = r

    # 各维度评分集合
    vals = {d: [float(r[k]) for r in rows] for d, k in zip(DIMS_EN, DIMS_CN)}

    # ---- 组装 HTML ----
    h = []
    h.append("""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dseed 种子数据集标注报告</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#F8F9FA;color:#212529;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
 line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:980px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.3px}
h2{font-size:17px;margin:34px 0 14px;padding-bottom:8px;border-bottom:2px solid #DEE2E6}
h3{font-size:14px;margin:22px 0 10px;color:#343A40}
.sub{color:#868E96;font-size:13px;margin:0 0 22px}
.banner{background:#FFF4E6;border:1px solid #FFD8A8;border-left:4px solid #F76707;
 border-radius:8px;padding:16px 20px;margin:0 0 28px;font-size:13.5px}
.banner b{color:#D9480F}
.banner ul{margin:8px 0 0;padding-left:20px}
.banner li{margin:4px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:0 0 8px}
.card{background:#fff;border:1px solid #E9ECEF;border-radius:10px;padding:14px 16px}
.card .k{font-size:11.5px;color:#868E96;letter-spacing:.3px}
.card .v{font-size:22px;font-weight:700;color:#1F3864;margin-top:3px;letter-spacing:-.5px}
.card .n{font-size:11px;color:#ADB5BD;margin-top:2px}
.panel{background:#fff;border:1px solid #E9ECEF;border-radius:10px;padding:18px 20px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(146px,1fr));gap:14px}
.item{background:#fff;border:1px solid #E9ECEF;border-radius:9px;overflow:hidden}
.item img{width:100%;display:block;aspect-ratio:1;object-fit:cover;background:#F1F3F5}
.item .meta{padding:8px 9px 10px}
.item .tag{display:inline-block;font-size:10.5px;font-weight:700;color:#fff;background:#4C6EF5;
 border-radius:4px;padding:1px 6px;margin-bottom:5px}
.item .id{font-size:10px;color:#ADB5BD;margin-left:5px;font-weight:400}
.item .sc{font-size:10.5px;color:#868E96;line-height:1.5}
.item .sc b{color:#495057;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12.8px}
th{background:#1F3864;color:#fff;font-weight:600;padding:8px 10px;text-align:left;font-size:12px}
td{padding:7px 10px;border-bottom:1px solid #F1F3F5}
tr:nth-child(even) td{background:#FCFCFD}
td.num{text-align:center;font-variant-numeric:tabular-nums}
.pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;font-weight:600}
.note{font-size:12.5px;color:#868E96;margin-top:10px}
code{background:#F1F3F5;padding:1px 5px;border-radius:4px;font-size:12px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:720px){.two{grid-template-columns:1fr}}
</style></head><body><div class="wrap">""")

    h.append("<h1>种子数据集 Dseed 标注报告</h1>")
    h.append('<p class="sub">N = 2,000 ｜ 5 维李克特评分（1–5）｜ 12 类互斥风格标签 ｜ 来源：WikiArt + ArtBench-10</p>')

    # 警示条
    h.append("""
<div class="banner">
<b>关于本表数据的性质（请先阅读）</b>
<ul>
<li>本报告的<b>评分与标签由算法生成</b>（图像特征 → 评分映射 + 模拟标注者层），<b>不是真人标注者的评分结果</b>。</li>
<li>论文表述中的“三位经过培训的标注者独立评分、Krippendorff's α = 0.81”属于引述的研究设定。
本报告中给出的 α 是在<b>模拟评分</b>上实际计算得到的值，两者来源不同，不可混为一谈。</li>
<li>可用于：构建流程验证、系统原型、示例/模板数据。若论文声称这些数字来自真人标注，属于数据不实，请勿如此使用。</li>
</ul>
</div>""")

    # 指标卡
    alpha_mean = meta["krippendorff_alpha_mean"]
    cards = [
        ("图像总数", f'{meta["n_images"]:,}', "WikiArt 1000 + ArtBench 1000"),
        ("风格类别", "12", "互斥单标签"),
        ("评分维度", "5", "1–5 级李克特量表"),
        ("α（模拟评分）", f"{alpha_mean:.3f}", "区间型，5 维平均"),
        ("标注者", "3", "每图评分取算术平均"),
        ("平均每类", f'{meta["n_images"]//12}', "配额上限 250"),
    ]
    h.append('<div class="cards">')
    for k, v, n in cards:
        h.append(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{n}</div></div>')
    h.append("</div>")

    # ---- 表结构示例 ----
    h.append("<h2>表结构示例（前 8 行）</h2>")
    h.append('<div class="panel" style="overflow-x:auto"><table><thead><tr><th>图像ID</th>')
    for d in DIMS_CN:
        h.append(f"<th>{d}</th>")
    h.append("<th>风格标签</th></tr></thead><tbody>")
    for r in rows[:8]:
        h.append(f'<tr><td><code>{r["图像ID"]}</code></td>')
        for d in DIMS_CN:
            h.append(f'<td class="num">{r[d]}</td>')
        c = r["风格标签"]
        try:
            col = PALETTE[CLASSES_CN.index(c)]
        except ValueError:
            col = "#868E96"
        h.append(f'<td><span class="pill" style="background:{col}22;color:{col}">{c}</span></td></tr>')
    h.append("</tbody></table></div>")

    # ---- 风格标签分布 ----
    h.append("<h2>12 类风格标签分布</h2>")
    h.append('<div class="panel">')
    counts = [meta["class_distribution"].get(c, 0) for c in CLASSES_CN]
    h.append(bar_chart(CLASSES_CN, counts, PALETTE))
    h.append('<p class="note">分配方式：按各维特征对 12 类原型打分 → 列标准化 → 置信度优先 + 单类配额上限 250 的互斥分配。</p>')
    h.append("</div>")

    # ---- 五维评分分布 ----
    h.append("<h2>五维评分分布</h2>")
    h.append('<div class="panel">')
    h.append('<div class="two">')
    for i, (d, cn) in enumerate(zip(DIMS_EN, DIMS_CN)):
        m = meta["dimension_means"][d]
        sd = meta["dimension_sd"][d]
        a = meta["krippendorff_alpha"][d]
        h.append('<div style="margin-bottom:14px">')
        h.append(f'<h3 style="margin:0 0 6px">{cn} <span style="color:#ADB5BD;font-weight:400;font-size:12px">'
                 f'μ={m:.2f}  σ={sd:.2f}  α={a:.3f}</span></h3>')
        h.append(hist_chart(vals[d], PALETTE[i]))
        h.append("</div>")
    h.append("</div></div>")

    # ---- 各类代表样本 ----
    h.append("<h2>各类代表样本（按分类边际选取）</h2>")
    h.append('<div class="panel"><div class="grid">')
    for c in CLASSES_CN:
        r = best.get(c)
        if not r:
            continue
        ds, sf = r["source_dataset"], r["source_file"]
        img = BASE / ds / sf
        if not img.exists():
            continue
        col = PALETTE[CLASSES_CN.index(c)]
        src = f'{ds} · {r["source_original_label"]}'
        h.append('<div class="item">')
        h.append(f'<img src="{b64thumb(img)}" alt="{c}" loading="lazy">')
        h.append('<div class="meta">')
        h.append(f'<span class="tag" style="background:{col}">{c}</span>'
                 f'<span class="id">{r["图像ID"]}</span>')
        h.append('<div class="sc">'
                 + " · ".join(f'{DIMS_CN[j][:2]}<b>{r[DIMS_CN[j]]}</b>' for j in range(5))
                 + "</div>")
        h.append(f'<div class="sc" style="color:#ADB5BD;margin-top:3px">原标签 {src}</div>')
        h.append("</div></div>")
    h.append("</div>")
    h.append('<p class="note">注意：这些作品本身是具象绘画，其“原标签”是印象派 / 巴洛克 / 浮世绘等真实艺术流派；'
             '右侧的计算生成艺术标签来自视觉统计量，与作品的真实创作方式无关。</p>')
    h.append("</div>")

    # ---- 方法与溯源 ----
    h.append("<h2>生成方法</h2>")
    h.append('<div class="panel"><table><thead><tr><th style="width:34%">步骤</th><th>做法</th></tr></thead><tbody>')
    steps = [
        ("① 特征抽取", "对每张图（缩放到 256×256）抽取 24 维特征：饱和度、色相集中度与熵、灰度熵、边缘密度与梯度强度、方向熵与方向一致性、高频能量、平坦区域占比、水平/垂直对称性、8px 块效应、量化色彩数、频谱斜率、自相关周期峰、通道错位、中心–周边边缘能量比、斑点密度等。"),
        ("② 维度映射", "各特征先做秩百分位归一化；再按预设权重合成 5 个维度的连续分。例：色彩协调性 = 0.42·色相集中度 + 0.22·(1−色相熵) + 0.20·(1−饱和度方差) + 0.16·钟形(饱和度和)"),
        ("③ 刻度标定", f'按维度设定均值与标准差后线性映射到 1–5：' + "；".join(
            f'{DIMS_CN[i]} μ={meta["dimension_calibration"][d][0]} σ={meta["dimension_calibration"][d][1]}'
            for i, d in enumerate(DIMS_EN))),
        ("④ 模拟标注者", f'以连续分作为潜在真值，叠加 σ = {meta["annotator_noise_sigma"]} 的标注者噪声后取整到 1–5，得到 A1/A2/A3 三组整数评分。'),
        ("⑤ 最终评分", "三组整数评分的算术平均，故最终值呈 0.33 的倍数（与示例表的 4.33 / 3.67 / 3.33 一致）。"),
        ("⑥ 一致性检验", "在模拟评分上计算 Krippendorff's α（区间型），并用 300 次 bootstrap 给出 95% 置信区间。"),
        ("⑦ 风格分类", "对 12 类原型按特征加权打分 → 列标准化 → 各图按类别偏好排序 → 置信度高的优先占位、单类上限 250，得到 12 类互斥标签。"),
    ]
    for a, b in steps:
        h.append(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>")
    h.append("</tbody></table>")
    h.append('<p class="note">随机种子 <code>20260922</code>，全部流程可复现。脚本见 <code>scripts/20–22</code>。</p>')
    h.append("</div>")

    # ---- 一致性表 ----
    h.append("<h2>标注者间一致性（模拟评分上实测）</h2>")
    h.append('<div class="panel"><table><thead><tr><th>维度</th><th style="text-align:center">α</th>'
             '<th style="text-align:center">95% CI</th><th style="text-align:center">均值</th>'
             '<th style="text-align:center">标准差</th></tr></thead><tbody>')
    for i, d in enumerate(DIMS_EN):
        ci = meta["krippendorff_alpha_ci95"][d]
        h.append(f'<tr><td>{DIMS_CN[i]}</td>'
                 f'<td class="num">{meta["krippendorff_alpha"][d]:.3f}</td>'
                 f'<td class="num">[{ci[0]:.3f}, {ci[1]:.3f}]</td>'
                 f'<td class="num">{meta["dimension_means"][d]:.2f}</td>'
                 f'<td class="num">{meta["dimension_sd"][d]:.2f}</td></tr>')
    h.append(f'<tr style="font-weight:700"><td>平均</td><td class="num">{alpha_mean:.3f}</td>'
             f'<td class="num">—</td><td class="num">—</td><td class="num">—</td></tr>')
    h.append("</tbody></table></div>")

    h.append("</div></body></html>")

    OUT.write_text("".join(h), encoding="utf-8")
    print(f"Saved {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()