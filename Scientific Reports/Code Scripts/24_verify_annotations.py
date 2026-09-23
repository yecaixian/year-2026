"""Step 5: 最终校验。"""
import csv
import json
from collections import Counter
from pathlib import Path

ANN = Path(r"C:\Users\lenovo\WorkBuddy\2026-09-21-20-29-42\data\annotations")
DIMS_CN = ["色彩协调性", "构图新颖性", "主题清晰度", "感知技术复杂度", "整体美学偏好"]
CLASSES_CN = ["GAN 合成", "噪声场", "像素拼贴", "光流", "故障", "粒子",
              "程序化纹理", "数据映射", "体素", "着色器", "分形", "算法笔触"]

ok = True

with open(ANN / "Dseed_annotation_table.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"主表行数: {len(rows)}  (期望 2000)")
ok &= len(rows) == 2000
print(f"列名: {list(rows[0].keys())}")

# 评分合法性
bad = 0
for r in rows:
    for d in DIMS_CN:
        v = float(r[d])
        if not (1.0 <= v <= 5.0):
            bad += 1
print(f"越界评分: {bad}")
ok &= bad == 0

# 三分位倍数检查
mult_ok = 0
for r in rows:
    for d in DIMS_CN:
        v = round(float(r[d]) * 3)
        if abs(v / 3 - float(r[d])) < 0.006:
            mult_ok += 1
print(f"符合 1/3 步长的评分占比: {mult_ok}/{len(rows)*5} = {mult_ok/(len(rows)*5):.1%}")

# 标签完整性
cnt = Counter(r["风格标签"] for r in rows)
print(f"出现的标签数: {len(cnt)}  (期望 12)")
missing = [c for c in CLASSES_CN if c not in cnt]
print(f"缺失标签: {missing if missing else '无'}")
ok &= len(cnt) == 12 and not missing
print("标签分布:", dict(sorted(cnt.items(), key=lambda x: -x[1])))

# 各维度评分范围
for d in DIMS_CN:
    vs = [float(r[d]) for r in rows]
    print(f"  {d}: min={min(vs):.2f} max={max(vs):.2f} mean={sum(vs)/len(vs):.2f}")

# 原始标注者表
with open(ANN / "Dseed_raw_annotators.csv", encoding="utf-8-sig") as f:
    raw = list(csv.DictReader(f))
print(f"\n标注者原始评分行数: {len(raw)}  (期望 30000)")
ok &= len(raw) == 30000
rv = Counter(int(r["评分"]) for r in raw)
print(f"原始评分取值为 {sorted(rv.keys())}, 均在 1-5: {set(rv.keys()) <= {1,2,3,4,5}}")
ok &= set(rv.keys()) <= {1, 2, 3, 4, 5}

# 扩展表
with open(ANN / "Dseed_annotation_full.csv", encoding="utf-8-sig") as f:
    full = list(csv.DictReader(f))
print(f"扩展表行数: {len(full)}")
ok &= len(full) == 2000
print(f"来源构成: {dict(Counter(r['source_dataset'] for r in full))}")

# meta
meta = json.loads((ANN / "Dseed_meta.json").read_text(encoding="utf-8"))
print(f"\nalpha: {meta['krippendorff_alpha']}")
print(f"alpha mean: {meta['krippendorff_alpha_mean']}")

# 文件清单
print("\n产出文件:")
for p in sorted(ANN.iterdir()):
    print(f"  {p.name:<40} {p.stat().st_size/1024:>9.1f} KB")

print(f"\n=== 校验结果: {'全部通过' if ok else '存在问题'} ===")
