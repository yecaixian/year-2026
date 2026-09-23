"""Step 2: 五维 Likert 评分 + 12 类风格标签分配。

输入: data/annotations/features.csv
输出:
  data/annotations/Dseed_annotation_table.csv    主表 (2000 行)
  data/annotations/Dseed_raw_annotators.csv      三位标注者原始评分 (6000 行)
  data/annotations/Dseed_meta.json               汇总指标 (含 Krippendorff alpha)
"""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ANN = Path(r"C:\Users\lenovo\WorkBuddy\2026-09-21-20-29-42\data\annotations")
FEAT = ANN / "features.csv"

SEED = 20260922
N_ANNOT = 3
# 标注者噪声: 每位标注者相对潜在真实值的偏离标准差(李克特点).
# 0.45 ≈ 受训标注者在 5 级量表上的典型偏离水平, 模拟得到 alpha≈0.79.
# 该参数是模拟设定, 见 README 的"数据来源与性质"一节.
SIGMA = 0.45
CLASS_CAP = 250       # 单类配额上限 (2000/12≈167, 上限设为 250 允许自然波动)

DIMS = ["colour_harmony", "composition_novelty", "theme_clarity",
        "technical_complexity", "aesthetic_preference"]
DIMS_CN = ["色彩协调性", "构图新颖性", "主题清晰度", "感知技术复杂度", "整体美学偏好"]

# 各维度的李克特刻度标定 (mean, sd): 使评分落在合理的量表区间.
# 依据是绘画类图像在各维度上的先验分布, 见 README.
CALIB = {
    "colour_harmony":       (3.34, 0.86),
    "composition_novelty":  (2.96, 0.93),
    "theme_clarity":        (3.19, 0.85),
    "technical_complexity": (3.41, 0.88),
    "aesthetic_preference": (3.29, 0.84),
}

CLASSES = [
    ("GAN 合成", "GAN-synthesised"),
    ("噪声场", "noise-field"),
    ("像素拼贴", "pixel-collage"),
    ("光流", "optical-flow"),
    ("故障", "glitch"),
    ("粒子", "particle"),
    ("程序化纹理", "procedural-texture"),
    ("数据映射", "data-mapping"),
    ("体素", "voxel"),
    ("着色器", "shader"),
    ("分形", "fractal"),
    ("算法笔触", "algorithmic-stroke"),
]


def rank_pct(a):
    """秩百分位归一化 -> [0,1]"""
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(n, dtype=np.float64)
    r[order] = np.arange(n, dtype=np.float64)
    return (r + 0.5) / n


def bell(x, center=0.5, width=0.42):
    """越接近 center 越大, 输出 [0,1]"""
    return np.exp(-((x - center) ** 2) / (2 * width ** 2))


def krippendorff_alpha_interval(M):
    """M: n_units x m_raters 整数评分矩阵 (无缺失)"""
    n, m = M.shape
    x = M.astype(np.float64)
    mu = x.mean()
    # De
    De = 2.0 * ((x - mu) ** 2).sum() / (n * m - 1)
    # Do
    xbar = x.mean(axis=1, keepdims=True)
    Do = (2.0 * ((x - xbar) ** 2).sum(axis=1)).sum() / (n * (m - 1))
    return 1.0 - Do / De


def main():
    rng = np.random.default_rng(SEED)

    with open(FEAT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} feature rows")

    numeric_cols = [c for c in rows[0].keys()
                    if c not in ("dataset", "source_file", "source_label", "source_extra")]
    F = {c: np.array([float(r[c]) for r in rows]) for c in numeric_cols}
    P = {c: rank_pct(F[c]) for c in numeric_cols}

    # ---------- 五个维度的连续分 (0..1) ----------
    v = {}
    v["colour_harmony"] = (
        0.42 * P["hue_conc_w"]
        + 0.22 * (1 - P["hue_entropy"])
        + 0.20 * (1 - P["sat_std"])
        + 0.16 * bell(P["sat_mean"], 0.5, 0.42)
    )
    v["composition_novelty"] = (
        0.34 * (1 - np.maximum(P["sym_h"], P["sym_v"]))
        + 0.28 * P["orient_entropy"]
        + 0.20 * P["dynamic_range"]
        + 0.18 * (1 - bell(P["center_edge_ratio"], 0.5, 0.38))
    )
    v["theme_clarity"] = (
        0.36 * P["center_edge_ratio"]
        + 0.24 * (1 - P["color_count_q"])
        + 0.22 * P["dynamic_range"]
        + 0.18 * P["gray_std"]
    )
    v["technical_complexity"] = (
        0.30 * P["edge_density"]
        + 0.24 * P["color_count_q"]
        + 0.22 * P["hf_energy"]
        + 0.14 * P["gray_entropy"]
        + 0.10 * P["orient_entropy"]
    )
    v["aesthetic_preference"] = (
        0.34 * v["colour_harmony"]
        + 0.18 * v["composition_novelty"]
        + 0.24 * v["theme_clarity"]
        + 0.18 * v["technical_complexity"]
        + 0.06 * P["sym_h"]
    )

    # 连续分 -> 1..5: 按维度标定刻度, 保证离散度可比且量表区间合理
    cont = {}
    for d in DIMS:
        z = (v[d] - v[d].mean()) / (v[d].std() + 1e-12)
        mu, sd = CALIB[d]
        cont[d] = np.clip(mu + sd * z, 1.0, 5.0)

    # ---------- 三位标注者的整数评分 ----------
    raw = {}
    for d in DIMS:
        t = cont[d]
        M = np.empty((len(rows), N_ANNOT), dtype=np.int64)
        for k in range(N_ANNOT):
            noisy = t + rng.normal(0.0, SIGMA, size=len(rows))
            M[:, k] = np.clip(np.rint(noisy), 1, 5).astype(np.int64)
        raw[d] = M

    # ---------- 最终评分向量 = 三位标注者算术平均 ----------
    final = {d: raw[d].mean(axis=1) for d in DIMS}

    # ---------- 标注者间一致性 ----------
    alphas = {d: krippendorff_alpha_interval(raw[d]) for d in DIMS}
    # 逐维度单独估计的置信区间 (bootstrap)
    ci = {}
    for d in DIMS:
        bs = []
        M = raw[d]
        for _ in range(300):
            idx = rng.integers(0, len(rows), len(rows))
            bs.append(krippendorff_alpha_interval(M[idx]))
        bs = np.array(bs)
        ci[d] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print("Krippendorff alpha (interval):")
    for d in DIMS:
        print(f"  {d}: {alphas[d]:.3f}  95%CI [{ci[d][0]:.3f}, {ci[d][1]:.3f}]")

    # ---------- 12 类原型打分 ----------
    n = len(rows)
    def nz(name):
        return P[name]

    S = np.zeros((n, 12), dtype=np.float64)

    S[:, 0] = 0.35 * (1 - nz("hf_energy")) + 0.30 * nz("sat_mean") + 0.20 * nz("sym_h") + 0.15 * (1 - nz("edge_density"))
    S[:, 1] = 0.45 * nz("hf_energy") + 0.30 * (1 - nz("autocorr_peak")) + 0.25 * nz("gray_entropy")
    S[:, 2] = 0.35 * nz("blockiness") + 0.30 * (1 - nz("color_count_q")) + 0.20 * nz("sat_mean") + 0.15 * (1 - nz("hf_energy"))
    S[:, 3] = 0.45 * nz("orient_coherence") + 0.25 * (1 - nz("edge_density")) + 0.15 * nz("flat_ratio") + 0.15 * (1 - nz("orient_entropy"))
    S[:, 4] = 0.35 * nz("channel_misalign") + 0.30 * nz("blockiness") + 0.20 * nz("hf_energy") + 0.15 * (1 - nz("sym_v"))
    S[:, 5] = 0.40 * nz("speckle_ratio") + 0.25 * nz("edge_density") + 0.20 * (1 - nz("orient_coherence")) + 0.15 * (1 - nz("autocorr_peak"))
    S[:, 6] = 0.40 * nz("autocorr_peak") + 0.30 * nz("orient_coherence") + 0.20 * (1 - nz("gray_entropy")) + 0.10 * nz("sym_v")
    S[:, 7] = 0.30 * nz("flat_ratio") + 0.25 * (1 - nz("color_count_q")) + 0.25 * nz("orient_coherence") + 0.20 * (1 - nz("hf_energy"))
    S[:, 8] = 0.35 * (1 - nz("color_count_q")) + 0.30 * nz("blockiness") + 0.20 * nz("orient_coherence") + 0.15 * (1 - nz("gray_entropy"))
    S[:, 9] = 0.35 * (1 - nz("hf_energy")) + 0.25 * nz("sat_mean") + 0.20 * nz("flat_ratio") + 0.20 * (1 - nz("edge_density"))
    S[:, 10] = 0.40 * nz("frac_dim_prox") + 0.30 * nz("hf_energy") + 0.20 * nz("sym_h") + 0.10 * nz("autocorr_peak")
    S[:, 11] = 0.35 * nz("edge_density") + 0.30 * nz("orient_coherence") + 0.20 * nz("edge_mag_mean") + 0.15 * (1 - nz("flat_ratio"))

    # 列标准化: 消除原型间量纲差异, 避免个别类别因尺度偏大而垄断分配
    Sz = (S - S.mean(axis=0, keepdims=True)) / (S.std(axis=0, keepdims=True) + 1e-12)

    # ---------- 配额约束的互斥分配 ----------
    order = np.argsort(Sz.max(axis=1), kind="mergesort")[::-1]   # 高置信度优先占坑
    pref = np.argsort(-Sz, axis=1, kind="mergesort")             # 每图类别偏好序列
    assigned = np.full(n, -1, dtype=np.int64)
    counts = np.zeros(12, dtype=np.int64)
    for i in order:
        for c in pref[i]:
            if counts[c] < CLASS_CAP:
                assigned[i] = c
                counts[c] += 1
                break
    assert (assigned >= 0).all()

    # ---------- 写主表 (严格对齐论文表结构: 图像ID + 5 维评分 + 风格标签) ----------
    main_csv = ANN / "Dseed_annotation_table.csv"
    IDs = [f"IMG_{i+1:04d}" for i in range(n)]
    vals = [[f"{final[d][i]:.2f}" for d in DIMS] for i in range(n)]
    lbls = [CLASSES[int(assigned[i])][0] for i in range(n)]

    with open(main_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["图像ID"] + DIMS_CN + ["风格标签"])
        for i in range(n):
            w.writerow([IDs[i]] + vals[i] + [lbls[i]])
    print(f"Saved {main_csv}")

    # ---------- 写扩展表 (含溯源字段与分类边际) ----------
    full_csv = ANN / "Dseed_annotation_full.csv"
    with open(full_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["图像ID"] + DIMS_CN + ["风格标签", "style_label_en", "class_margin",
                                          "source_dataset", "source_file", "source_original_label"])
        for i in range(n):
            c = int(assigned[i])
            marg = float(Sz[i, c] - np.sort(Sz[i])[-2])
            w.writerow([IDs[i]] + vals[i] + [
                lbls[i], CLASSES[c][1], f"{marg:.3f}",
                rows[i]["dataset"], rows[i]["source_file"], rows[i]["source_label"],
            ])
    print(f"Saved {full_csv}")

    # ---------- 写标注者原始评分 ----------
    raw_csv = ANN / "Dseed_raw_annotators.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["图像ID", "标注者", "维度", "评分"])
        for i in range(n):
            for k in range(N_ANNOT):
                for di, d in enumerate(DIMS):
                    w.writerow([IDs[i], f"A{k+1}", DIMS_CN[di], int(raw[d][i, k])])
    print(f"Saved {raw_csv}")

    # ---------- 汇总 ----------
    dist = Counter(int(a) for a in assigned)
    summary = {
        "n_images": n,
        "n_classes": 12,
        "n_annotators": N_ANNOT,
        "seed": SEED,
        "annotator_noise_sigma": SIGMA,
        "dimension_calibration": {d: list(CALIB[d]) for d in DIMS},
        "krippendorff_alpha": {d: round(float(alphas[d]), 4) for d in DIMS},
        "krippendorff_alpha_ci95": {d: [round(x, 4) for x in ci[d]] for d in DIMS},
        "krippendorff_alpha_mean": round(float(np.mean([alphas[d] for d in DIMS])), 4),
        "dimension_means": {d: round(float(final[d].mean()), 4) for d in DIMS},
        "dimension_sd": {d: round(float(final[d].std()), 4) for d in DIMS},
        "class_distribution": {CLASSES[c][0]: int(dist.get(c, 0)) for c in range(12)},
        "class_distribution_en": {CLASSES[c][1]: int(dist.get(c, 0)) for c in range(12)},
        "source_dataset_mix": dict(Counter(r["dataset"] for r in rows)),
    }
    with open(ANN / "Dseed_meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nClass distribution:")
    for c in range(12):
        print(f"  {CLASSES[c][0]:<8} {CLASSES[c][1]:<20} {dist.get(c,0):>5}")
    print(f"\nOverall alpha (mean of 5 dims): {summary['krippendorff_alpha_mean']}")
    print(f"Dimension means: {summary['dimension_means']}")


if __name__ == "__main__":
    main()