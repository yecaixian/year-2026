"""Step 1: 对 2000 张图抽取可复现的图像特征。

输出: data/annotations/features.csv  (每行一张图, 20+ 维特征)
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BASE = Path(r"C:\Users\lenovo\WorkBuddy\2026-09-21-20-29-42\data\processed")
OUT = Path(r"C:\Users\lenovo\WorkBuddy\2026-09-21-20-29-42\data\annotations")
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 256
EPS = 1e-9


def entropy(hist, bins):
    h = hist.astype(np.float64)
    s = h.sum()
    if s <= 0:
        return 0.0
    p = h / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / math.log2(bins))


def rgb_to_hsv_np(rgb):
    """rgb: HxWx3 in [0,1] -> hsv HxWx3, h in [0,1)"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn
    # hue
    h = np.zeros_like(mx)
    mask = diff > EPS
    idx = (mx == r) & mask
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6
    idx = (mx == g) & mask
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2
    idx = (mx == b) & mask
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4
    h = h / 6.0
    s = np.where(mx > EPS, diff / np.maximum(mx, EPS), 0.0)
    v = mx
    return h, s, v


def sobel(gray):
    """简单 Sobel 梯度"""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[1:-1, 1:-1] = (
        kx[0, 0] * gray[:-2, :-2] + kx[0, 1] * gray[:-2, 1:-1] + kx[0, 2] * gray[:-2, 2:]
        + kx[1, 0] * gray[1:-1, :-2] + kx[1, 2] * gray[1:-1, 2:]
        + kx[2, 0] * gray[2:, :-2] + kx[2, 1] * gray[2:, 1:-1] + kx[2, 2] * gray[2:, 2:]
    )
    gy[1:-1, 1:-1] = (
        ky[0, 0] * gray[:-2, :-2] + ky[0, 1] * gray[:-2, 1:-1] + ky[0, 2] * gray[:-2, 2:]
        + ky[1, 0] * gray[1:-1, :-2] + ky[1, 2] * gray[1:-1, 2:]
        + ky[2, 0] * gray[2:, :-2] + ky[2, 1] * gray[2:, 1:-1] + ky[2, 2] * gray[2:, 2:]
    )
    return gx, gy


def laplacian_abs(gray):
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    return np.abs(lap)


def extract(path: Path):
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.float64) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    h, s, v = rgb_to_hsv_np(rgb)

    f = {}

    # --- 色彩 ---
    f["sat_mean"] = float(s.mean())
    f["sat_std"] = float(s.std())
    hbins = np.histogram(h, bins=36, range=(0, 1))[0]
    f["hue_entropy"] = entropy(hbins, 36)
    f["hue_conc"] = float(hbins.max() / max(1, hbins.sum()))
    # 色相环形集中度 (dominant hue +/- 15 deg = +/-1.5 bins of 36)
    k = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    kb = np.convolve(np.r_[hbins[-2:], hbins, hbins[:2]], k, mode="same")[2:-2]
    f["hue_conc_w"] = float(kb.max() / max(1, hbins.sum()))

    # --- 灰度 ---
    f["gray_entropy"] = entropy(np.histogram(gray, bins=256, range=(0, 1))[0], 256)
    f["gray_std"] = float(gray.std())
    f["dynamic_range"] = float(np.percentile(gray, 95) - np.percentile(gray, 5))

    # --- 边缘 / 方向 ---
    gx, gy = sobel(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    f["edge_mag_mean"] = float(mag.mean())
    thr = 0.12
    f["edge_density"] = float((mag > thr).mean())
    orient = (np.arctan2(gy, gx) + np.pi) % np.pi  # [0, pi)
    oidx = np.clip((orient / np.pi * 18).astype(int), 0, 17)
    obins = np.zeros(18, dtype=np.float64)
    w = mag.ravel()
    np.add.at(obins, oidx.ravel(), w)
    f["orient_entropy"] = entropy(obins, 18)
    tot = obins.sum()
    f["orient_coherence"] = float(np.abs(obins / (tot + EPS)).max())

    # --- 高频 / 平滑 ---
    lap = laplacian_abs(gray)
    f["hf_energy"] = float(lap.mean())
    f["flat_ratio"] = float((lap < 0.02).mean())

    # --- 对称性 ---
    f["sym_h"] = float(1.0 - np.abs(gray - gray[:, ::-1]).mean())
    f["sym_v"] = float(1.0 - np.abs(gray - gray[::-1, :]).mean())

    # --- 块效应 (8px 网格边界 vs 内部) ---
    d_h = np.abs(np.diff(gray, axis=1))
    bcol = d_h[:, 7::8].mean() if d_h[:, 7::8].size else 0.0
    icol = np.delete(d_h, np.s_[7::8], axis=1).mean()
    d_v = np.abs(np.diff(gray, axis=0))
    brow = d_v[7::8, :].mean() if d_v[7::8, :].size else 0.0
    irow = np.delete(d_v, np.s_[7::8], axis=0).mean()
    f["blockiness"] = float(((bcol - icol) + (brow - irow)) * 0.5)

    # --- 量化色彩数 ---
    q = (rgb * 15).astype(np.uint8)
    qcode = q[..., 0].astype(np.int32) * 256 + q[..., 1].astype(np.int32) * 16 + q[..., 2]
    f["color_count_q"] = float(len(np.unique(qcode)) / 4096.0)
    f["unique_ratio"] = float(len(np.unique(qcode)) / qcode.size)

    # --- 频谱斜率 (log-log 径向平均) ---
    g0 = gray - gray.mean()
    F = np.fft.fftshift(np.abs(np.fft.fft2(g0)) ** 2)
    cy, cx = SIZE // 2, SIZE // 2
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = SIZE // 2 - 1
    ps = np.bincount(rr.ravel(), weights=F.ravel(), minlength=rmax + 1)[: rmax + 1]
    cnt = np.bincount(rr.ravel(), minlength=rmax + 1)[: rmax + 1]
    ps = ps / np.maximum(cnt, 1)
    lo, hi = 4, rmax
    fr = np.arange(lo, hi)
    pv = ps[lo:hi]
    ok = pv > 0
    if ok.sum() > 5:
        slope = float(np.polyfit(np.log(fr[ok]), np.log(pv[ok]), 1)[0])
    else:
        slope = -2.0
    f["spectral_slope"] = slope
    f["frac_dim_prox"] = float(-abs(slope + 1.0))  # 越接近 0 越像 1/f 分形

    # --- 自相关周期峰值 ---
    gz = (gray - gray.mean()) / (gray.std() + EPS)
    Fc = np.fft.ifft2(np.abs(np.fft.fft2(gz)) ** 2).real
    ac = np.fft.fftshift(Fc) / (SIZE * SIZE)
    m = 8
    win = ac[cy - m:cy + m + 1, cx - m:cx + m + 1].copy()
    win[m, m] = 0.0
    f["autocorr_peak"] = float(win.max())

    # --- 通道错位 (glitch 代理) ---
    rb = np.asarray(im.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float64) / 255.0
    f["channel_misalign"] = float(np.abs(rb[..., 0] - rb[..., 1]).mean())

    # --- 中心-周边边缘能量比 (主题显著性代理) ---
    q1 = SIZE // 4
    core = mag[q1:-q1, q1:-q1].mean()
    ring = (mag.sum() - mag[q1:-q1, q1:-q1].sum()) / (mag.size - mag[q1:-q1, q1:-q1].size)
    f["center_edge_ratio"] = float(core / (ring + EPS))

    # --- 斑点密度 (孤立高对比小结构) ---
    lm = lap > np.percentile(lap, 99)
    f["speckle_ratio"] = float(lm.mean())

    return f


def main():
    rows = []
    tasks = []
    for ds in ["wikiart", "artbench"]:
        with open(BASE / ds / "metadata.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                tasks.append((ds, r))

    print(f"Total images: {len(tasks)}")
    for i, (ds, r) in enumerate(tasks):
        p = BASE / ds / r["path"]
        try:
            f = extract(p)
        except Exception as e:
            print(f"FAIL {p}: {e}", file=sys.stderr)
            continue
        f["dataset"] = ds
        f["source_file"] = r["path"]
        if ds == "wikiart":
            f["source_label"] = r["style"]
            f["source_extra"] = f"{r['artist']}|{r['genre']}"
        else:
            f["source_label"] = r["class"]
            f["source_extra"] = r.get("prompt", "")
        rows.append(f)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(tasks)}")

    cols = list(rows[0].keys())
    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} feature rows -> {OUT/'features.csv'}")


if __name__ == "__main__":
    main()