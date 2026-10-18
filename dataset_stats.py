"""Статистика вхідних даних для підрозділу 2.7 (без Whisper і без GPU).
Запуск з кореня проєкту:  python dataset_stats.py
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

# --- 1. метадані зашумлених записів
cands = glob.glob("**/metadata.csv", recursive=True)
if not cands:
    sys.exit("metadata.csv не знайдено: запустіть з кореня проєкту")
sizes = {c: len(pd.read_csv(c)) for c in cands}
print("знайдені metadata.csv:", sizes)
META = max(sizes, key=sizes.get)
meta = pd.read_csv(META)
print("metadata:", META, meta.shape)
print("колонки:", list(meta.columns))


def col(*keys):
    for c in meta.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None


c_noisy = col("noisy_path", "noisy")
c_clean = col("clean")
c_noise = col("noise")
c_snr = col("snr")
c_text = col("reference", "ref", "text", "transcript")
c_id = col("utt", "id", "phrase")
print("використано:", dict(noisy=c_noisy, clean=c_clean, noise=c_noise,
                           snr=c_snr, text=c_text, id=c_id))

# --- 2. кількість файлів
print("\nФайлів у метаданих:", len(meta))
print(meta.groupby([c_noise, c_snr]).size().unstack().to_string())

# --- 3. тривалість і довжина текстів (по унікальних фразах)
rows = []
first = meta[(meta[c_noise] == meta[c_noise].iloc[0]) & (meta[c_snr] == meta[c_snr].max())]
for _, r in first.iterrows():
    info = sf.info(r[c_noisy])
    words = len(str(r[c_text]).split()) if c_text else np.nan
    rows.append(dict(id=r[c_id] if c_id else "", dur=info.duration, sr=info.samplerate, words=words))
st = pd.DataFrame(rows)
print("\nФраз:", len(st), "| частота дискретизації:", sorted(st["sr"].unique()))
print(st[["dur", "words"]].describe().round(2).to_string())
print("Сумарна тривалість, с:", round(st["dur"].sum(), 1))
st.to_csv("results/dataset_stats.csv", index=False)

# --- 4. спектри шумів: шум = noisy - clean (форма спектра збігається з шумом)
bands = {"<500 Гц": (0, 500), "500-2000 Гц": (500, 2000), ">2000 Гц": (2000, 8000)}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
summ = []
for noise, g in meta[meta[c_snr] == 0].groupby(c_noise):
    psds = []
    for _, r in g.head(15).iterrows():
        y, sr = sf.read(r[c_noisy])
        s, _ = sf.read(r[c_clean])
        n = min(len(y), len(s))
        d = y[:n] - s[:n]
        f, p = welch(d, sr, nperseg=1024)
        psds.append(p)
    p = np.mean(psds, axis=0)
    tot = p.sum()
    ax.semilogx(f[1:], 10 * np.log10(p[1:] / tot + 1e-12), label=noise)
    cen = float((f * p).sum() / tot)
    row = {"noise": noise, "centroid_Hz": round(cen)}
    for b, (lo, hi) in bands.items():
        row[b] = round(100 * p[(f >= lo) & (f < hi)].sum() / tot, 1)
    summ.append(row)
ax.set_xlabel("Частота, Гц")
ax.set_ylabel("Відносна спектральна густина, дБ")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
os.makedirs("results/figures", exist_ok=True)
fig.tight_layout()
fig.savefig("results/figures/noise_spectra.png", dpi=300)
print("\nСпектральні характеристики шумів (частка потужності, %):")
print(pd.DataFrame(summ).to_string(index=False))

# --- 5. кількість оброблених файлів за методами
print("\nОброблені записи (results/enhanced):")
for d in sorted(glob.glob("results/enhanced/*/")):
    print(os.path.basename(d.rstrip("/\\")), len(glob.glob(d + "**/*.wav", recursive=True)))