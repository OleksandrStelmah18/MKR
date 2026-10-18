"""Створення навчального та валідаційного наборів для дотренування SGMSE+.

Структура виводу (її очікує train.py з sgmse):
    <out>/train/clean/*.wav   <out>/train/noisy/*.wav
    <out>/valid/clean/*.wav   <out>/valid/noisy/*.wav
    <out>/metadata_train.csv  <out>/metadata_valid.csv

Принципи розділення даних (щоб тест залишався незалежним):
  * мовці: train-clean-100 та dev-clean не перетинаються з test-clean (властивість LibriSpeech);
  * шуми: три тестові середовища DEMAND (NRIVER, OMEETING, STRAFFIC) виключені;
    у навчанні використано решту середовищ DEMAND, а також нові реалізації гомону та білого шуму;
  * валідація: шум береться з останніх 20 % кожного запису, навчання з перших 80 %.

Запуск з кореня проєкту:
    python make_train_set.py --libri <LibriSpeech> --noise_dir <MKR_noise> --out <папка>
"""
import argparse
import csv
import os

import numpy as np
import soundfile as sf

from src.loader import list_utterances, load_noises, read_audio, select_utterances
from src.mixer import add_noise

TEST_NOISES = ("NRIVER", "OMEETING", "STRAFFIC")


def cap_per_speaker(utts, cap, rng):
    """Не більше cap фраз від одного диктора (різноманітність голосів)."""
    counts, picked = {}, []
    for i in rng.permutation(len(utts)):
        s = utts[i]["speaker"]
        if counts.get(s, 0) < cap:
            picked.append(utts[i])
            counts[s] = counts.get(s, 0) + 1
    return picked


def split_noises(noises, exclude, frac=0.8):
    train, valid = {}, {}
    for name, arr in noises.items():
        if name in exclude:
            continue
        cut = int(len(arr) * frac)
        train[name], valid[name] = arr[:cut], arr[cut:]
    return train, valid


def build(split, utts, noises, pool, n_draws, out, rng, snr_lo, snr_hi):
    cdir = os.path.join(out, split, "clean")
    ndir = os.path.join(out, split, "noisy")
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(ndir, exist_ok=True)
    types = list(noises) + ["babble", "white"]
    rows = []
    for n, u in enumerate(utts, 1):
        clean = read_audio(u["path"])
        pool_other = [p for p in pool if p["speaker"] != u["speaker"]]
        for k in range(n_draws):
            ntype = types[int(rng.integers(len(types)))]
            snr = float(rng.uniform(snr_lo, snr_hi))
            noisy, _ = add_noise(clean, ntype, snr, noises, rng, babble_pool=pool_other)
            c = clean
            peak = float(np.max(np.abs(noisy)))
            if peak > 0.99:                      # однаковий множник для обох записів
                g = 0.99 / peak
                noisy, c = noisy * g, clean * g
            name = f"{u['id']}_{k}.wav"
            sf.write(os.path.join(cdir, name), np.clip(c, -1, 1), 16000, subtype="PCM_16")
            sf.write(os.path.join(ndir, name), np.clip(noisy, -1, 1), 16000, subtype="PCM_16")
            rows.append(dict(split=split, file=name, id=u["id"], speaker=u["speaker"],
                             noise=ntype, snr=round(snr, 2)))
        if n % 250 == 0:
            print(f"{split}: {n}/{len(utts)} фраз", flush=True)
    with open(os.path.join(out, f"metadata_{split}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{split}: {len(rows)} пар збережено")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", required=True, help="папка LibriSpeech (містить train-clean-100, dev-clean)")
    ap.add_argument("--noise_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_valid", type=int, default=300)
    ap.add_argument("--draws", type=int, default=3, help="кількість шумових варіантів на фразу (навчання)")
    ap.add_argument("--cap", type=int, default=15, help="макс. фраз на одного диктора")
    ap.add_argument("--snr_min", type=float, default=-5.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    noises_all = load_noises(a.noise_dir)
    print("Знайдено шумів:", sorted(noises_all))
    tr_noises, va_noises = split_noises(noises_all, TEST_NOISES)
    print("Шуми для навчання:", sorted(tr_noises), "+ babble, white")

    tr_all = list_utterances(a.libri, "train-clean-100")
    va_all = list_utterances(a.libri, "dev-clean")
    tr = select_utterances(cap_per_speaker(tr_all, a.cap, rng), a.n_train, rng)
    va = select_utterances(cap_per_speaker(va_all, a.cap, rng), a.n_valid, rng)
    print(f"Фраз: навчання {len(tr)}, валідація {len(va)}")

    tr_pool = [tr_all[i] for i in rng.permutation(len(tr_all))[:300]]
    va_pool = [va_all[i] for i in rng.permutation(len(va_all))[:100]]

    build("train", tr, tr_noises, tr_pool, a.draws, a.out, rng, a.snr_min, a.snr_max)
    build("valid", va, va_noises, va_pool, 1, a.out, rng, a.snr_min, a.snr_max)


if __name__ == "__main__":
    main()