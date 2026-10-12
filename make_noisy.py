import argparse
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

os.chdir(Path(__file__).parent)

from src.config import load_config
from src.loader import list_utterances, select_utterances, load_noises, read_audio
from src.mixer import add_noise, peak_safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, help="кількість фраз")
    p.add_argument("--noises", nargs="+", help="типи шуму")
    p.add_argument("--snr", nargs="+", type=int, help="рівні SNR")
    args = p.parse_args()

    cfg = load_config()
    sr = cfg["sample_rate"]
    n = args.n or cfg["data"]["n_utterances"]
    noise_types = args.noises or (cfg["noise"]["test_envs"] + cfg["noise"]["extra_types"])
    snrs = args.snr or cfg["noise"]["snr_db"]

    utts = list_utterances(cfg["paths"]["librispeech"], cfg["data"]["test_subset"])
    noises = load_noises(cfg["paths"]["noise"])
    rng = np.random.default_rng(cfg["seed"])
    chosen = select_utterances(
        utts, n, rng, cfg["data"]["min_duration"], cfg["data"]["max_duration"]
    )
    print(f"Фраз: {len(chosen)}, типів шуму: {len(noise_types)}, рівнів SNR: {len(snrs)}")

    out_root = Path(cfg["paths"]["results"]) / "noisy_test"
    rows = []
    for utt in chosen:
        clean = read_audio(utt["path"])
        # для babble беремо фрази інших дикторів, щоб голос цілі не потрапив у гамір
        pool = [u for u in utts if u["speaker"] != utt["speaker"]]
        for nt in noise_types:
            for snr in snrs:
                seed = zlib.crc32(f"{cfg['seed']}-{utt['id']}-{nt}-{snr}".encode())
                noisy, _ = add_noise(
                    clean, nt, snr, noises, np.random.default_rng(seed),
                    babble_pool=pool, min_power=cfg["noise"]["min_chunk_power"],
                )
                path = out_root / nt / f"snr{snr}" / f"{utt['id']}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, peak_safe(noisy), sr, subtype="PCM_16")
                rows.append({
                    "id": utt["id"], "noise": nt, "snr": snr,
                    "noisy_path": str(path), "clean_path": str(utt["path"]),
                    "text": utt["text"], "speaker": utt["speaker"],
                })
        print(f"  готово: {utt['id']}")

    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "metadata.csv", index=False)
    print(f"Збережено {len(rows)} файлів. Метадані: {out_root / 'metadata.csv'}")


if __name__ == "__main__":
    main()