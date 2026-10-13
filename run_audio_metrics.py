import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

os.chdir(Path(__file__).parent)

from src.config import load_config
from src.enhance import apply_method, is_available
from src.loader import read_audio

from pystoi import stoi

try:
    from pesq import pesq as pesq_fn
except Exception:
    pesq_fn = None
    print("Пакет pesq недоступний: PESQ пропускаємо, рахуємо STOI і SI-SDR")


def si_sdr(ref, est):
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))


def safe_pesq(ref, est, sr):
    if pesq_fn is None:
        return float("nan")
    try:
        return pesq_fn(sr, ref, est, "wb")
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["none", "spectral", "demucs"])
    args = p.parse_args()

    cfg = load_config()
    sr = cfg["sample_rate"]
    res_dir = Path(cfg["paths"]["results"])
    meta = pd.read_csv(res_dir / "noisy_test" / "metadata.csv")

    rows = pd.concat([meta.assign(method=m) for m in args.methods], ignore_index=True)
    rows["key"] = (rows["id"] + "|" + rows["noise"] + "|"
                   + rows["snr"].astype(str) + "|" + rows["method"])

    out = res_dir / "audio_metrics.csv"
    done = set(pd.read_csv(out)["key"]) if out.exists() else set()
    todo = rows[~rows["key"].isin(done)]
    keep = [i for i, m, p in zip(todo.index, todo["method"], todo["noisy_path"]) if is_available(m, p)]
    todo = todo.loc[keep]
    print(f"Залишилось обробити {len(todo)} із {len(rows)}")

    for _, r in tqdm(todo.iterrows(), total=len(todo)):
        clean = read_audio(r["clean_path"])
        est = apply_method(r["method"], r["noisy_path"])
        n = min(len(clean), len(est))
        clean, est = clean[:n].astype(np.float64), est[:n].astype(np.float64)
        pd.DataFrame([{
            "key": r["key"], "id": r["id"], "noise": r["noise"], "snr": r["snr"],
            "method": r["method"],
            "si_sdr": si_sdr(clean, est),
            "stoi": stoi(clean, est, sr, extended=False),
            "pesq": safe_pesq(clean, est, sr),
        }]).to_csv(out, mode="a", header=not out.exists(), index=False)

    df = pd.read_csv(out)
    for metric in ["si_sdr", "stoi", "pesq"]:
        print(f"\n{metric} (середнє):")
        print(df.pivot_table(index="method", columns="snr", values=metric, aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    main()