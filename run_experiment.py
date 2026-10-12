import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

os.chdir(Path(__file__).parent)

from src.asr import transcribe
from src.config import load_config
from src.enhance import apply_method
from src.loader import read_audio
from src.metrics import count_errors, normalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="розмір Whisper")
    p.add_argument("--methods", nargs="+", default=["none", "spectral"])
    args = p.parse_args()

    cfg = load_config()
    model_name = args.model or cfg["asr"]["model"]
    lang = cfg["asr"]["language"]
    res_dir = Path(cfg["paths"]["results"])
    meta = pd.read_csv(res_dir / "noisy_test" / "metadata.csv")

    # чисті записи: один раз, без відновлення
    clean = meta.drop_duplicates("id")[["id", "clean_path", "text"]].copy()
    clean["noisy_path"] = clean["clean_path"]
    clean["noise"] = "clean"
    clean["snr"] = float("nan")
    clean["method"] = "none"

    noisy = pd.concat([meta.assign(method=m) for m in args.methods], ignore_index=True)
    allrows = pd.concat([clean, noisy], ignore_index=True)
    allrows["key"] = (allrows["id"] + "|" + allrows["noise"] + "|"
                      + allrows["snr"].astype(str) + "|" + allrows["method"])

    out = res_dir / f"experiment_{model_name}.csv"
    done = set(pd.read_csv(out)["key"]) if out.exists() else set()
    todo = allrows[~allrows["key"].isin(done)]
    print(f"Whisper '{model_name}', методи {args.methods}: "
          f"залишилось {len(todo)} із {len(allrows)}")

    for _, r in tqdm(todo.iterrows(), total=len(todo)):
        audio = apply_method(r["method"], r["noisy_path"])
        hyp = transcribe(audio, model_name, lang)
        ref_n, hyp_n = normalize(r["text"]), normalize(hyp)
        errors, words = count_errors(ref_n, hyp_n)
        pd.DataFrame([{
            "key": r["key"], "id": r["id"], "noise": r["noise"], "snr": r["snr"],
            "method": r["method"], "ref": ref_n, "hyp": hyp_n,
            "errors": errors, "words": words,
        }]).to_csv(out, mode="a", header=not out.exists(), index=False)

    # ---- підсумки ----
    df = pd.read_csv(out)
    c = df[df["noise"] == "clean"]
    print(f"\nWER на чистих записах: {100 * c['errors'].sum() / c['words'].sum():.1f}%")

    nz = df[(df["noise"] != "clean") & (df["method"].isin(args.methods))]
    s = nz.groupby(["method", "snr"])[["errors", "words"]].sum()
    s["wer"] = 100 * s["errors"] / s["words"]
    table = s["wer"].unstack("snr").round(1)
    print("\nWER (%) усереднено за всіма типами шуму:")
    print(table.to_string())
    nz = nz.assign(hyp_words=nz["hyp"].fillna("").astype(str).str.split().str.len())
    nz["halluc"] = nz["hyp_words"] > 2 * nz["words"]
    nz["utt_wer_clip"] = (nz["errors"] / nz["words"]).clip(upper=1.0)

    print("\nСередній WER по фразах, обмежений 100% (%):")
    print((100 * nz.groupby(["method", "snr"])["utt_wer_clip"].mean()).unstack("snr").round(1).to_string())

    print("\nКількість галюцинацій (hyp > 2× еталону):")
    print(nz.groupby(["method", "snr"])["halluc"].sum().unstack("snr").to_string())

    g = nz.groupby(["method", "noise", "snr"])[["errors", "words"]].sum()
    g["wer"] = (100 * g["errors"] / g["words"]).round(1)
    tbl_path = res_dir / f"table_{model_name}.csv"
    try:
        g["wer"].unstack("snr").to_csv(tbl_path)
    except PermissionError:
        print(f"Не вдалося записати {tbl_path}: закрийте файл в Excel і запустіть ще раз")

    fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for m, row in table.iterrows():
        plt.plot(row.index, row.values, marker="o", label=m)
    plt.xlabel("SNR, дБ")
    plt.ylabel("WER, %")
    plt.title(f"Whisper '{model_name}': порівняння методів")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"methods_{model_name}.png", dpi=150)
    print(f"\nГрафік: {fig_dir / f'methods_{model_name}.png'}")


if __name__ == "__main__":
    main()