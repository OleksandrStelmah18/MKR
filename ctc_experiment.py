"""Експеримент 2: ті самі записи, інший розпізнавач (NeMo FastConformer-CTC).
Результат: results/experiment_ctc.csv у тому самому форматі, що experiment_small.csv.

Спектральне віднімання та Demucs беруться з results/enhanced_local (prepare_enhanced.py),
методи pre_* з results/enhanced.
Використання:  python ctc_experiment.py --methods none spectral demucs pre_sgmse_vb ...
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).parent)

from src.config import load_config
from src.enhance import _enhanced_path
from src.metrics import count_errors, normalize

NEMO_PY = os.environ.get("NEMO_PY", "/content/nemo_env/bin/python")
ASR_SCRIPT = os.environ.get("CTC_ASR_SCRIPT", "ctc_asr.py")
LOCAL = "enhanced_local"


def audio_path(method, noisy_path):
    if method == "none":
        return str(noisy_path)
    if method.startswith("pre_"):
        return _enhanced_path(method, noisy_path)
    return str(noisy_path).replace("noisy_test", f"{LOCAL}/{method}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["none"])
    p.add_argument("--tag", default="ctc")
    args = p.parse_args()

    cfg = load_config()
    res_dir = Path(cfg["paths"]["results"])
    meta = pd.read_csv(res_dir / "noisy_test" / "metadata.csv")

    clean = meta.drop_duplicates("id")[["id", "clean_path", "text"]].copy()
    clean["noisy_path"] = clean["clean_path"]
    clean["noise"] = "clean"
    clean["snr"] = float("nan")
    clean["method"] = "none"
    noisy = pd.concat([meta.assign(method=m) for m in args.methods], ignore_index=True)
    allrows = pd.concat([clean, noisy], ignore_index=True)
    allrows["key"] = (allrows["id"] + "|" + allrows["noise"] + "|"
                      + allrows["snr"].map(str) + "|" + allrows["method"])
    allrows = allrows.drop_duplicates("key")
    allrows["path"] = [audio_path(m, pth) for m, pth in zip(allrows["method"], allrows["noisy_path"])]

    out = res_dir / f"experiment_{args.tag}.csv"
    raw = res_dir / f"{args.tag}_raw.jsonl"
    done = set(pd.read_csv(out)["key"]) if out.exists() else set()
    todo = allrows[~allrows["key"].isin(done)]
    exists = [Path(x).exists() for x in todo["path"]]
    missing = int(len(todo) - sum(exists))
    todo = todo[exists]
    print(f"{args.tag}, методи {args.methods}: розпізнати {len(todo)}; "
          f"немає файлів (не оброблялись) {missing}")

    if len(todo):
        jobs = [{"key": k, "path": pth} for k, pth in zip(todo["key"], todo["path"])]
        jobs_file = res_dir / f"{args.tag}_jobs.json"
        jobs_file.write_text(json.dumps(jobs), encoding="utf-8")
        subprocess.run([NEMO_PY, ASR_SCRIPT, str(jobs_file), str(raw)], check=True)

        hyps = {}
        for line in open(raw, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                hyps[d["key"]] = d["hyp"]
        rows = []
        for _, r in todo.iterrows():
            if r["key"] not in hyps:
                continue
            ref_n, hyp_n = normalize(r["text"]), normalize(hyps[r["key"]])
            errors, words = count_errors(ref_n, hyp_n)
            rows.append({"key": r["key"], "id": r["id"], "noise": r["noise"], "snr": r["snr"],
                         "method": r["method"], "ref": ref_n, "hyp": hyp_n,
                         "errors": errors, "words": words})
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)

    df = pd.read_csv(out)
    c = df[df["noise"] == "clean"]
    print(f"\nWER на чистих записах: {100 * c['errors'].sum() / c['words'].sum():.1f}%")
    nz = df[(df["noise"] != "clean") & (df["method"].isin(args.methods))].copy()
    s = nz.groupby(["method", "snr"])[["errors", "words"]].sum()
    s["wer"] = 100 * s["errors"] / s["words"]
    print("\nWER (%) усереднено за всіма типами шуму:")
    print(s["wer"].unstack("snr").round(1).to_string())
    nz["halluc"] = nz["hyp"].fillna("").astype(str).str.split().str.len() > 2 * nz["words"]
    print("\nКількість галюцинацій (hyp > 2x еталону):")
    print(nz.groupby(["method", "snr"])["halluc"].sum().unstack("snr").to_string())


if __name__ == "__main__":
    main()