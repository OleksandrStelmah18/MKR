"""Зберігає результат спектрального віднімання та Demucs у results/enhanced_local/<метод>/...
(структура як у noisy_test), щоб інший розпізнавач міг читати готові файли.
Використання:  python prepare_enhanced.py [spectral] [demucs]
"""
import os
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

os.chdir(Path(__file__).parent)

from src.enhance import get_method
from src.loader import read_audio

LOCAL = "enhanced_local"


def main():
    methods = sys.argv[1:] or ["spectral", "demucs"]
    meta = pd.read_csv("results/noisy_test/metadata.csv")
    for name in methods:
        f = get_method(name)
        done = 0
        for p in tqdm(meta["noisy_path"], desc=name):
            out = Path(str(p).replace("noisy_test", f"{LOCAL}/{name}"))
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            y = f(read_audio(p))
            sf.write(out, y, 16000, subtype="FLOAT")
            done += 1
        print(f"{name}: оброблено {done}, усього файлів {len(meta)}")


if __name__ == "__main__":
    main()