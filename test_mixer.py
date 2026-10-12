import os
from pathlib import Path

import numpy as np
import soundfile as sf

os.chdir(Path(__file__).parent)   # щоб відносні шляхи працювали з будь-якого запуску

from src.config import load_config
from src.loader import list_utterances, load_noises, read_audio
from src.mixer import add_noise, measure_snr, peak_safe

cfg = load_config()
rng = np.random.default_rng(cfg["seed"])
sr = cfg["sample_rate"]

utts = list_utterances(cfg["paths"]["librispeech"], cfg["data"]["test_subset"])
noises = load_noises(cfg["paths"]["noise"])
print(f"Фраз у тестовому наборі: {len(utts)}")
print(f"Завантажено шумів: {len(noises)}")
for name, arr in noises.items():
    print(f"  {name}: {len(arr) / sr:.0f} с")

# одна випадкова фраза
utt = utts[int(rng.integers(len(utts)))]
clean = read_audio(utt["path"])
print(f"\nФраза {utt['id']} ({len(clean) / sr:.1f} с): {utt['text']}")

out = Path(cfg["paths"]["results"]) / "examples"
out.mkdir(parents=True, exist_ok=True)
sf.write(out / f"{utt['id']}_clean.wav", clean, sr)

# перший шум зі списку, гамір і білий шум
test_types = [next(iter(noises))] + cfg["noise"]["extra_types"]

print(f"\n{'шум':<12}{'заданий SNR':>12}{'виміряний SNR':>15}")
for noise_type in test_types:
    for snr in [20, 10, 0, -5]:
        noisy, noise_part = add_noise(
            clean, noise_type, snr, noises, rng,
            babble_pool=utts, min_power=cfg["noise"]["min_chunk_power"],
        )
        print(f"{noise_type:<12}{snr:>12}{measure_snr(clean, noise_part):>15.2f}")
        sf.write(out / f"{utt['id']}_{noise_type}_{snr}dB.wav", peak_safe(noisy), sr)

print(f"\nФайли збережено в {out}")