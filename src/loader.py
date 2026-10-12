from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def read_audio(path, target_sr=TARGET_SR):
    """Читає файл, повертає моно-масив float32 з частотою target_sr."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def list_utterances(librispeech_root, subset="test-clean"):
    """Повертає список фраз: id, шлях до аудіо, еталонний текст, диктор."""
    root = Path(librispeech_root) / subset
    if not root.exists():
        raise FileNotFoundError(f"Не знайдено {root}. Перевірте шлях у configs/main.yaml")

    utts = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, text = line.split(" ", 1)
            path = trans.parent / f"{utt_id}.flac"
            if path.exists():
                utts.append({
                    "id": utt_id,
                    "path": path,
                    "text": text,
                    "speaker": utt_id.split("-")[0],
                })
    return utts


def select_utterances(utts, n, rng, min_dur=2.0, max_dur=15.0):
    """Випадково вибирає n фраз потрібної тривалості."""
    order = rng.permutation(len(utts))
    chosen = []
    for i in order:
        dur = sf.info(str(utts[i]["path"])).duration
        if min_dur <= dur <= max_dur:
            chosen.append(utts[i])
            if len(chosen) == n:
                break
    return chosen


def load_noises(noise_dir):
    """Завантажує шуми. Повертає словник {назва: масив}.

    Для папки береться ch01.wav (або перший wav у ній), для окремого файлу сам файл.
    """
    noise_dir = Path(noise_dir)
    if not noise_dir.exists():
        raise FileNotFoundError(f"Не знайдено {noise_dir}")

    noises = {}
    for item in sorted(noise_dir.iterdir()):
        if item.is_dir():
            wavs = sorted(item.rglob("ch01.wav")) or sorted(item.rglob("*.wav"))
            if wavs:
                noises[item.name] = read_audio(wavs[0])
        elif item.suffix.lower() in (".wav", ".flac"):
            noises[item.stem] = read_audio(item)
    return noises