import numpy as np

from .loader import read_audio


def power(x):
    return float(np.mean(x.astype(np.float64) ** 2))


def pick_noise_chunk(noise, length, rng, min_power=1e-7, tries=30):
    """Випадковий шматок шуму потрібної довжини (тихі шматки відкидаємо)."""
    if len(noise) < length:
        noise = np.tile(noise, length // len(noise) + 1)
    chunk = None
    for _ in range(tries):
        start = int(rng.integers(0, len(noise) - length + 1))
        chunk = noise[start:start + length]
        if power(chunk) > min_power:
            return chunk
    return chunk


def make_white(length, rng):
    return rng.standard_normal(length).astype(np.float32)


def make_babble(pool, length, rng, n_speakers=5):
    """Гамір: накладання кількох випадкових фраз із пулу."""
    mix = np.zeros(length, dtype=np.float32)
    idx = rng.choice(len(pool), size=min(n_speakers, len(pool)), replace=False)
    for i in idx:
        audio = read_audio(pool[i]["path"])
        if len(audio) < length:
            audio = np.tile(audio, length // len(audio) + 1)
        start = int(rng.integers(0, len(audio) - length + 1))
        seg = audio[start:start + length]
        mix += seg / (np.sqrt(power(seg)) + 1e-8)
    return mix


def mix_at_snr(clean, noise, snr_db):
    """Масштабує шум так, щоб SNR дорівнював snr_db. Повертає (зашумлений, шум)."""
    p_clean = power(clean)
    p_noise = power(noise) + 1e-12
    k = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    scaled_noise = (k * noise).astype(np.float32)
    return (clean + scaled_noise).astype(np.float32), scaled_noise


def measure_snr(clean, scaled_noise):
    return 10 * np.log10(power(clean) / power(scaled_noise))


def peak_safe(x, limit=0.99):
    """Зменшує гучність усього запису, якщо є перевантаження (SNR не змінюється)."""
    peak = float(np.max(np.abs(x)))
    return x * (limit / peak) if peak > limit else x


def add_noise(clean, noise_type, snr_db, noises, rng, babble_pool=None, min_power=1e-7):
    """Головна функція: який запис, який шум, який рівень.

    noise_type: назва зі словника noises, або "white", або "babble".
    Повертає (зашумлений, шум із вже підібраною гучністю).
    """
    length = len(clean)
    if noise_type == "white":
        noise = make_white(length, rng)
    elif noise_type == "babble":
        if not babble_pool:
            raise ValueError("Для babble потрібен babble_pool (список фраз)")
        noise = make_babble(babble_pool, length, rng)
    elif noise_type in noises:
        noise = pick_noise_chunk(noises[noise_type], length, rng, min_power)
    else:
        raise KeyError(f"Невідомий тип шуму: {noise_type}")
    return mix_at_snr(clean, noise, snr_db)