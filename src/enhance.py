import numpy as np
from scipy.signal import stft, istft
from functools import partial


def identity(x, sr=16000):
    """Без відновлення (контрольний варіант)."""
    return x


def spectral_subtraction(x, sr=16000, alpha=2.0, beta=0.05, nperseg=512, q=0.2):
    """Спектральне віднімання.

    Шум оцінюється як нижній квантиль потужності кожної частоти за часом,
    з поправкою на зміщення оцінки (для експоненційного розподілу).
    alpha відповідає за агресивність, beta за мінімальне посилення (проти "музичного шуму").
    """
    hop_overlap = nperseg * 3 // 4
    _, _, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=hop_overlap)
    p = np.abs(Z) ** 2
    noise = np.quantile(p, q, axis=1, keepdims=True) / (-np.log(1 - q))
    gain = np.sqrt(np.maximum(1.0 - alpha * noise / (p + 1e-12), beta ** 2))
    _, y = istft(Z * gain, fs=sr, nperseg=nperseg, noverlap=hop_overlap)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)

_demucs = {}

def demucs_dns64(x, sr=16000):
    """Demucs (facebookresearch/denoiser), модель dns64, готові ваги."""
    import torch
    from denoiser import pretrained
    from denoiser.dsp import convert_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "m" not in _demucs:
        _demucs["m"] = pretrained.dns64().to(device).eval()
    model = _demucs["m"]

    wav = torch.from_numpy(x).float().unsqueeze(0).to(device)       # (1, час)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    with torch.no_grad():
        out = model(wav[None])[0]                                    # (1, час)
    y = out.squeeze(0).cpu().numpy().astype(np.float32)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y

METHODS = {
    "none": identity,
    "spectral": spectral_subtraction,
    "demucs": demucs_dns64,
}


def get_method(name):
    if name not in METHODS:
        raise KeyError(f"Невідомий метод '{name}'. Доступні: {list(METHODS)}")
    return METHODS[name]

from .loader import read_audio


def apply_method(name, noisy_path):
    if name.startswith("pre_"):
        enhanced = str(noisy_path).replace("noisy_test", f"enhanced/{name[4:]}")
        return read_audio(enhanced)
    return get_method(name)(read_audio(noisy_path))