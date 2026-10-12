import numpy as np
import torch
import whisper

_models = {}


def get_model(name="base"):
    if name not in _models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _models[name] = whisper.load_model(name, device=device)
    return _models[name]


def transcribe(audio, model_name="base", language="en"):
    """Аудіо (масив float32, 16 кГц) -> текст."""
    model = get_model(model_name)
    fp16 = next(model.parameters()).is_cuda
    result = model.transcribe(
        np.asarray(audio, dtype=np.float32),
        language=language,
        temperature=0.0,
        fp16=fp16,
        condition_on_previous_text=False,
    )
    return result["text"].strip()