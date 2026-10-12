import jiwer
from whisper.normalizers import EnglishTextNormalizer

_norm = EnglishTextNormalizer()


def normalize(text):
    return _norm(text).strip()


def count_errors(ref, hyp):
    """Повертає (кількість помилок, кількість слів в еталоні)."""
    n_words = len(ref.split())
    if not hyp.strip():
        return n_words, n_words          # нічого не розпізнано: усі слова пропущено
    o = jiwer.process_words(ref, hyp)
    return o.substitutions + o.deletions + o.insertions, n_words