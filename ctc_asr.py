"""Розпізнавання моделлю NVIDIA NeMo (FastConformer-CTC). Запускається в окремому
середовищі з NeMo.  Використання:  python ctc_asr.py jobs.json raw.jsonl
jobs.json: список {"key", "path", ...}; raw.jsonl: результат (додається, вже готові key пропускаються).
"""
import json
import os
import sys

import nemo.collections.asr as nemo_asr

MODEL = os.environ.get("CTC_MODEL", "stt_en_fastconformer_ctc_xlarge")
BATCH = 64

jobs = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2]
done = set()
if os.path.exists(out):
    done = {json.loads(l)["key"] for l in open(out, encoding="utf-8") if l.strip()}
todo = [j for j in jobs if j["key"] not in done]
print(f"лишилось розпізнати: {len(todo)} із {len(jobs)}", flush=True)
if not todo:
    sys.exit(0)

m = nemo_asr.models.ASRModel.from_pretrained(MODEL).cuda().eval()
try:  # швидше декодування, результат той самий
    from omegaconf import open_dict
    cfg = m.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
    m.change_decoding_strategy(cfg)
except Exception as e:
    print("decoding strategy не змінено:", e)

with open(out, "a", encoding="utf-8") as f:
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        res = m.transcribe([j["path"] for j in chunk], batch_size=16)
        if isinstance(res, tuple):
            res = res[0]
        for j, h in zip(chunk, res):
            f.write(json.dumps({"key": j["key"], "hyp": getattr(h, "text", h)}, ensure_ascii=False) + "\n")
        f.flush()
        print(f"{i + len(chunk)} / {len(todo)}", flush=True)