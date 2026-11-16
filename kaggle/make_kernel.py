"""Inline service/server.py and service/load_test.py into bench/bench.py -> kaggle/kernel.py (single-file Kaggle script)."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = (root / "bench" / "bench.py").read_text(encoding="utf-8")
for key, f in (("__SERVER_SRC__", "server.py"), ("__LOAD_SRC__", "load_test.py")):
    body = (root / "service" / f).read_text(encoding="utf-8")
    assert "'''" not in body
    src = src.replace(key, body)
import sys
if len(sys.argv) > 1:  # re-run only some stages: python make_kernel.py env,ollama,...
    src = src.replace('OUT = os.environ.get("OUT_DIR", "/kaggle/working")',
                      f'os.environ.setdefault("ONLY", "{sys.argv[1]}")\nOUT = os.environ.get("OUT_DIR", "/kaggle/working")', 1)
(root / "kaggle" / "kernel.py").write_text(src, encoding="utf-8")
print("wrote kaggle/kernel.py", len(src))
