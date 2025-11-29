from pathlib import Path
text = Path("source/_posts/GGML.md").read_text(encoding="utf-8")
print(repr(text))
