#!/usr/bin/env python3
"""Cache approved remote friend avatars for the static Pages site.

Run once before optimize-images.py. Existing verified snapshots are reused;
--refresh explicitly fetches current avatars and retains old files for caches.
Neither the site build nor a visitor needs to contact these avatar providers.
Requires Pillow, like optimize-images.py.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = REPO_ROOT / "public"
CACHE_ROOT = PUBLIC_ROOT / "media" / "friends"
MANIFEST_PATH = REPO_ROOT / "data" / "friend-avatars.json"
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
REMOTE_AVATARS = {
    "https://github.com/liumengxuan04.png": "liumengxuan04",
    "https://github.com/chenchishui.png": "chenchishui",
}


def valid_snapshot(src: str | None) -> bool:
    if not src or not src.startswith("/media/friends/"):
        return False
    try:
        path = (PUBLIC_ROOT / src.lstrip("/")).resolve()
        path.relative_to(CACHE_ROOT.resolve())
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def fetch_avatar(source: str, name: str) -> str:
    request = Request(f"{source}?size=256", headers={"User-Agent": "gu.log-avatar-cache/1.0"})
    with urlopen(request, timeout=20) as response:
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Avatar exceeds the 5 MiB download limit")

    with Image.open(io.BytesIO(payload)) as raw:
        if raw.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
            raise ValueError("Unsupported avatar format")
        if raw.width * raw.height > 16_000_000:
            raise ValueError("Avatar dimensions exceed the decoding limit")
        image = ImageOps.exif_transpose(raw)
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        image = image.convert("RGBA" if has_alpha else "RGB")
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    output = buffer.getvalue()
    digest = hashlib.sha256(output).hexdigest()[:12]
    filename = f"friend-{name}-{digest}.png"
    target = CACHE_ROOT / filename
    if not target.exists() or target.read_bytes() != output:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".png.tmp")
        temporary.write_bytes(output)
        temporary.replace(target)
    return f"/media/friends/{filename}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Fetch current remote avatars")
    args = parser.parse_args()
    previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    snapshots = {}
    for source, name in REMOTE_AVATARS.items():
        cached = previous.get(source)
        if valid_snapshot(cached) and not args.refresh:
            snapshots[source] = cached
            print(f"Reused {name}: {cached}")
            continue
        try:
            snapshots[source] = fetch_avatar(source, name)
            print(f"Cached {name}: {snapshots[source]}")
        except Exception as error:
            if not valid_snapshot(cached):
                raise RuntimeError(f"Could not cache {name}; no valid previous snapshot") from error
            snapshots[source] = cached
            print(f"Kept previous {name} snapshot after download failure: {error}")

    payload = json.dumps(snapshots, ensure_ascii=False, indent=2) + "\n"
    if not MANIFEST_PATH.exists() or MANIFEST_PATH.read_text(encoding="utf-8") != payload:
        temporary = MANIFEST_PATH.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(MANIFEST_PATH)
    print("Next: python scripts/optimize-images.py")


if __name__ == "__main__":
    main()
