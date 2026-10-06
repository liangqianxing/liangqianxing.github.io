#!/usr/bin/env python3
"""Build the small, responsive image assets used by the static Pages frontend.

Run from any directory with `python scripts/optimize-images.py`.
Requires Pillow; no Node dependency or build-time image service is needed.
Original public files are never modified. Content-addressed outputs stay under
public/media/, outside the public/images/ path that triggers Halo synchronization.

Extend RASTER_SOURCES or SVG_SOURCES when a new image is actually used by a page.
Commit the generated media and manifest with the component changes that use them.
Old hashed assets are kept so a cached page can still request its previous image.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = REPO_ROOT / "public"
MEDIA_ROOT = PUBLIC_ROOT / "media"
MANIFEST_PATH = REPO_ROOT / "data" / "image-assets.json"
WEBP_QUALITY = 84

# Explicitly scoped to raster images currently referenced by the Pages frontend.
# Widths are CSS-pixel breakpoints; 96/144 cover institution logos at 2x/3x DPR.
RASTER_SOURCES: dict[str, tuple[int, ...]] = {
    "/avatar.jpg": (160, 320, 640),
    "/logos/ecnu.png": (48, 96, 144),
    "/logos/westlake.png": (48, 96, 144),
}

# SVGs keep their original vector data. Dimensions reserve space before loading.
SVG_SOURCES: tuple[str, ...] = (
    "/logos/meituan.svg",
    "/images/posts/bpe-tokenizer-from-scratch/bpe-tokenizer-overview.svg",
    "/images/posts/bpe-tokenizer-from-scratch/bpe-utf8-byte-roundtrip.svg",
    "/images/posts/bpe-tokenizer-from-scratch/bpe-non-overlapping-merge.svg",
    "/images/posts/bpe-tokenizer-from-scratch/bpe-abab-training-steps.svg",
    "/images/posts/bpe-tokenizer-from-scratch/bpe-rank-priority.svg",
    "/images/posts/bpe-tokenizer-from-scratch/bpe-pretokenization-boundaries.svg",
)


def public_file(src: str) -> Path:
    """Resolve only site-root paths contained within public/."""
    if not src.startswith("/") or src.startswith("//"):
        raise ValueError(f"Expected a site-root image path: {src}")
    path = (PUBLIC_ROOT / src.lstrip("/")).resolve()
    path.relative_to(PUBLIC_ROOT.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_if_changed(path: Path, payload: bytes) -> None:
    if not path.exists() or path.read_bytes() != payload:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def raster_assets(src: str, widths: tuple[int, ...]) -> tuple[dict, list[dict]]:
    source = public_file(src)
    original_bytes = source.stat().st_size
    variants = []
    report = []
    with Image.open(source) as raw:
        # Rotate using EXIF first, then strip metadata from the encoded derivative.
        image = ImageOps.exif_transpose(raw)
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        image = image.convert("RGBA" if has_alpha else "RGB")
        width, height = image.size
        output_widths = sorted({min(width, candidate) for candidate in widths})
        for target_width in output_widths:
            target_height = max(1, round(height * target_width / width))
            resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=6)
            payload = buffer.getvalue()
            digest = hashlib.sha256(payload).hexdigest()[:12]
            filename = f"{source.stem}-{target_width}w-{digest}.webp"
            target = MEDIA_ROOT / filename
            write_if_changed(target, payload)
            variants.append({
                "src": f"/media/{filename}",
                "width": target_width,
                "mimeType": "image/webp",
            })
            report.append({
                "src": src,
                "width": target_width,
                "height": target_height,
                "original_bytes": original_bytes,
                "optimized_bytes": len(payload),
            })
    return {"width": width, "height": height, "variants": variants}, report


def svg_dimensions(src: str) -> dict:
    root = ET.parse(public_file(src)).getroot()
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError(f"Not an SVG document: {src}")

    def dimension(value: str | None) -> float | None:
        if not value or not re.fullmatch(r"\s*\d+(?:\.\d+)?(?:px)?\s*", value):
            return None
        return float(value.strip().removesuffix("px"))

    width = dimension(root.get("width"))
    height = dimension(root.get("height"))
    view_box = root.get("viewBox", "").replace(",", " ").split()
    if (width is None or height is None) and len(view_box) == 4:
        _, _, view_width, view_height = map(float, view_box)
        if width is None and height is None:
            width, height = view_width, view_height
        elif width is None:
            width = height * view_width / view_height
        else:
            height = width * view_height / view_width
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError(f"SVG has no valid intrinsic dimensions: {src}")
    return {
        "width": int(width) if width.is_integer() else width,
        "height": int(height) if height.is_integer() else height,
        "variants": [],
    }


def main() -> None:
    manifest = {}
    report = []
    for src, widths in RASTER_SOURCES.items():
        manifest[src], rows = raster_assets(src, widths)
        report.extend(rows)
    for src in SVG_SOURCES:
        manifest[src] = svg_dimensions(src)

    manifest_payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_if_changed(MANIFEST_PATH, manifest_payload)

    print("Responsive WebP files (each variant replaces one original download):")
    for row in report:
        saved = row["original_bytes"] - row["optimized_bytes"]
        percent = saved / row["original_bytes"] * 100
        print(
            f"  {row['src']} {row['width']}x{row['height']}: "
            f"{row['original_bytes']:,} -> {row['optimized_bytes']:,} bytes "
            f"({percent:.1f}% smaller)"
        )
    total_generated = sum(row["optimized_bytes"] for row in report)
    original_total = sum(public_file(src).stat().st_size for src in RASTER_SOURCES)
    print(f"Original raster source total: {original_total:,} bytes")
    print(f"All {len(report)} WebP files on disk: {total_generated:,} bytes")
    print("A browser downloads one matching variant per image, not every variant.")
    print(f"Registered {len(SVG_SOURCES)} original SVGs without rewriting them.")
    print(f"Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
