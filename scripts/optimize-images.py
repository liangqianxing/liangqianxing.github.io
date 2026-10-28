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
FRIEND_AVATARS_PATH = REPO_ROOT / "data" / "friend-avatars.json"
WEBP_QUALITY = 84

# Explicitly scoped to raster images currently referenced by the Pages frontend.
# Widths are CSS-pixel breakpoints; 96/144 cover institution logos at 2x/3x DPR.
RASTER_SOURCES: dict[str, tuple[int, ...]] = {
    "/avatar.jpg": (160, 320, 640),
    "/logos/ecnu.png": (48, 96, 144),
    "/logos/westlake.png": (48, 96, 144),
    "/logos/pku.png": (48, 96, 144),
    "/logos/xju.jpg": (48, 96, 144),
    "/logos/csu.jpg": (48, 96, 144),
    "/images/friends/sssn.jpg": (56, 112, 168),
    "/media/publications/autofigure-edit-overview.png": (320, 640, 960),
    "/media/publications/deepreviewer-v2-overview.png": (320, 640, 960),
    "/images/posts/llm-mrd-distillation-auditable-review-agents/llm-mrd-calibration-distillation-20261016.webp": (320, 640, 960),
    "/images/posts/llm-mrd-distillation-auditable-review-agents/mrd-review-distill-cover-20261016.webp": (320, 640, 960, 1440),
    "/images/posts/xgrammar-concurrent-cache-generation-fix/xgrammar-cache-fix-cover-20261022.webp": (320, 640, 960, 1440),
    "/images/posts/multi-constraint-agent-routing-dag-eval/agent-dag-routing-cover-20261020.webp": (320, 640, 960, 1440),
    "/images/posts/qwen25-vl-sglang-verl-single-gpu-lab/qvl-lab-cover-20261010.webp": (320, 640, 960, 1440),
    "/images/posts/embodied-pickplace-lab-robustness/pickplace-grasp-cover-20261016.webp": (320, 640, 960, 1440),
    "/images/posts/editcache-video-dit-inference-acceleration/editcache-dit-cover-20261025.webp": (320, 640, 960, 1440),
    "/images/posts/in-situ-autoguidance-editable-scientific-illustration/autoguidance-autofigure-cover-20261024.webp": (320, 640, 960, 1440),
    "/images/posts/qwen25-sft-dpo-alignment-tax-ablation/qwen25-sft-dpo-cover-20261018.webp": (320, 640, 960, 1440),
    "/images/posts/llm-gateway-semantic-cache-dynamic-batching/llm-gateway-cover-20261019.webp": (320, 640, 960, 1440),
    "/images/posts/cuda-fused-rmsnorm-kernel/cuda-rmsnorm-cover-20261018.webp": (320, 640, 960, 1440),
    "/images/posts/deepscientist-tech-stack/deepscientist-cover-20261018.webp": (320, 640, 960, 1440),
    "/images/posts/minicode-source-analysis/minicode-cover-20261018.webp": (320, 640, 960, 1440),
    "/images/posts/agent-memory/agent-memory-cover-20261018.webp": (320, 640, 960, 1440),
    "/images/posts/cycleresearcher-post-training/cycleresearcher-cover-20261018.webp": (320, 640, 960, 1440),
    "/images/posts/backend-stack-deep-dive/backend-stack-cover-20261018.webp": (320, 640, 960, 1440),
    "/images/posts/qwen3-mid-post-train-mlx/qwen3-mlx-cover-20261018.webp": (320, 640, 960, 1440),
    "/images/posts/bpe-tokenizer-from-scratch/bpe-cover.webp": (320, 640, 960, 1440),
    "/images/posts/bpe-tokenizer-from-scratch/bpe-merge-process.webp": (640, 960, 1440),
    "/images/posts/transformer-from-scratch/transformer-cover.webp": (320, 640, 960, 1440),
    "/images/posts/transformer-from-scratch/transformer-architecture.webp": (640, 960, 1440),
    "/images/posts/transformer-from-scratch/transformer-causal-mask.webp": (640, 960, 1440),
    "/images/posts/lm-lab-project-complete/lm-lab-cover.webp": (320, 640, 960, 1440),
    "/images/posts/lm-lab-project-complete/lm-lab-training-cycle.webp": (640, 960, 1440),
    "/images/posts/cmu-15-779-pytorch-autograd-linear-regression/cmu-pytorch-autograd-cover-20261015.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-pytorch-linear-module-relu/cmu-pytorch-relu-cover-20261015.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-pytorch-dataloader-checkpoints-batching/cmu-pytorch-batching-cover-20261015.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-pytorch-linear-relu-absolute-value/cmu-pytorch-absolute-cover-20261015.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-profiler-operator-batching/cmu-pytorch-profiling-cover-20261015.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-profiler-operator-batching/cmu-batch-size-sweep-20261015.webp": (640, 960, 1440, 1920),
    "/images/posts/cmu-15-779-cuda-thread-indexing/cmu-cuda-indexing-cover-20261015.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-matmul-tiling-tree-reduction/cmu-matmul-reduction-cover-20261016.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-attention-online-softmax/cmu-online-attention-cover-20261016.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-cpu-compiler-optimizations/cmu-cpu-compiler-cover-20261019.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-cpu-parallelism-pipeline/cmu-parallel-cover-20261019.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-kv-serving-speculative-sampling/cmu-serving-cover-20261019.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-lora-uniform-quantization/cmu-lora-cover-20261019.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-moe-routing-capacity/cmu-moe-cover-20261019.webp": (320, 640, 960, 1440),
    "/images/posts/cmu-15-779-kv-cache-granularity-study/cmu-kv-granularity-architecture-20261019.png": (640, 960, 1440, 1920),
    "/images/posts/cmu-15-779-kv-cache-granularity-study/cmu-kv-granularity-storage-20261019.png": (640, 960, 1440, 1920),
    "/images/posts/cmu-15-779-kv-cache-granularity-study/cmu-kv-granularity-latency-20261019.png": (640, 960, 1440, 1920),
    "/images/posts/cs336-linear-embedding-shapes/cs336-linear-embedding-shapes-cover.webp": (320, 640, 960, 1440),
    "/images/posts/cs336-rmsnorm-swiglu/cs336-rmsnorm-swiglu-cover.webp": (320, 640, 960, 1440),
    "/images/posts/cs336-softmax-attention-mask/cs336-softmax-attention-mask-cover.webp": (320, 640, 960, 1440),
    "/images/posts/cs336-multihead-attention-shapes/cs336-multihead-attention-shapes-cover.webp": (320, 640, 960, 1440),
    "/images/posts/cs336-rope-rotary-position-embedding/cs336-rope-rotary-position-embedding-cover.webp": (320, 640, 960, 1440),
    "/images/posts/cs336-pre-norm-transformer-block-lm/cs336-pre-norm-transformer-block-lm-cover.webp": (320, 640, 960, 1440),
    "/images/posts/cs336-bpe-tokenizer-training-encoding/cs336-bpe-tokenizer-training-encoding-cover.webp": (320, 640, 960, 1440),
    "/images/posts/cs336-cross-entropy-adamw-training-loop/cs336-cross-entropy-adamw-training-loop-cover.webp": (320, 640, 960, 1440),
}

# Remote friend sources are downloaded explicitly, never during site builds.
# Their verified local snapshots are also retained as the fallback originals.
for snapshot in json.loads(FRIEND_AVATARS_PATH.read_text(encoding="utf-8")).values():
    RASTER_SOURCES[snapshot] = (56, 112, 168)

# SVGs keep their original vector data. Dimensions reserve space before loading.
SVG_SOURCES: tuple[str, ...] = (
    "/logos/meituan.svg",
    "/images/posts/cmu-15-779-cuda-thread-indexing/cmu-cuda-thread-map-20261015.svg",
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
