"""Regenerate the original BPE teaching diagrams with Python's standard library."""

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public/images/posts/bpe-tokenizer-from-scratch"
INK = "#203f43"
MUTED = "#65777b"
TEAL = "#216b70"
BLUE = "#416d91"
CORAL = "#a95d43"
PURPLE = "#79629c"
MINT = "#e7f3ef"
SKY = "#eaf2fa"
PEACH = "#fcebe1"
LILAC = "#f0eafa"


class Diagram:
    def __init__(self, number, title, subtitle, description, height):
        self.height = height
        self.parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" '
            f'height="{height}" viewBox="0 0 640 {height}" role="img" '
            'aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title>",
            f"<desc id=\"desc\">{escape(description)}</desc>",
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 1 L 8 5 L 0 9" fill="none" stroke="{MUTED}" '
            'stroke-width="1.6" stroke-linejoin="round"/></marker></defs>',
            '<g font-family="Segoe UI, PingFang SC, Microsoft YaHei, sans-serif">',
        ]
        self.rect(0, 0, 640, height, "#fffdf8", radius=0, stroke="none")
        self.text(36, 36, "BPE / 从零理解", size=17, color=TEAL, weight=600)
        self.text(36, 79, title, size=32, weight=700)
        self.text(36, 113, subtitle, size=21, color=MUTED)
        self.text(603, 47, f"{number:02}", size=24, color=PURPLE, anchor="end")

    def rect(self, x, y, width, height, fill, radius=18, stroke="#dbe5e5"):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )

    def text(self, x, y, value, size=26, color=INK, weight=400, anchor="start", mono=False):
        family = ' font-family="Consolas, DejaVu Sans Mono, Microsoft YaHei, monospace" xml:space="preserve"' if mono else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{family}>{escape(str(value))}</text>'
        )

    def line(self, x1, y1, x2, y2, arrow=False, dash=False, color=MUTED):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dashed = ' stroke-dasharray="6 6"' if dash else ""
        self.parts.append(
            f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
            f'stroke-width="2.4" stroke-linecap="round"{marker}{dashed}/>'
        )

    def path(self, d, color=MUTED, arrow=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-linejoin="round"{marker}/>'
        )

    def token(self, x, y, label, token_id=None, width=108, fill=MINT, color=TEAL):
        height = 88 if token_id is not None else 60
        self.rect(x, y, width, height, fill, radius=13, stroke="none")
        self.text(x + width / 2, y + 39, label, size=30, color=color, weight=600, anchor="middle", mono=True)
        if token_id is not None:
            self.text(x + width / 2, y + 70, f"ID {token_id}", size=20, color=MUTED, anchor="middle", mono=True)

    def save(self, name):
        self.line(36, self.height - 49, 604, self.height - 49, color="#dbe5e5")
        self.text(36, self.height - 23, "原创教学示意 · gu.log", size=16, color=MUTED)
        self.text(604, self.height - 23, "BPE TOKENIZER", size=15, color=MUTED, anchor="end", mono=True)
        self.parts.extend(["</g>", "</svg>"])
        target = OUT / name
        target.write_text("\n".join(self.parts) + "\n", encoding="utf-8")
        print(target.relative_to(ROOT).as_posix())


def overview():
    d = Diagram(1, "先学规则，再用规则", "训练建立词表；编码遵循已学到的合并顺序。", "训练将语料转为 UTF-8 字节，反复统计并合并最高频相邻对，得到词表和有序合并规则。编码按 rank 应用规则；解码拼接 token 对应的字节后统一解码。", 740)
    d.rect(32, 140, 576, 300, MINT)
    d.text(54, 179, "训练 / Learn", size=27, color=TEAL, weight=700)
    d.rect(54, 198, 532, 61, "#ffffff", radius=12, stroke="none")
    d.text(320, 238, "语料 → UTF-8 字节序列", size=27, anchor="middle")
    d.line(320, 266, 320, 283, arrow=True)
    d.rect(54, 291, 532, 61, "#ffffff", radius=12, stroke="none")
    d.text(320, 331, "统计相邻对 → 合并最高频对", size=26, anchor="middle")
    d.path("M 570 320 L 597 320 L 597 271 L 556 271", arrow=True)
    d.text(54, 390, "反复更新序列，直到达到词表上限", size=23, color=TEAL)
    d.text(54, 420, "无可合并对时也会提前停止", size=21, color=MUTED)
    d.line(320, 449, 320, 466, arrow=True)
    d.rect(32, 476, 576, 82, LILAC, stroke="none")
    d.text(320, 510, "训练产物", size=21, color=PURPLE, anchor="middle", weight=600)
    d.text(320, 544, "词表 + 有序合并规则（rank）", size=27, color=PURPLE, anchor="middle", weight=600)
    d.rect(32, 578, 576, 95, SKY, stroke="none")
    d.text(54, 616, "编码", size=25, color=BLUE, weight=700)
    d.text(143, 616, "文本 → 按 rank 合并 → token ID", size=25)
    d.text(54, 653, "解码", size=25, color=BLUE, weight=700)
    d.text(143, 653, "ID → 拼接全部字节 → 文本", size=25)
    d.save("bpe-tokenizer-overview.svg")


def utf8():
    d = Diagram(2, "一个字符，可以有多个字节", "图中的数字均为 UTF-8 字节的十进制值。", "A 的 UTF-8 字节为 65；中的字节为 228、184、173；笑脸的字节为 240、159、153、130。多字节字符不能逐字节解码，应先拼接完整字节序列再统一解码。", 670)
    for y, char, values, fill, color in [
        (146, "A", list("A".encode()), MINT, TEAL),
        (265, "中", list("中".encode()), SKY, BLUE),
        (384, "🙂", list("🙂".encode()), PEACH, CORAL),
    ]:
        d.rect(32, y, 576, 99, "#ffffff")
        d.text(81, y + 51, char, size=40, anchor="middle", weight=600)
        d.text(81, y + 82, f"{len(values)} 字节", size=19, color=MUTED, anchor="middle")
        d.line(133, y + 49, 160, y + 49, arrow=True)
        for i, value in enumerate(values):
            d.token(174 + i * 101, y + 20, value, width=91, fill=fill, color=color)
    d.rect(32, 508, 576, 95, MINT, stroke="none")
    d.text(320, 549, "[228, 184, 173] → “中”", size=30, color=TEAL, anchor="middle", weight=600)
    d.text(320, 582, "先拼接完整字节，再统一 decode", size=23, color=TEAL, anchor="middle")
    d.save("bpe-utf8-byte-roundtrip.svg")


def non_overlap():
    d = Diagram(3, "统计可重叠，合并不重叠", "同一位置只能属于一次合并。", "序列 a a a a 中相邻 a a 出现三次，统计时三个相邻窗口都计数。应用合并时从左到右选择不重叠的两个 a a，匹配后跳过两个位置，得到 x x。", 636)
    d.rect(32, 143, 576, 160, SKY, stroke="none")
    d.text(54, 183, "统计：滑动窗口，共 3 对", size=26, color=BLUE, weight=600)
    for i in range(4):
        d.token(62 + i * 134, 205, "a", width=112, fill="#ffffff", color=BLUE)
    for i, y in enumerate([273, 281, 289]):
        x = 118 + i * 134
        d.line(x, y, x + 134, y, color=BLUE)
    d.rect(32, 324, 576, 232, MINT, stroke="none")
    d.text(54, 364, "合并：从左向右，共 2 次", size=26, color=TEAL, weight=600)
    for i in range(4):
        d.token(62 + i * 134, 386, "a", width=112, fill="#ffffff")
    d.path("M 118 452 L 118 463 L 252 463 L 252 452", color=TEAL)
    d.path("M 386 452 L 386 463 L 520 463 L 520 452", color=TEAL)
    d.line(185, 469, 185, 482, arrow=True)
    d.line(453, 469, 453, 482, arrow=True)
    d.token(129, 487, "x", width=112)
    d.token(397, 487, "x", width=112)
    d.text(320, 574, "匹配成功：i += 2，不复用刚合并的字节", size=24, color=TEAL, anchor="middle")
    d.save("bpe-non-overlapping-merge.svg")


def training():
    seq = list(b"abab")
    counts = {pair: sum(tuple(seq[i:i + 2]) == pair for i in range(len(seq) - 1)) for pair in [(97, 98), (98, 97)]}
    assert counts == {(97, 98): 2, (98, 97): 1}
    d = Diagram(4, "“abab” 如何变成一个 token", "教学语料只有 “abab”，目标词表大小为 258。", "初始字节序列为 97、98、97、98。第一轮 a b 出现两次，b a 出现一次，合并 a b 为 token 256，得到 256、256。第二轮合并 256、256 为 257。词表保留 256 个基础字节，再增加 ab 与 abab 两项。", 804)
    d.rect(32, 142, 576, 148, "#ffffff")
    d.text(54, 181, "起点 · 256 个基础字节", size=25, color=MUTED, weight=600)
    for i, (label, tid) in enumerate(zip("abab", seq)):
        d.token(59 + i * 136, 196, label, tid, width=114)
    d.line(320, 297, 320, 322, arrow=True)
    d.rect(32, 332, 576, 190, SKY, stroke="none")
    d.text(54, 372, "第 1 轮 · (a, b) 频率最高", size=26, color=BLUE, weight=600)
    d.text(54, 408, "(97, 98) × 2     (98, 97) × 1", size=24, color=BLUE, mono=True)
    d.token(171, 424, "ab", 256, width=136, fill="#ffffff", color=BLUE)
    d.token(333, 424, "ab", 256, width=136, fill="#ffffff", color=BLUE)
    d.line(320, 529, 320, 554, arrow=True)
    d.rect(32, 564, 576, 157, PEACH, stroke="none")
    d.text(54, 604, "第 2 轮 · (256, 256) × 1", size=26, color=CORAL, weight=600)
    d.token(222, 619, "abab", 257, width=196, fill="#ffffff", color=CORAL)
    d.text(320, 739, "最终词表：256 个基础字节 + 2 个新 token", size=23, color=MUTED, anchor="middle")
    d.save("bpe-abab-training-steps.svg")


def ranks():
    d = Diagram(5, "编码时，rank 决定合并顺序", "沿用正文的三条规则，输入为 “abc”。", "rank 0 合并 b c 为 256，rank 1 合并 a b 为 257，rank 2 合并 a 与 256 为 258。正确编码先得到 a、bc，rank 1 不再匹配，再得到 abc。若忽略 rank 先合并 a b，则得到 ab、c，后续规则无法继续合并。这个错误分支不是最长词表匹配。", 853)
    d.rect(32, 142, 576, 156, LILAC, stroke="none")
    for i, rule in enumerate(["rank 0   b + c  → bc    ID 256", "rank 1   a + b  → ab    ID 257", "rank 2   a + bc → abc   ID 258"]):
        d.text(54, 186 + i * 42, rule, size=25, color=PURPLE, mono=True)
    d.rect(32, 321, 576, 290, MINT, stroke="none")
    d.text(54, 363, "按 rank：先 bc，再 abc", size=27, color=TEAL, weight=700)
    for i, (label, tid) in enumerate([("a", 97), ("b", 98), ("c", 99)]):
        d.token(75 + i * 172, 383, label, tid, width=146, fill="#ffffff")
    d.line(320, 478, 320, 499, arrow=True)
    d.text(320, 535, "[ a | bc ]  →  [ abc ]", size=30, color=TEAL, anchor="middle", weight=600, mono=True)
    d.text(320, 574, "rank 1 已无 a+b；rank 2 合并 a+bc", size=23, color=TEAL, anchor="middle")
    d.rect(32, 634, 576, 150, PEACH, stroke="none")
    d.text(54, 675, "忽略 rank：先合并 ab", size=27, color=CORAL, weight=700)
    d.text(320, 721, "[ ab | c ]  =  [257, 99]", size=30, color=CORAL, anchor="middle", mono=True)
    d.text(320, 760, "没有 ab+c 规则，无法得到 [258]", size=23, color=CORAL, anchor="middle")
    d.save("bpe-rank-priority.svg")


def boundaries():
    d = Diagram(6, "边界保留下来，BPE 各自进行", "示例：先允许 <eos>，再处理普通文本。", "输入 Hi,<eos> 世界!。允许 eos 特殊 token 时，先识别并隔离 eos；普通片段依字符类别拆成 Hi、逗号、空格、世界、感叹号。各片段独立应用 BPE，不能跨片段或特殊 token 边界。空格仍被保留；未允许的 eos 将按普通文本处理。", 742)
    d.rect(32, 143, 576, 95, "#ffffff")
    d.text(320, 186, "Hi,<eos> 世界!", size=36, anchor="middle", mono=True, weight=600)
    d.text(320, 216, 'allowed_special = {"<eos>"}', size=23, color=PURPLE, anchor="middle", mono=True)
    d.line(320, 245, 320, 269, arrow=True)
    d.text(36, 305, "① 先识别并隔离特殊 token", size=26, color=PURPLE, weight=600)
    d.token(32, 323, "Hi,", width=174, fill=SKY, color=BLUE)
    d.token(224, 323, "<eos>", width=192, fill=LILAC, color=PURPLE)
    d.token(434, 323, " 世界!", width=174, fill=SKY, color=BLUE)
    d.text(320, 415, "<eos> → 单个专用 token ID", size=25, color=PURPLE, anchor="middle")
    d.text(36, 460, "② 普通文本分段后，独立应用 BPE", size=25, color=BLUE, weight=600)
    for x, label, width, fill, color in [
        (32, "Hi", 96, SKY, BLUE), (144, ",", 64, SKY, BLUE),
        (224, "<eos>", 144, LILAC, PURPLE), (384, "␠", 64, PEACH, CORAL),
        (464, "世界", 80, SKY, BLUE), (560, "!", 48, SKY, BLUE),
    ]:
        d.token(x, 482, label, width=width, fill=fill, color=color)
    for x in [136, 216, 376, 456, 552]:
        d.line(x, 475, x, 551, dash=True, color=PURPLE)
    d.rect(32, 574, 576, 102, MINT, stroke="none")
    d.text(320, 611, "空格保留 · 不跨边界合并", size=28, color=TEAL, anchor="middle", weight=600)
    d.text(320, 646, "未允许的 <eos> 则按普通文本处理", size=23, color=TEAL, anchor="middle")
    d.save("bpe-pretokenization-boundaries.svg")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for render in [overview, utf8, non_overlap, training, ranks, boundaries]:
        render()
