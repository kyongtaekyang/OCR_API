from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "writing_benchmark_user_guide.md"
OUTPUT = ROOT / "writing_benchmark_user_guide.pdf"


def _register_korean_font() -> tuple[str, str]:
    candidates = [
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        Path(r"C:\Windows\Fonts\gulim.ttc"),
    ]
    regular = next((p for p in candidates if p.exists()), None)
    if regular is None:
        return "Helvetica", "Helvetica-Bold"

    pdfmetrics.registerFont(TTFont("KoreanRegular", str(regular)))
    bold = Path(r"C:\Windows\Fonts\malgunbd.ttf")
    if bold.exists():
        pdfmetrics.registerFont(TTFont("KoreanBold", str(bold)))
        return "KoreanRegular", "KoreanBold"
    return "KoreanRegular", "KoreanRegular"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _inline_markup(text: str) -> str:
    text = _escape(text)
    text = re.sub(r"`([^`]+)`", r"<font backColor='#f1f5f9'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def _styles():
    regular, bold = _register_korean_font()
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "TitleKo",
        parent=base["Title"],
        fontName=bold,
        fontSize=22,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
        wordWrap="CJK",
    )
    h1 = ParagraphStyle(
        "Heading1Ko",
        parent=base["Heading1"],
        fontName=bold,
        fontSize=15,
        leading=22,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=10,
        spaceAfter=7,
        wordWrap="CJK",
    )
    h2 = ParagraphStyle(
        "Heading2Ko",
        parent=base["Heading2"],
        fontName=bold,
        fontSize=12.5,
        leading=18,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=5,
        wordWrap="CJK",
    )
    body = ParagraphStyle(
        "BodyKo",
        parent=base["BodyText"],
        fontName=regular,
        fontSize=9.5,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5,
        wordWrap="CJK",
    )
    bullet = ParagraphStyle(
        "BulletKo",
        parent=body,
        leftIndent=12,
        firstLineIndent=-8,
    )
    code = ParagraphStyle(
        "CodeKo",
        parent=base["Code"],
        fontName=regular,
        fontSize=8.2,
        leading=11,
        leftIndent=6,
        rightIndent=6,
        borderWidth=0.5,
        borderColor=colors.HexColor("#cbd5e1"),
        borderPadding=6,
        backColor=colors.HexColor("#f8fafc"),
        textColor=colors.HexColor("#0f172a"),
        wordWrap="CJK",
    )
    small = ParagraphStyle(
        "SmallKo",
        parent=body,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
    )
    return {
        "title": title,
        "h1": h1,
        "h2": h2,
        "body": body,
        "bullet": bullet,
        "code": code,
        "small": small,
    }


def _build_flowables(markdown: str):
    styles = _styles()
    story = []
    in_code = False
    code_lines: list[str] = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            story.append(Preformatted("\n".join(code_lines), styles["code"]))
            story.append(Spacer(1, 4))
            code_lines = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            story.append(Spacer(1, 3))
            continue

        if line.startswith("# "):
            story.append(Paragraph(_inline_markup(line[2:].strip()), styles["title"]))
            story.append(Spacer(1, 5))
        elif line.startswith("## "):
            story.append(Paragraph(_inline_markup(line[3:].strip()), styles["h1"]))
        elif line.startswith("### "):
            story.append(Paragraph(_inline_markup(line[4:].strip()), styles["h2"]))
        elif line == "---":
            story.append(PageBreak())
        elif line.startswith("- "):
            story.append(Paragraph("• " + _inline_markup(line[2:].strip()), styles["bullet"]))
        elif re.match(r"^\d+\.\s+", line):
            story.append(Paragraph(_inline_markup(line), styles["bullet"]))
        else:
            story.append(Paragraph(_inline_markup(line), styles["body"]))

    if in_code:
        flush_code()

    return story


def _draw_footer(canvas, doc):
    canvas.saveState()
    regular, _ = _register_korean_font()
    canvas.setFont(regular, 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawCentredString(
        A4[0] / 2,
        10 * mm,
        f"Writing Image Benchmark System 사용 설명서 | {doc.page}",
    )
    canvas.restoreState()


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Writing Image Benchmark System 사용 설명서",
        author="Cursor",
    )
    doc.build(_build_flowables(markdown), onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
