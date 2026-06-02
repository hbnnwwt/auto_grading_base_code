"""答题卡 HTML 渲染器。"""

from __future__ import annotations

from typing import List

from answer_sheet_generator.layout_engine import PAGE_SIZES, Page, paginate
from answer_sheet_generator.schema import AnswerSheetConfig


def _page_css(paper_size: str) -> str:
    """根据纸张尺寸返回 @page 的 size 值。"""
    size_map = {
        "A4": "210mm 297mm",
        "B5": "176mm 250mm",
    }
    return size_map.get(paper_size, "210mm 297mm")


def _render_exam_info() -> str:
    """渲染考生信息填写区域。"""
    return """<div class="exam-info">
  <div class="info-row"><span class="info-label">科目：</span><span class="info-blank"></span></div>
  <div class="info-row"><span class="info-label">日期：</span><span class="info-blank"></span></div>
  <div class="info-row"><span class="info-label">姓名：</span><span class="info-blank"></span></div>
</div>"""


def render_html(cfg: AnswerSheetConfig, pages: List[Page]) -> str:
    """将分页结果渲染为完整 HTML 文档。

    Parameters
    ----------
    cfg: AnswerSheetConfig
        答题卡配置。
    pages: List[Page]
        分页结果（由 paginate 生成）。

    Returns
    -------
    str
        完整 HTML 文档字符串，包含内联 CSS。
    """
    paper_size = cfg.meta.paper_size
    page_size_css = _page_css(paper_size)
    total_pages = len(pages)

    pages_html = ""
    for page_idx, page in enumerate(pages):
        # 渲染本页所有组件
        components_html = ""
        for comp, y_offset in page.components:
            components_html += comp.render(page.page_number, y_offset)

        # 每页标题：优先使用 page_config.title，其次使用 meta.title
        page_title = cfg.meta.title
        if page_idx < len(cfg.pages) and cfg.pages[page_idx].title:
            page_title = cfg.pages[page_idx].title

        page_mark = f"第 {page.page_number} 页 / 共 {total_pages} 页"

        pages_html += f"""<div class="page">
  <div class="print-hint">打印提示：请使用 A4 纸，在浏览器打印对话框中选择「另存为 PDF」或连接打印机直接打印。</div>
  <div class="page-title">{page_title}</div>
  {_render_exam_info()}
  {components_html}
  <div class="page-mark">{page_mark}</div>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cfg.meta.title}</title>
<style>
@page {{ size: {page_size_css}; margin: 10mm; }}

/* ---------- Base ---------- */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif; font-size: 10.5pt; line-height: 1.4; color: #222; }}

/* ---------- Screen preview ---------- */
@media screen {{
  body {{ background: #e5e5e5; padding: 20px; }}
  .page {{ background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.12); margin: 0 auto 20px; }}
  .print-hint {{ display: block; }}
}}

/* ---------- Print ---------- */
@media print {{
  body {{ background: #fff; padding: 0; }}
  .page {{ box-shadow: none; margin: 0; }}
  .print-hint {{ display: none; }}
}}

/* ---------- Page container ---------- */
.page {{
  position: relative;
  width: {PAGE_SIZES.get(paper_size, PAGE_SIZES["A4"])["usable_width"]}mm;
  min-height: {PAGE_SIZES.get(paper_size, PAGE_SIZES["A4"])["usable_height"]}mm;
  padding: 8mm;
  margin-bottom: 0;
  page-break-after: always;
}}
.page:last-child {{ page-break-after: auto; }}

/* ---------- Print hint ---------- */
.print-hint {{
  background: #fffbe6;
  border: 1px solid #ffe58f;
  border-radius: 4px;
  color: #ad8b00;
  font-size: 9pt;
  padding: 6px 10px;
  margin-bottom: 8mm;
  text-align: center;
}}

/* ---------- Title ---------- */
.page-title {{
  font-size: 18pt;
  font-weight: bold;
  text-align: center;
  margin-bottom: 6mm;
  letter-spacing: 2px;
}}

/* ---------- Exam info ---------- */
.exam-info {{
  display: flex;
  gap: 12mm;
  margin-bottom: 6mm;
  padding-bottom: 3mm;
  border-bottom: 1px solid #ccc;
}}
.info-row {{ display: flex; align-items: baseline; }}
.info-label {{ font-size: 10.5pt; color: #444; white-space: nowrap; }}
.info-blank {{
  display: inline-block;
  width: 35mm;
  border-bottom: 1px solid #333;
  margin-left: 2mm;
  height: 1em;
}}

/* ---------- Section boxes (all have outer border) ---------- */
.student-id-section,
.choice-section,
.judge-section,
.essay-section {{
  position: absolute;
  left: 8mm;
  right: 8mm;
  border: 1px solid #333;
  padding: 3mm;
}}
.sid-title {{
  font-size: 12pt;
  font-weight: bold;
  margin-bottom: 2mm;
}}
.sid-instruction {{
  font-size: 9pt;
  color: #666;
  margin-bottom: 2mm;
}}
.sid-write-row {{
  display: flex;
  align-items: center;
  gap: 2mm;
  margin-bottom: 3mm;
}}
.sid-write-box {{
  display: inline-block;
  width: 8mm;
  height: 8mm;
  border: 1px solid #333;
  text-align: center;
  line-height: 8mm;
}}
.sid-grid {{
  display: grid;
  gap: 1mm 2mm;
  margin-top: 2mm;
}}
.sid-digit-header {{
  text-align: center;
  font-size: 9pt;
  font-weight: bold;
  border-bottom: 1px solid #333;
  padding-bottom: 1mm;
}}
.sid-cell {{
  width: 6mm;
  height: 6mm;
  border: 1px solid #333;
  margin: 0 auto;
}}

/* ---------- Choice / Judge / Essay section ---------- */
.sec-title {{
  font-size: 11pt;
  font-weight: bold;
  margin-bottom: 3mm;
  padding-left: 2mm;
  border-left: 3px solid #333;
}}
.choice-grid, .judge-grid {{
  display: grid;
  gap: 3mm 4mm;
}}
.q-item {{
  display: flex;
  align-items: center;
  gap: 2mm;
  font-size: 10pt;
}}
.q-num {{ font-weight: bold; }}
.opt {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 5.5mm;
  height: 5.5mm;
  border: 1px solid #333;
  border-radius: 50%;
  font-size: 8.5pt;
}}

/* ---------- Essay section ---------- */
.essay-list {{ display: flex; flex-direction: column; gap: 3mm; }}
.essay-item {{ display: flex; gap: 2mm; }}
.essay-label {{
  font-weight: bold;
  font-size: 10pt;
  min-width: 6mm;
}}
.essay-lines {{ flex: 1; display: flex; flex-direction: column; gap: 2mm; }}
.essay-line {{
  height: 8mm;
  border-bottom: 1px solid #999;
}}

/* ---------- Page mark ---------- */
.page-mark {{
  position: absolute;
  bottom: 4mm;
  right: 8mm;
  font-size: 9pt;
  color: #666;
}}
</style>
</head>
<body>
{pages_html}</body>
</html>"""
    return html


def generate(cfg: AnswerSheetConfig) -> str:
    """一键生成答题卡 HTML。

    先调用 paginate(cfg) 进行分页，再调用 render_html 渲染。
    """
    pages = paginate(cfg)
    return render_html(cfg, pages)
