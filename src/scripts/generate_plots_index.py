#!/usr/bin/env python3
"""
Generate static HTML indexes for evaluation plot PDFs.

Scans a plots tree (config subdirs like ``plots/default/``, ``plots/hyperscale/``,
or a legacy flat layout with ``plots/classification/`` at the top level) and writes:

  - ``<plots-dir>/index.html`` — landing page with one card per configuration
  - ``<plots-dir>/<config>/index.html`` — categorized links for that configuration

Use ``--base-url`` when publishing to S3/CloudFront so PDF links are absolute.
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]

KNOWN_CATEGORIES = frozenset(
    {"classification", "regression", "steps_combined", "small_paper"}
)
SKIP_DIRS = frozenset({"tmp_results", "_site", "configs", "__pycache__"})

CATEGORY_ORDER = ("classification", "regression", "steps_combined", "small_paper")
CATEGORY_LABELS = {
    "classification": "Classification",
    "regression": "Energy regression",
    "steps_combined": "Training curves",
    "small_paper": "Paper figures",
}
SUBCATEGORY_ORDER = ("light", "q3", "w_bins", "pions", "steps")
SUBCATEGORY_LABELS = {
    "light": "Light appendix",
    "q3": "Q³ tagging",
    "w_bins": "W-bin tagging",
    "pions": "Pion tagging",
    "steps": "Per-task steps",
}

CONFIG_DESCRIPTIONS = {
    "default": "Full model lineup across all tasks.",
    "hyperscale": "HyperScale model family comparisons.",
    "V1Paper": "Figures for the v1 paper draft.",
    "bert_vs_ol": "BERT vs OmniLearned head-to-head.",
    "20260606_Comparison": "June 2026 comparison snapshot.",
    "dataset_distributions": "Dataset summary and kinematic distributions.",
}


@dataclass
class PlotFile:
    rel_path: Path
    display_name: str


@dataclass
class PlotSection:
    slug: str
    title: str
    plots: list[PlotFile] = field(default_factory=list)
    subsections: list[PlotSection] = field(default_factory=list)

    @property
    def plot_count(self) -> int:
        return len(self.plots) + sum(s.plot_count for s in self.subsections)


@dataclass
class PlotConfig:
    slug: str
    root: Path
    sections: list[PlotSection] = field(default_factory=list)

    @property
    def plot_count(self) -> int:
        return sum(s.plot_count for s in self.sections)


def prettify_filename(stem: str) -> str:
    """Turn ``eval_cc1pi_tagging_q3_1A`` into a readable label."""
    text = stem
    replacements = [
        (r"^eval_classification_light_", ""),
        (r"^eval_", ""),
        (r"_tagging_", " tagging — "),
        (r"cc1pi0", "CC1π⁰"),
        (r"cc1pi", "CC1π"),
        (r"ccnpi", "CCNπ"),
        (r"Npi_Ngt1", "Nπ (N>1)"),
        (r"Npi", "Nπ"),
        (r"_global_fpr", " (global FPR)"),
        (r"_pion_kinematics", " — pion kinematics"),
        (r"_q3_", " — Q³, "),
        (r"_W_", " — W, "),
        (r"_1A_1B", " playlists 1A & 1B"),
        (r"_1A", " playlist 1A"),
        (r"_1B", " playlist 1B"),
        (r"event_composition", "Event composition"),
        (r"confusion_matrices", "Confusion matrices"),
        (r"pi0_baseline_deltam", "π⁰ baseline Δm"),
        (r"log_flops_vs_val_loss", "Validation loss vs log₁₀ FLOPs"),
        (r"log_steps_vs_val_loss", "Validation loss vs training steps"),
        (r"q3_vs_iqr_rms", "Q³ vs IQR RMS"),
        (r"residuals_by_", "Residuals by "),
        (r"debug_E_pred_vs_true", "Predicted vs true available energy"),
        (r"regression_e_ratio_hist", "E ratio histogram"),
        (r"tpr_fixed_fpr", "TPR at fixed FPR"),
        (r"counts_sb", "Signal/background counts"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = text.replace("_", " ").strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text or stem


def collect_pdfs(directory: Path, prefix: Path) -> list[PlotFile]:
    plots: list[PlotFile] = []
    for path in sorted(directory.glob("*.pdf")):
        rel = path.relative_to(prefix)
        plots.append(PlotFile(rel_path=rel, display_name=prettify_filename(path.stem)))
    return plots


def build_category_section(category_dir: Path, slug: str, prefix: Path) -> PlotSection:
    section = PlotSection(slug=slug, title=CATEGORY_LABELS.get(slug, slug.replace("_", " ").title()))

    subdirs = [
        d for d in category_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    ]
    pdf_subdirs = [d for d in subdirs if any(d.glob("*.pdf"))]
    direct_pdfs = collect_pdfs(category_dir, prefix)

    if slug == "classification" and pdf_subdirs:
        for sub_slug in SUBCATEGORY_ORDER:
            sub_dir = category_dir / sub_slug
            if not sub_dir.is_dir():
                continue
            plots = collect_pdfs(sub_dir, prefix)
            if not plots:
                continue
            section.subsections.append(
                PlotSection(
                    slug=sub_slug,
                    title=SUBCATEGORY_LABELS.get(sub_slug, sub_slug.replace("_", " ").title()),
                    plots=plots,
                )
            )
        for sub_dir in sorted(pdf_subdirs, key=lambda p: p.name):
            if sub_dir.name in SUBCATEGORY_ORDER:
                continue
            plots = collect_pdfs(sub_dir, prefix)
            if plots:
                section.subsections.append(
                    PlotSection(
                        slug=sub_dir.name,
                        title=sub_dir.name.replace("_", " ").title(),
                        plots=plots,
                    )
                )
    elif direct_pdfs:
        section.plots = direct_pdfs
    else:
        for sub_dir in sorted(subdirs, key=lambda p: p.name):
            plots = collect_pdfs(sub_dir, prefix)
            if plots:
                section.subsections.append(
                    PlotSection(
                        slug=sub_dir.name,
                        title=sub_dir.name.replace("_", " ").title(),
                        plots=plots,
                    )
                )

    return section


def discover_misc_config(child: Path) -> PlotConfig | None:
    """Configs like ``dataset_distributions/`` with PDFs but no eval category layout."""
    pdfs = sorted(child.rglob("*.pdf"))
    if not pdfs:
        return None

    by_group: dict[str, list[PlotFile]] = {}
    for path in pdfs:
        rel = path.relative_to(child)
        group = rel.parts[0] if len(rel.parts) > 1 else "overview"
        by_group.setdefault(group, []).append(
            PlotFile(rel_path=rel, display_name=prettify_filename(path.stem))
        )

    sections = []
    for group in sorted(by_group):
        title = "Overview" if group == "overview" else group.replace("_", " ").title()
        sections.append(
            PlotSection(
                slug=group,
                title=title,
                plots=sorted(by_group[group], key=lambda p: p.rel_path.as_posix()),
            )
        )
    return PlotConfig(slug=child.name, root=child, sections=sections)


def discover_configs(plots_root: Path) -> list[PlotConfig]:
    configs: list[PlotConfig] = []

    for child in sorted(plots_root.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        if not any((child / cat).is_dir() for cat in KNOWN_CATEGORIES):
            continue
        sections = []
        for cat in CATEGORY_ORDER:
            cat_dir = child / cat
            if not cat_dir.is_dir():
                continue
            section = build_category_section(cat_dir, cat, child)
            if section.plot_count:
                sections.append(section)
        if sections:
            configs.append(PlotConfig(slug=child.name, root=child, sections=sections))

    known_slugs = {c.slug for c in configs}
    for child in sorted(plots_root.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS or child.name in known_slugs:
            continue
        misc = discover_misc_config(child)
        if misc is not None:
            configs.append(misc)

    if any((plots_root / cat).is_dir() for cat in KNOWN_CATEGORIES):
        sections = []
        for cat in CATEGORY_ORDER:
            cat_dir = plots_root / cat
            if not cat_dir.is_dir():
                continue
            section = build_category_section(cat_dir, cat, plots_root)
            if section.plot_count:
                sections.append(section)
        if sections and not any(c.slug == "default" for c in configs):
            configs.insert(0, PlotConfig(slug="default", root=plots_root, sections=sections))

    return configs


def url_join(base: str, *parts: str) -> str:
    path = "/".join(str(p).strip("/") for p in parts if p)
    if not base:
        return path
    return f"{base.rstrip('/')}/{path}"


def plot_href(base_url: str, config_slug: str, rel_path: Path | str) -> str:
    """Link to a PDF from a config ``index.html`` page."""
    rel = Path(rel_path).as_posix()
    if base_url:
        return url_join(base_url, config_slug, rel)
    # Page lives in <config>/index.html — omit config_slug for relative paths.
    return rel


def render_page(
    *,
    title: str,
    body: str,
    breadcrumbs: list[tuple[str, str | None]],
    generated_at: str,
) -> str:
    crumb_html = ""
    if breadcrumbs:
        items = []
        for i, (label, href) in enumerate(breadcrumbs):
            if href and i < len(breadcrumbs) - 1:
                items.append(f'<a href="{html.escape(href)}">{html.escape(label)}</a>')
            else:
                items.append(f"<span>{html.escape(label)}</span>")
        crumb_html = f'<nav class="breadcrumbs">{" › ".join(items)}</nav>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #0369a1;
      --accent-soft: #e0f2fe;
      --border: #e2e8f0;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }}
    header {{
      background: linear-gradient(135deg, #0c4a6e, #0369a1);
      color: white;
      padding: 2rem 1.25rem;
      margin-bottom: 2rem;
    }}
    header .wrap {{ padding: 0; }}
    header h1 {{ margin: 0 0 0.35rem; font-size: 1.75rem; font-weight: 650; }}
    header p {{ margin: 0; opacity: 0.9; }}
    .breadcrumbs {{
      font-size: 0.9rem;
      color: var(--muted);
      margin-bottom: 1.25rem;
    }}
    .breadcrumbs a {{ color: var(--accent); text-decoration: none; }}
    .breadcrumbs a:hover {{ text-decoration: underline; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 1rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.1rem 1.2rem;
      box-shadow: var(--shadow);
      transition: border-color 0.15s, transform 0.15s;
    }}
    .card:hover {{ border-color: #7dd3fc; transform: translateY(-1px); }}
    .card h2, .card h3 {{ margin: 0 0 0.35rem; font-size: 1.05rem; }}
    .card p {{ margin: 0 0 0.75rem; color: var(--muted); font-size: 0.92rem; }}
    .card a.stretch {{
      display: inline-block;
      color: var(--accent);
      font-weight: 600;
      text-decoration: none;
    }}
    .card a.stretch:hover {{ text-decoration: underline; }}
    .badge {{
      display: inline-block;
      background: var(--accent-soft);
      color: #075985;
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.15rem 0.5rem;
      border-radius: 999px;
      margin-bottom: 0.5rem;
    }}
    section.block {{ margin-bottom: 2rem; }}
    section.block > h2 {{
      font-size: 1.2rem;
      margin: 0 0 0.75rem;
      padding-bottom: 0.35rem;
      border-bottom: 2px solid var(--accent-soft);
    }}
    section.block h3 {{
      font-size: 1rem;
      margin: 1.25rem 0 0.5rem;
      color: #334155;
    }}
    ul.plot-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 0.45rem;
    }}
    ul.plot-list li {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.55rem 0.75rem;
    }}
    ul.plot-list a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 500;
    }}
    ul.plot-list a:hover {{ text-decoration: underline; }}
    ul.plot-list .path {{
      display: block;
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 0.15rem;
      word-break: break-all;
    }}
    footer {{
      margin-top: 2.5rem;
      font-size: 0.82rem;
      color: var(--muted);
      text-align: center;
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>MINERvA ML Evaluation Plots</h1>
      <p>PDF figures from <code>src.eval</code> — browse by plot configuration.</p>
    </div>
  </header>
  <div class="wrap">
    {crumb_html}
    {body}
    <footer>Generated {html.escape(generated_at)} UTC</footer>
  </div>
</body>
</html>
"""


def render_root_index(configs: list[PlotConfig], base_url: str, generated_at: str) -> str:
    cards = []
    for cfg in configs:
        desc = CONFIG_DESCRIPTIONS.get(cfg.slug, "Evaluation figures for this configuration.")
        href = url_join(base_url, cfg.slug, "index.html") if base_url else f"{cfg.slug}/index.html"
        cards.append(
            f"""<article class="card">
  <span class="badge">{cfg.plot_count} plots</span>
  <h2><a class="stretch" href="{html.escape(href)}">{html.escape(cfg.slug)}</a></h2>
  <p>{html.escape(desc)}</p>
</article>"""
        )
    body = f"""
    <section class="block">
      <h2>Plot configurations</h2>
      <div class="grid">{"".join(cards)}</div>
    </section>
    """
    return render_page(
        title="MINERvA ML Evaluation Plots",
        body=body,
        breadcrumbs=[("All configurations", None)],
        generated_at=generated_at,
    )


def render_section(section: PlotSection, config_slug: str, base_url: str) -> str:
    chunks: list[str] = []
    if section.plots:
        items = []
        for plot in section.plots:
            href = plot_href(base_url, config_slug, plot.rel_path)
            items.append(
                f"""<li>
  <a href="{html.escape(href)}">{html.escape(plot.display_name)}</a>
  <span class="path">{html.escape(plot.rel_path.as_posix())}</span>
</li>"""
            )
        chunks.append(f'<ul class="plot-list">{"".join(items)}</ul>')

    for sub in section.subsections:
        chunks.append(f"<h3>{html.escape(sub.title)}</h3>")
        chunks.append(render_section(sub, config_slug, base_url))
    return "".join(chunks)


def render_config_index(cfg: PlotConfig, base_url: str, generated_at: str) -> str:
    sections_html = []
    for section in cfg.sections:
        sections_html.append(
            f"""<section class="block" id="{html.escape(section.slug)}">
  <h2>{html.escape(section.title)}</h2>
  {render_section(section, cfg.slug, base_url)}
</section>"""
        )

    root_href = url_join(base_url, "index.html") if base_url else "../index.html"
    body = f"""
    <p style="color: var(--muted); margin-top: 0;">
      {cfg.plot_count} PDF plots in this configuration.
    </p>
    {"".join(sections_html)}
    """
    return render_page(
        title=f"{cfg.slug} — MINERvA ML Plots",
        body=body,
        breadcrumbs=[
            ("All configurations", root_href),
            (cfg.slug, None),
        ],
        generated_at=generated_at,
    )


def write_indexes(plots_root: Path, base_url: str) -> list[Path]:
    configs = discover_configs(plots_root)
    if not configs:
        raise SystemExit(f"No plot configurations found under {plots_root}")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    written: list[Path] = []

    root_index = plots_root / "index.html"
    root_index.write_text(render_root_index(configs, base_url, generated_at), encoding="utf-8")
    written.append(root_index)

    for cfg in configs:
        out = cfg.root / "index.html"
        out.write_text(render_config_index(cfg, base_url, generated_at), encoding="utf-8")
        written.append(out)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=_REPO_ROOT / "plots",
        help="Root directory containing plot PDFs (default: repo plots/)",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Public URL prefix for PDF links (e.g. CloudFront base). "
        "Omit for relative links suitable for local preview.",
    )
    args = parser.parse_args()

    plots_root = args.plots_dir.resolve()
    if not plots_root.is_dir():
        raise SystemExit(f"Plots directory not found: {plots_root}")

    written = write_indexes(plots_root, args.base_url.rstrip("/"))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
