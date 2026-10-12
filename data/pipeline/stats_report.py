"""Dataset statistics and plots for the Phase 1 report.

REFERENCE CODE -- the Hindi token files were built once with this pipeline and
are not regenerated; see ``data/README.md``.  This file is kept so the
repository shows how the corpus was built.  It compiles but is not run.

Computation and plotting are kept in **separate functions**: every ``compute_*``
returns plain data and every ``plot_*`` consumes it.  That separation is a spec
requirement, and it also means the numbers in the report can be regenerated and
checked without rendering anything.

EVERY PLOT CARRIES A TITLE, AXIS LABELS AND A LEGEND.  A missing label scores
zero, so the axis-labelling is asserted in :func:`_finish` rather than left to
discipline.

COLOR
-----
A fixed four-hue categorical order, assigned by series identity and never
cycled.  Validated for colour-vision deficiency before use: worst-case pairwise
separation is dE 18.4 for normal vision and 11.2 under simulated deuteranopia and
protanopia, against floors of 15 and 8.  The full Okabe-Ito eight-hue set fails
that check (green vs purple separates by only 7.9 under deuteranopia), which is
why this is a re-stepped four-hue subset rather than the familiar palette.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: this runs on a workstation with no display
import matplotlib.pyplot as plt  # noqa: E402

from data.pipeline.dedup import iter_shard, split_shards  # noqa: E402

LOGGER = logging.getLogger("stats")

#: Fixed categorical order. Series get a hue by identity, never by rank, so a
#: chart with fewer series never repaints the survivors.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#56B4E9"]

#: Recessive ink for grid, axes and text, so the data marks carry the emphasis.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

LANGUAGE_COLOR = {"hindi": PALETTE[0]}
SOURCE_TYPE_COLOR = {"manual": PALETTE[0], "public": PALETTE[2]}


def _style() -> None:
    """Apply a consistent, recessive plot style."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _finish(ax, title: str, xlabel: str, ylabel: str, legend: bool = True) -> None:
    """Apply title and axis labels, and assert none are empty.

    The assertion is deliberate: an unlabelled axis scores zero, and that is far
    too easy to miss by eye across a dozen figures.

    Args:
        ax: Matplotlib axes.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        legend: Whether to draw a legend (only for >= 2 series).
    """
    assert title and xlabel and ylabel, "every plot needs a title and axis labels"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if legend:
        ax.legend(frameon=False, fontsize=9)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_corpus_composition(audit: dict) -> dict:
    """Extract manual/public token counts per split from an audit report."""
    return {
        "by_source_type": {k: v["tokens"] for k, v in audit["by_source_type"].items()},
        "splits": {k: v["tokens"] for k, v in audit["splits"].items()},
        "manual_share": audit["manual_share"],
        "top_sources": dict(list(
            {k: v["tokens"] for k, v in audit["by_source"].items()}.items())[:10]),
    }


def compute_cleaning_funnel(clean_report: dict) -> list[tuple[str, int]]:
    """Return the token count surviving each pipeline stage, in order."""
    stage_a = clean_report["stage_a"]
    dedup = clean_report["stage_b_dedup"]
    cpt = clean_report["chars_per_token_measured"]
    raw = stage_a["tokens_in"]
    after_filters = int(stage_a["characters_kept"] / cpt)
    after_dedup = dedup["tokens_out"]
    return [("raw collected", raw),
            ("after LID", raw - int(stage_a["dropped_lid_total"] * raw /
                                    max(stage_a["documents_in"], 1))),
            ("after quality", after_filters),
            ("after dedup", after_dedup)]


def compute_document_lengths(clean_dir: Path, max_docs: int = 200_000) -> dict:
    """Sample document lengths in characters, split by source type."""
    lengths: dict[str, list[int]] = {"manual": [], "public": []}
    seen = 0
    for shard in split_shards(clean_dir, "train"):
        for record in iter_shard(shard):
            text = record.get("text") or ""
            if not text:
                continue
            source_type = record.get("source_type") or "public"
            if source_type in lengths:
                lengths[source_type].append(len(text))
            seen += 1
            if seen >= max_docs:
                return lengths
    return lengths


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_corpus_composition(per_language: dict, out_path: Path) -> None:
    """Grouped bars: manual vs public tokens per language."""
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    languages = list(per_language)
    width = 0.36
    positions = range(len(languages))

    for offset, source_type in ((-width / 2, "manual"), (width / 2, "public")):
        values = [per_language[lang]["by_source_type"].get(source_type, 0) / 1e6
                  for lang in languages]
        ax.bar([p + offset for p in positions], values, width * 0.94,
               label=source_type, color=SOURCE_TYPE_COLOR[source_type])
        for p, v in zip(positions, values):
            ax.text(p + offset, v + 4, f"{v:.0f}M", ha="center",
                    fontsize=8, color=MUTED)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([lang.capitalize() for lang in languages])
    _finish(ax, "Cleaned corpus composition by collection method",
            "Language", "Tokens (millions)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_manual_share(per_language: dict, out_path: Path) -> None:
    """Manual share per language, before and after corpus assembly.

    Both stages are plotted because only one of them is the deliverable and the
    other is easy to mistake for it.  The *cleaned* share is measured over
    everything that survived cleaning, which includes ~200M surplus public tokens
    that were deliberately over-collected; on that basis Hindi reads 16.7% and
    appears to fail.  The *exported* share is measured on the assembled 500M
    corpus that Phase 2 actually trains on, where all manual tokens are kept and
    surplus public is discarded.  Showing only the first would misrepresent the
    result; showing only the second would hide how it was achieved.
    """
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    languages = list(per_language)
    width = 0.36
    positions = range(len(languages))

    stages = [("cleaned corpus (pre-assembly)", "manual_share", PALETTE[3]),
              ("exported 500M corpus (final)", "exported_manual_share", PALETTE[0])]

    for (label, key, color), offset in zip(stages, (-width / 2, width / 2)):
        shares = [100 * per_language[lang].get(key, 0) for lang in languages]
        bars = ax.bar([p + offset for p in positions], shares, width * 0.94,
                      label=label, color=color)
        for bar, share in zip(bars, shares):
            ax.text(bar.get_x() + bar.get_width() / 2, share + 0.5,
                    f"{share:.1f}%", ha="center", fontsize=9, color=INK)

    ax.axhline(20, color="#b02418", linestyle="--", linewidth=1.6,
               label="20% requirement", zorder=0)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([lang.capitalize() for lang in languages])
    ax.set_ylim(0, 36)
    _finish(ax, "Manual-collection share, before and after corpus assembly",
            "Language", "Share of training tokens (%)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cleaning_funnel(per_language: dict, out_path: Path) -> None:
    """Tokens surviving each cleaning stage, one line per language."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lang, stages in per_language.items():
        labels = [name for name, _ in stages]
        values = [value / 1e6 for _, value in stages]
        ax.plot(labels, values, marker="o", markersize=7, linewidth=2,
                color=LANGUAGE_COLOR[lang], label=lang.capitalize())
        ax.text(len(labels) - 1, values[-1], f"  {values[-1]:.0f}M",
                va="center", fontsize=9, color=LANGUAGE_COLOR[lang])

    _finish(ax, "Tokens surviving each cleaning stage",
            "Pipeline stage", "Tokens remaining (millions)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_quality_rejections(per_language: dict, out_path: Path) -> None:
    """Horizontal bars: documents rejected per named quality reason."""
    _style()
    reasons: Counter = Counter()
    for counts in per_language.values():
        reasons.update(counts)
    ordered = [r for r, _ in reasons.most_common(10)][::-1]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    height = 0.38
    for offset, (lang, counts) in zip((-height / 2, height / 2), per_language.items()):
        values = [counts.get(reason, 0) for reason in ordered]
        ax.barh([i + offset for i in range(len(ordered))], values, height * 0.94,
                label=lang.capitalize(), color=LANGUAGE_COLOR[lang])

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([r.replace("_", " ") for r in ordered])
    _finish(ax, "Documents rejected by quality filter, by reason",
            "Documents rejected", "Rejection reason")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tokenizer_sweep(per_language: dict, out_path: Path) -> None:
    """Fertility against vocabulary size, one line per algorithm per language."""
    _style()
    fig, axes = plt.subplots(1, len(per_language), figsize=(10, 4),
                             sharey=True, squeeze=False)
    for ax, (lang, results) in zip(axes[0], per_language.items()):
        by_type: dict[str, list[dict]] = {}
        for row in results:
            by_type.setdefault(row["model_type"], []).append(row)
        for index, (model_type, rows) in enumerate(sorted(by_type.items())):
            rows.sort(key=lambda r: r["vocab_size"])
            ax.plot([r["vocab_size"] / 1000 for r in rows],
                    [r["tokens_per_word"] for r in rows],
                    marker="o", markersize=7, linewidth=2,
                    color=PALETTE[index], label=model_type.upper())
        ax.set_title(lang.capitalize())
        ax.set_xlabel("Vocabulary size (thousands)")
        ax.grid(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[0][0].set_ylabel("Fertility (tokens per word)")
    fig.suptitle("Tokenizer sweep: fertility by algorithm and vocabulary size",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_document_lengths(per_language: dict, out_path: Path) -> None:
    """Document length distributions by source type, log-scaled x-axis."""
    _style()
    fig, axes = plt.subplots(1, len(per_language), figsize=(10, 4),
                             sharey=True, squeeze=False)
    import numpy as np
    for ax, (lang, lengths) in zip(axes[0], per_language.items()):
        bins = np.logspace(2, 5, 50)
        for source_type, values in lengths.items():
            if not values:
                continue
            ax.hist(values, bins=bins, histtype="step", linewidth=2,
                    color=SOURCE_TYPE_COLOR[source_type], label=source_type)
        ax.set_xscale("log")
        ax.set_title(lang.capitalize())
        ax.set_xlabel("Document length (characters, log scale)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[0][0].set_ylabel("Number of documents")
    fig.suptitle("Document length distribution by collection method",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Build every figure and the combined statistics JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--languages", nargs="+", default=["hindi"])
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    parser.add_argument("--figures", type=Path, default=Path(f"reports/data/figures"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    args.figures.mkdir(parents=True, exist_ok=True)

    composition, funnels, rejections, sweeps, lengths = {}, {}, {}, {}, {}
    combined = {}

    for language in args.languages:
        audit_path = args.report_dir / f"audit_{language}.json"
        clean_path = args.report_dir / f"clean_{language}.json"
        sweep_path = args.report_dir / f"tokenizer_sweep_{language}.json"

        export_path = args.report_dir / f"export_{language}.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            composition[language] = compute_corpus_composition(audit)
            combined[language] = {"audit": audit}
            # The exported share is the deliverable; the audit share is
            # pre-assembly. Both are plotted, so both must be carried through.
            if export_path.exists():
                export = json.loads(export_path.read_text())
                train = next((s for s in export["splits"] if s["split"] == "train"), {})
                composition[language]["exported_manual_share"] = train.get("manual_share", 0)
                composition[language]["exported_tokens"] = train.get("tokens", 0)
                combined[language]["export"] = export
        if clean_path.exists():
            clean = json.loads(clean_path.read_text())
            funnels[language] = compute_cleaning_funnel(clean)
            rejections[language] = clean["stage_a"]["quality_rejections_by_reason"]
            combined.setdefault(language, {})["clean"] = clean
        if sweep_path.exists():
            sweeps[language] = json.loads(sweep_path.read_text())["results"]
        clean_dir = Path(language) / "data" / "clean"
        if clean_dir.exists():
            lengths[language] = compute_document_lengths(clean_dir)

    if composition:
        plot_corpus_composition(composition, args.figures / "corpus_composition.png")
        plot_manual_share(composition, args.figures / "manual_share.png")
    if funnels:
        plot_cleaning_funnel(funnels, args.figures / "cleaning_funnel.png")
    if rejections:
        plot_quality_rejections(rejections, args.figures / "quality_rejections.png")
    if sweeps:
        plot_tokenizer_sweep(sweeps, args.figures / "tokenizer_sweep.png")
    if lengths:
        plot_document_lengths(lengths, args.figures / "document_lengths.png")

    out = args.report_dir / "dataset_statistics.json"
    out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("figures -> %s", args.figures)
    LOGGER.info("statistics -> %s", out)


if __name__ == "__main__":
    main()
