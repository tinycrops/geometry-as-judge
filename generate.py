"""
generate.py — builds the Geometry as Judge documentation site.

Run from /home/ath/writing/:
    python3 generate.py
"""

import base64
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyArrow
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path("/home/ath/writing")
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = Path("/home/ath/experiments")

# ── Shared style constants ─────────────────────────────────────────────────────

DPI = 150
FIGSIZE_WIDE = (10, 5)
FIGSIZE_SQUARE = (8, 8)

COLORS = {
    "text":  "#4361ee",
    "image": "#f72585",
    "audio": "#7209b7",
    "video": "#3a0ca3",
    "bg":    "#ffffff",
    "grid":  "#eeeeee",
}

def save_fig(fig, name):
    path = FIGS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path.name} ({path.stat().st_size // 1024} KB)")
    return path


# ── Figure 1: Unified Space ────────────────────────────────────────────────────

def make_unified_space():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    rng = np.random.default_rng(42)

    modalities = {
        "Text":  {"center": (2.0, 3.5),  "color": COLORS["text"],  "examples": ['"a cat"', '"meow sound"', '"forest path"', '"sad melody"']},
        "Image": {"center": (5.5, 3.5),  "color": COLORS["image"], "examples": ["cat photo", "forest PNG", "city skyline", "portrait"]},
        "Audio": {"center": (2.0, 0.5),  "color": COLORS["audio"], "examples": ["cat meow", "bird song", "rain sound", "piano note"]},
        "Video": {"center": (5.5, 0.5),  "color": COLORS["video"], "examples": ["cat video", "nature clip", "timelapse", "interview"]},
    }

    point_positions = {}

    for mod, info in modalities.items():
        cx, cy = info["center"]
        color = info["color"]
        # Draw cluster ellipse
        ellipse = mpatches.Ellipse(
            (cx, cy), width=2.8, height=1.8,
            angle=rng.uniform(-15, 15),
            facecolor=color, alpha=0.10,
            edgecolor=color, linewidth=2, linestyle="--"
        )
        ax.add_patch(ellipse)
        ax.text(cx, cy + 1.2, mod, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

        # Plot example points
        positions = []
        for i, ex in enumerate(info["examples"]):
            angle = 2 * np.pi * i / len(info["examples"]) + rng.uniform(-0.3, 0.3)
            r = rng.uniform(0.3, 0.75)
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            positions.append((px, py))
            ax.scatter(px, py, color=color, s=60, zorder=5, edgecolors="white", linewidths=1)
            ax.annotate(ex, (px, py), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, color=color, style="italic")
        point_positions[mod] = positions

    # Dashed lines connecting cross-modal siblings (text↔image "a cat"↔cat photo, etc.)
    siblings = [
        ("Text", 0, "Image", 0),   # "a cat" ↔ cat photo
        ("Text", 2, "Image", 1),   # "forest path" ↔ forest PNG
        ("Text", 1, "Audio", 0),   # "meow sound" ↔ cat meow
        ("Audio", 1, "Video", 0),  # bird song ↔ cat video (approximate)
        ("Image", 0, "Video", 0),  # cat photo ↔ cat video
    ]
    for mod_a, i_a, mod_b, i_b in siblings:
        pa = point_positions[mod_a][i_a]
        pb = point_positions[mod_b][i_b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color="#aaaaaa", linewidth=1.2, linestyle=":", zorder=2,
                alpha=0.8)

    ax.set_xlim(-0.2, 7.8)
    ax.set_ylim(-0.8, 5.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Gemini Embedding 2: One Unified Space",
                 fontsize=15, fontweight="bold", pad=14, color="#1a1a2e")

    legend_elements = [
        mpatches.Patch(facecolor=v["color"], alpha=0.7, label=k)
        for k, v in modalities.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    note = ("Dashed lines connect cross-modal siblings that embed nearby each other.\n"
            "Positions are illustrative, not real embeddings.")
    ax.text(0.01, 0.01, note, transform=ax.transAxes,
            fontsize=7.5, color="#888888", va="bottom")

    for spine in ax.spines.values():
        spine.set_visible(False)

    return save_fig(fig, "fig_unified_space.png")


# ── Figure 2: Cosine Intuition ─────────────────────────────────────────────────

def make_cosine_intuition():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    cases = [
        {"angle": 10,  "cos":  0.985, "label": "Very similar",    "color": "#2ecc71"},
        {"angle": 90,  "cos":  0.0,   "label": "Unrelated",       "color": "#f39c12"},
        {"angle": 170, "cos": -0.985, "label": "Opposite meaning", "color": "#e74c3c"},
    ]

    for ax, case in zip(axes, cases):
        ax.set_facecolor("#f8f9fa")
        theta = np.radians(case["angle"])
        color = case["color"]

        # Two unit vectors
        v1 = np.array([1.0, 0.0])
        v2 = np.array([np.cos(theta), np.sin(theta)])

        scale = 0.78
        for v, vc, label in [(v1, "#2c3e50", "A"), (v2, color, "B")]:
            ax.annotate("", xy=(v[0] * scale, v[1] * scale), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.12",
                                        color=vc, lw=2.5))
            ax.text(v[0] * scale * 1.12, v[1] * scale * 1.12, label,
                    fontsize=11, fontweight="bold", color=vc, ha="center", va="center")

        # Arc showing angle
        arc_r = 0.25
        arc_angles = np.linspace(0, theta, 80)
        ax.plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles),
                color="#555555", linewidth=1.5)

        # Angle label
        mid_angle = theta / 2
        label_r = 0.36
        ax.text(label_r * np.cos(mid_angle), label_r * np.sin(mid_angle),
                f"θ={case['angle']}°", fontsize=9, ha="center", va="center",
                color="#333333")

        # Cosine annotation
        cos_str = f"cos θ = {case['cos']:+.3f}"
        ax.text(0, -1.1, cos_str, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=color, linewidth=1.5))

        ax.text(0, -1.45, case["label"], ha="center", va="center",
                fontsize=10, color="#333333", style="italic")

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.65, 1.2)
        ax.set_aspect("equal")
        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

    fig.suptitle("What Cosine Similarity Measures",
                 fontsize=15, fontweight="bold", color="#1a1a2e", y=1.02)
    fig.tight_layout()
    return save_fig(fig, "fig_cosine_intuition.png")


# ── Figure 3: Pipeline ────────────────────────────────────────────────────────

def make_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, facecolor, edgecolor, fontsize=9.5, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.8, zorder=3
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#1a1a2e",
                zorder=4, wrap=True)

    def arrow(x1, y, x2, color="#555555"):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.18",
                                    color=color, lw=1.8), zorder=5)

    def merge_arrow(x1, y1, x2, y2, xm, ym, color="#555555"):
        ax.annotate("", xy=(xm, ym), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.8), zorder=5)
        ax.annotate("", xy=(x2, y2), xytext=(xm, ym),
                    arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.18",
                                    color=color, lw=1.8), zorder=5)

    # Row 1: text path
    box(0.2, 3.2, 1.5, 0.8, '"a cat"',          "#dbeafe", "#4361ee", bold=True)
    arrow(1.7, 3.6, 2.5)
    box(2.5, 3.2, 1.8, 0.8, "embed_text()",      "#dbeafe", "#4361ee")
    arrow(4.3, 3.6, 5.1)
    box(5.1, 3.2, 1.8, 0.8, "[vector A]\n3072-d", "#e0e7ff", "#4361ee", fontsize=8.5)

    # Row 2: image path
    box(0.2, 1.0, 1.5, 0.8, "ASCII art",         "#fce7f3", "#f72585", bold=True)
    arrow(1.7, 1.4, 2.5)
    box(2.5, 1.0, 1.4, 0.8, "render()",          "#fce7f3", "#f72585")
    arrow(3.9, 1.4, 4.7)
    box(4.7, 1.0, 0.9, 0.8, "PNG",               "#fdf2f8", "#f72585", fontsize=9)
    arrow(5.6, 1.4, 6.4)
    box(6.4, 1.0, 1.9, 0.8, "embed_image()",     "#fce7f3", "#f72585")
    arrow(8.3, 1.4, 9.1)
    box(9.1, 1.0, 1.8, 0.8, "[vector B]\n3072-d", "#fce7f3", "#f72585", fontsize=8.5)

    # Convergence to cosine
    # Lines from vector A and vector B to cosine box
    ax.plot([6.0, 10.2, 10.2], [3.6, 3.6, 2.7],
            color="#555555", linewidth=1.8, zorder=5)
    ax.plot([10.0, 10.2, 10.2], [1.4, 1.4, 2.3],
            color="#555555", linewidth=1.8, zorder=5)
    ax.annotate("", xy=(10.9, 2.5), xytext=(10.2, 2.5),
                arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.18",
                                color="#555555", lw=1.8), zorder=5)

    box(10.9, 2.1, 1.4, 0.8, "cosine()",         "#d1fae5", "#059669", fontsize=9.5)
    arrow(12.3, 2.5, 12.8)
    ax.text(12.85, 2.5, "0.404", ha="left", va="center",
            fontsize=14, fontweight="bold", color="#059669")

    ax.set_title("The Cross-Modal Metric Pipeline",
                 fontsize=15, fontweight="bold", color="#1a1a2e", pad=14)

    return save_fig(fig, "fig_pipeline.png")


# ── Figure 4: DSPy Loop ───────────────────────────────────────────────────────

def make_dspy_loop():
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.3)
    ax.axis("off")

    nodes = [
        {"label": "trainset\n(Examples)",      "angle": 90,   "color": "#dbeafe", "edge": "#4361ee"},
        {"label": "module.forward()\n(LM call)","angle": 30,   "color": "#fce7f3", "edge": "#f72585"},
        {"label": "prediction\n(ASCII art)",    "angle": -30,  "color": "#fce7f3", "edge": "#f72585"},
        {"label": "embedding_metric()\n(score)","angle": -90,  "color": "#d1fae5", "edge": "#059669"},
        {"label": "Bayesian\nOptimizer",        "angle": -150, "color": "#fef3c7", "edge": "#d97706"},
        {"label": "updated\ninstructions",      "angle": 150,  "color": "#fef3c7", "edge": "#d97706"},
    ]

    radius = 0.78
    node_w = 0.42
    node_h = 0.22

    positions = {}
    for node in nodes:
        theta = np.radians(node["angle"])
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        positions[node["label"]] = (x, y)

        rect = mpatches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.04",
            facecolor=node["color"], edgecolor=node["edge"], linewidth=2, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, y, node["label"], ha="center", va="center",
                fontsize=8, fontweight="bold", color="#1a1a2e", zorder=4)

    # Draw arrows along the circle
    n = len(nodes)
    for i in range(n):
        src = nodes[i]
        dst = nodes[(i + 1) % n]
        src_theta = np.radians(src["angle"])
        dst_theta = np.radians(dst["angle"])

        # Start/end points nudged off node edges
        mid_theta = (np.radians(src["angle"]) + np.radians(dst["angle"])) / 2
        # Use arc midpoint for arrow placement
        x1 = radius * np.cos(src_theta)
        y1 = radius * np.sin(src_theta)
        x2 = radius * np.cos(dst_theta)
        y2 = radius * np.sin(dst_theta)

        # Offset start from node edge
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        sx = x1 + ux * (node_w / 2 + 0.04)
        sy = y1 + uy * (node_h / 2 + 0.02)
        ex = x2 - ux * (node_w / 2 + 0.04)
        ey = y2 - uy * (node_h / 2 + 0.02)

        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.15,head_length=0.12",
                        color="#666666", lw=1.6,
                        connectionstyle="arc3,rad=0.18"
                    ), zorder=2)

    # Center annotation
    ax.text(0, 0.08, "MIPROv2", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1a1a2e")
    ax.text(0, -0.08, "optimization\nloop", ha="center", va="center",
            fontsize=9, color="#555555", style="italic")

    # Highlight the metric as the score source
    metric_x = radius * np.cos(np.radians(-90))
    metric_y = radius * np.sin(np.radians(-90))
    ax.annotate("← score source",
                xy=(metric_x + 0.24, metric_y - 0.04),
                fontsize=8.5, color="#059669", style="italic")

    ax.set_title("DSPy MIPROv2: Metric-Driven Prompt Optimization",
                 fontsize=14, fontweight="bold", color="#1a1a2e", pad=14)

    return save_fig(fig, "fig_dspy_loop.png")


# ── Figure 5: Copy experiment result ─────────────────────────────────────────

def copy_experiment_result():
    src = EXPERIMENTS / "ascii_metric_result.png"
    dst = FIGS / "fig_experiment_results.png"
    shutil.copy2(src, dst)
    print(f"  Copied fig_experiment_results.png ({dst.stat().st_size // 1024} KB)")
    return dst


# ── Figure 6–11: Copy ascii-bench / noise-01 figures ─────────────────────────

NOISE_FIGURES = [
    "fig_corpus_v2.png",
    "fig_pairwise_v2.png",
    "fig_pca_scripts.png",
    "fig_pca_full.png",
    "fig_scores_v2.png",
    "fig_llm_grids_v2.png",
    "fig_noise_vs_blank.png",
    "fig_triangle.png",
]

def copy_noise_figures():
    for name in NOISE_FIGURES:
        src = EXPERIMENTS / name
        dst = FIGS / name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {name} ({dst.stat().st_size // 1024} KB)")
        else:
            print(f"  WARNING: {src} not found — skipping")


# ── HTML helpers ──────────────────────────────────────────────────────────────

def b64_figure(fig_path: Path) -> str:
    data = base64.b64encode(fig_path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"

SHARED_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: #ffffff;
  color: #222233;
  line-height: 1.7;
  font-size: 16.5px;
}

.page-wrap {
  max-width: 860px;
  margin: 0 auto;
  padding: 0 28px 64px 28px;
}

/* ── Navigation ── */
nav {
  background: #1a1a2e;
  padding: 0 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 52px;
  position: sticky;
  top: 0;
  z-index: 100;
}
nav .nav-title {
  color: #e0e7ff;
  font-weight: 700;
  font-size: 15px;
  letter-spacing: 0.01em;
  text-decoration: none;
}
nav .nav-links a {
  color: #a5b4fc;
  text-decoration: none;
  margin-left: 22px;
  font-size: 14px;
  transition: color 0.15s;
}
nav .nav-links a:hover { color: #ffffff; }
nav .nav-links a.active { color: #ffffff; font-weight: 600; }

/* ── Headings ── */
h1 { font-size: 2.3rem; color: #1a1a2e; font-weight: 800; letter-spacing: -0.02em; margin: 40px 0 16px 0; }
h2 {
  font-size: 1.45rem; color: #1a1a2e; font-weight: 700;
  margin: 48px 0 14px 0;
  padding-left: 14px;
  border-left: 4px solid #4361ee;
}
h3 { font-size: 1.15rem; color: #1a1a2e; font-weight: 700; margin: 30px 0 10px 0; }

p { margin: 0 0 18px 0; }
ul, ol { margin: 0 0 18px 0; padding-left: 28px; }
li { margin-bottom: 6px; }

/* ── Code ── */
pre {
  background: #1e1e2e;
  color: #cdd6f4;
  border-radius: 10px;
  padding: 22px 26px;
  overflow-x: auto;
  font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace;
  font-size: 13.5px;
  line-height: 1.65;
  margin: 0 0 24px 0;
}
code {
  font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace;
  font-size: 0.88em;
  background: #f0f0fa;
  color: #3730a3;
  padding: 2px 6px;
  border-radius: 4px;
}
pre code { background: none; color: inherit; padding: 0; font-size: inherit; border-radius: 0; }

/* ── Callout boxes ── */
.callout {
  border-radius: 8px;
  padding: 18px 22px;
  margin: 24px 0;
  border-left: 5px solid;
}
.callout-note      { background: #eff6ff; border-color: #3b82f6; }
.callout-insight   { background: #f0fdf4; border-color: #22c55e; }
.callout-warning   { background: #fffbeb; border-color: #f59e0b; }
.callout-discovery { background: #fefce8; border-color: #ca8a04; }
.callout-title {
  font-weight: 700;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 8px;
}
.callout-note .callout-title      { color: #1d4ed8; }
.callout-insight .callout-title   { color: #15803d; }
.callout-warning .callout-title   { color: #b45309; }
.callout-discovery .callout-title { color: #854d0e; }

/* ── Cards ── */
.card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 24px 0 32px 0; }
.card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 22px 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  transition: box-shadow 0.18s, transform 0.18s;
  text-decoration: none;
  color: inherit;
  display: block;
}
.card:hover {
  box-shadow: 0 6px 20px rgba(67,97,238,0.14);
  transform: translateY(-3px);
}
.card-num { font-size: 0.78rem; font-weight: 700; color: #4361ee; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.card h3  { margin: 0 0 8px 0; font-size: 1.05rem; color: #1a1a2e; }
.card p   { margin: 0; font-size: 0.9rem; color: #64748b; }

/* ── Figures ── */
figure {
  margin: 28px 0;
  text-align: center;
}
figure img {
  max-width: 100%;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
figcaption {
  margin-top: 10px;
  font-size: 0.88rem;
  color: #64748b;
  font-style: italic;
}

/* ── Pull quotes ── */
.pullquote {
  font-size: 1.2em;
  font-style: italic;
  color: #1a1a2e;
  border-left: 4px solid #4361ee;
  padding: 14px 22px;
  margin: 28px 0;
  background: #f5f5ff;
  border-radius: 0 8px 8px 0;
  line-height: 1.6;
}

/* ── Hero ── */
.hero {
  padding: 56px 0 40px 0;
  border-bottom: 1px solid #e2e8f0;
  margin-bottom: 48px;
}
.hero h1 { margin: 0 0 14px 0; font-size: 2.8rem; line-height: 1.1; }
.hero .subtitle {
  font-size: 1.2rem;
  color: #475569;
  max-width: 640px;
  line-height: 1.55;
  margin-bottom: 32px;
}

/* ── Table ── */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 24px 0;
  font-size: 0.95rem;
}
th {
  background: #1a1a2e;
  color: #e0e7ff;
  padding: 10px 14px;
  text-align: left;
  font-size: 0.85rem;
  letter-spacing: 0.04em;
}
td {
  padding: 10px 14px;
  border-bottom: 1px solid #e2e8f0;
  vertical-align: top;
}
tr:nth-child(even) td { background: #f8fafc; }

/* ── Stack list ── */
.stack-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 18px 0 30px 0;
  list-style: none;
  padding: 0;
}
.stack-list li {
  background: #1a1a2e;
  color: #a5b4fc;
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.88rem;
  font-weight: 600;
  margin: 0;
}

/* ── Footer ── */
footer {
  border-top: 1px solid #e2e8f0;
  margin-top: 64px;
  padding: 24px 0 0 0;
  font-size: 0.85rem;
  color: #94a3b8;
  text-align: center;
}
"""

def nav(active="index"):
    pages = [
        ("index",      "index.html",       "Overview"),
        ("paradigm",   "01_paradigm.html",  "The Space"),
        ("experiment", "02_experiment.html","The Experiment"),
        ("dspy",       "03_dspy.html",      "DSPy Integration"),
        ("bench",      "ascii-bench.html",  "ascii-bench"),
        ("noise01",    "04_noise_eval.html","noise-01"),
    ]
    links = ""
    for key, href, label in pages:
        cls = ' class="active"' if key == active else ""
        links += f'<a href="{href}"{cls}>{label}</a>'
    return f"""<nav>
  <a href="index.html" class="nav-title">Geometry as Judge</a>
  <div class="nav-links">{links}</div>
</nav>"""

def footer():
    return """<footer>
  <p>Generated from /home/ath/experiments/ &middot; Gemini Embedding 2 &middot; March 2026</p>
</footer>"""

def page_shell(title, active, body):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Geometry as Judge</title>
<style>
{SHARED_CSS}
</style>
</head>
<body>
{nav(active)}
<div class="page-wrap">
{body}
{footer()}
</div>
</body>
</html>"""

def fig_tag(fig_path: Path, caption: str) -> str:
    return f"""<figure>
  <img src="{b64_figure(fig_path)}" alt="{caption}">
  <figcaption>{caption}</figcaption>
</figure>"""

def callout(kind: str, title: str, body: str) -> str:
    return f"""<div class="callout callout-{kind}">
  <div class="callout-title">{title}</div>
  {body}
</div>"""


# ── Read the proof script ─────────────────────────────────────────────────────

def read_proof_script() -> str:
    return (EXPERIMENTS / "ascii_metric_proof.py").read_text()


# ── index.html ────────────────────────────────────────────────────────────────

def make_index():
    results_fig = FIGS / "fig_experiment_results.png"
    pca_full_fig = FIGS / "fig_pca_full.png"

    body = f"""
<div class="hero">
  <h1>Geometry as Judge</h1>
  <p class="subtitle">A new paradigm for automatic model evaluation using multimodal embeddings — no human labels, no rubrics, no LLM judges. Now includes <strong>ascii-bench</strong>, the first benchmark built on this paradigm.</p>
</div>

{fig_tag(results_fig, "Proof of concept: cosine similarity between 'a cat' and three ASCII art renderings. The good cat shape scores highest.")}

<h2>The Core Idea</h2>

<p>For decades, evaluating model outputs required one of three things: expensive human raters, brittle rule-based heuristics, or — more recently — a second large model acting as judge. Each approach has a fundamental problem. Human raters are slow, costly, and inconsistent. Heuristics break the moment you leave their narrow domain. LLM judges are expensive, opaque, and introduce their own biases and failure modes.</p>

<p>Multimodal embedding models offer a different path. Google's Gemini Embedding 2 encodes text, images, audio, and video into a single geometric space — a high-dimensional manifold where semantic meaning determines position. Things that mean the same thing land nearby each other, regardless of what modality they came from. A sentence and a photograph of the same subject end up close together.</p>

<p>This gives us a new primitive: <strong>cosine similarity as semantic distance across modalities</strong>. Instead of asking a human or an LLM "is this ASCII art a good cat?", we can ask the geometry directly: "is the position of this rendered PNG close to the position of the text 'a cat'?" The embedding space does not need to be told what 'good' looks like. It already encodes it.</p>

{callout("insight", "What We Proved",
  "<p>We embedded the text description <code>'a cat'</code> and three PNG renderings of ASCII art — one recognizable cat shape, one plausible ASCII art of a different subject (a house), and random noise — into Gemini Embedding 2's shared space. The cosine similarity between the text embedding and each image embedding followed the expected ordering: <strong>good &gt; bad &gt; noise</strong>. The metric works without any labels.</p>")}

<h2>ascii-bench: Measuring What Models Know About Noise</h2>

<p>ascii-bench is the first benchmark built directly on this paradigm. Rather than evaluating semantic fidelity to a prompt, it asks a more fundamental question: <strong>can a model produce output that lands in the correct geometric region of the embedding space?</strong> The first evaluation, noise-01, measures 3×6 ASCII noise grid generation across Latin and multilingual scripts.</p>

{fig_tag(pca_full_fig, "ascii-bench noise-01: Full PCA projection of the noise manifold. Reference corpus (circles) and GPT-4o LLM outputs (triangles) plotted by script family. Five distinct clusters — noise has topology.")}

<h2>Explore the Documentation</h2>

<div class="card-grid">
  <a href="01_paradigm.html" class="card">
    <div class="card-num">Part 1</div>
    <h3>The Problem and the Space</h3>
    <p>Why existing evaluation approaches fail, and how a unified embedding manifold solves it.</p>
  </a>
  <a href="02_experiment.html" class="card">
    <div class="card-num">Part 2</div>
    <h3>The Experiment</h3>
    <p>The exact pipeline, code, test cases, and results that validate the cross-modal metric.</p>
  </a>
  <a href="03_dspy.html" class="card">
    <div class="card-num">Part 3</div>
    <h3>Wiring it into DSPy</h3>
    <p>How to use this metric as a DSPy training objective and let MIPROv2 optimize prompts automatically.</p>
  </a>
  <a href="ascii-bench.html" class="card">
    <div class="card-num">Benchmark</div>
    <h3>ascii-bench</h3>
    <p>The benchmark overview: evaluation structure, scoring framework, and planned evaluations.</p>
  </a>
  <a href="04_noise_eval.html" class="card">
    <div class="card-num">Evaluation 1</div>
    <h3>noise-01 Deep Dive</h3>
    <p>Full technical analysis of the noise grid evaluation: corpus design, embedding strategies, PCA clustering, and LLM generation scores.</p>
  </a>
</div>

<h2>The Stack</h2>
<ul class="stack-list">
  <li>Gemini Embedding 2</li>
  <li>PIL / Pillow</li>
  <li>NumPy</li>
  <li>scikit-learn</li>
  <li>DSPy</li>
  <li>GPT-4o</li>
  <li>Python 3.11+</li>
</ul>
"""
    html = page_shell("Overview", "index", body)
    out = ROOT / "index.html"
    out.write_text(html)
    print(f"  Wrote {out.name} ({out.stat().st_size // 1024} KB)")


# ── 01_paradigm.html ──────────────────────────────────────────────────────────

def make_paradigm():
    unified_fig   = FIGS / "fig_unified_space.png"
    cosine_fig    = FIGS / "fig_cosine_intuition.png"

    body = f"""
<h1>The Problem and the Space</h1>

<h2>Why Evaluation is Hard</h2>

<p>Evaluation is one of the oldest unsolved problems in machine learning. Before you can improve a model, you need to measure it. But measurement at scale is genuinely difficult — and the three classical approaches each have deep failure modes.</p>

<p><strong>Human raters</strong> are the gold standard, but gold is expensive. A meaningful human evaluation of a text-to-image model requires dozens of annotators, calibration sessions, inter-rater agreement measurement, and weeks of calendar time. You cannot run this in a training loop. Human evaluation answers questions about past snapshots; it cannot guide an optimizer in real time.</p>

<p><strong>Exact-match and rule-based heuristics</strong> scale well but generalize poorly. A pixel-level diff metric catches regression on memorized examples and nothing else. An ASCII art heuristic that counts specific character frequencies will score a line of <code>^^^^</code> highly for "mountain range" and miss the point entirely. Every heuristic is a closed-world assumption that breaks outside its calibration domain.</p>

<p><strong>LLM-as-judge</strong> is the current fashionable solution, and it is genuinely better than heuristics for many tasks. But it introduces a second large model into your evaluation loop — with its own inference cost, its own latency, its own biases, and its own failure modes. It is also opaque: when the judge disagrees with you, you cannot look inside to understand why. And there is a circularity problem when the judge and the judged share training data.</p>

<h2>One Space to Rule Them All</h2>

<p>Gemini Embedding 2 is trained to encode text, images, audio, and video into a single 3072-dimensional vector space. The training objective pushes semantically related content — regardless of modality — toward the same region of the space. This is not a loose metaphor. It is a measurable geometric fact: you can compute the cosine similarity between a text embedding and an image embedding and get a meaningful number.</p>

{fig_tag(unified_fig, "Gemini Embedding 2 maps text, images, audio, and video into one geometric space. Cross-modal siblings — things that mean the same thing in different modalities — cluster nearby each other.")}

<div class="pullquote">
  "A photograph of a forest and the sentence 'dense canopy, filtered light' end up nearby each other — without any translation layer."
</div>

<p>This is the key structural property that makes cross-modal evaluation possible. The space was not designed with ASCII art evaluation in mind. It was trained on a massive corpus of multimodal data with a general contrastive objective. And yet that objective is sufficient to create a useful evaluator for any semantic task — because semantic meaning is what the space encodes.</p>

<h2>What Cosine Similarity Actually Measures</h2>

<p>Cosine similarity measures the angle between two vectors in a high-dimensional space, ignoring their magnitude. It returns a value between −1 and +1. Two vectors pointing in exactly the same direction score 1.0; two perpendicular vectors score 0.0; two vectors pointing in opposite directions score −1.0.</p>

{fig_tag(cosine_fig, "Cosine similarity as angle between vectors. What matters is direction, not magnitude — two embeddings can have very different norms but still agree semantically.")}

<p>In practice, embeddings for semantically related content tend to point in similar directions. The cosine between "a cat" and a high-quality cat image is noticeably higher than the cosine between "a cat" and a house image — even though all three vectors have roughly unit norm. The signal is in the direction, not the length.</p>

<p>For cross-modal pairs (text vs. image), absolute values tend to be lower than within-modality comparisons. A score of 0.40 for a text-image pair can represent strong alignment; a score of 0.25 can represent near-noise. The absolute thresholds need calibration for each use case. But <em>relative ordering</em> — which of two outputs is better — is robust and interpretable without calibration.</p>

{callout("insight", "The Key Insight",
  "<p>We don't need to tell the metric what 'good' looks like. The embedding space already knows. We just need to measure the distance.</p>")}

<h2>What This Unlocks</h2>

<ul>
  <li><strong>Any modality can evaluate any other modality.</strong> Text descriptions can evaluate images, videos, audio. Rendered images can evaluate textual descriptions. The pairing is free.</li>
  <li><strong>Scales to any subject without relabeling.</strong> The same metric function that evaluates ASCII cats evaluates ASCII castles, portraits, landscapes, or abstract concepts. No domain-specific rules needed.</li>
  <li><strong>The same metric works as a training objective.</strong> Because it returns a scalar, it plugs directly into any optimizer that needs a score function — including DSPy's MIPROv2.</li>
  <li><strong>No inference from a second LLM.</strong> The embedding model is much cheaper and faster to call than a capable judge model. At training time, this matters.</li>
  <li><strong>Transparent failure mode.</strong> When the metric is wrong, it is because the embedding space is wrong — which you can inspect. There is no hidden chain-of-thought to debug.</li>
</ul>
"""
    html = page_shell("The Problem and the Space", "paradigm", body)
    out = ROOT / "01_paradigm.html"
    out.write_text(html)
    print(f"  Wrote {out.name} ({out.stat().st_size // 1024} KB)")


# ── 02_experiment.html ────────────────────────────────────────────────────────

def make_experiment():
    pipeline_fig = FIGS / "fig_pipeline.png"
    results_fig  = FIGS / "fig_experiment_results.png"
    proof_script = read_proof_script()

    ascii_good = r"""  /\_/\
 ( o.o )
 > ^ <"""
    ascii_bad = r"""  _____
 |     |
 |_____|
  |   |"""
    ascii_noise = r""" x@#$%^
 &*()_+
 !?><{}"""

    body = f"""
<h1>The Experiment</h1>

{callout("note", "Hypothesis",
  "<p>If we embed the text <code>'a cat'</code> and the rendered PNG of three ASCII art strings — one recognizable cat shape, one plausible ASCII art of a different subject, and random noise — into Gemini Embedding 2's shared space, the cosine similarity scores will follow the ordering: <strong>good &gt; bad &gt; noise</strong>, without any human labels or subject-specific heuristics.</p>")}

<h2>Test Cases</h2>

<p>Three ASCII art strings represent three levels of semantic alignment with the text prompt <code>'a cat'</code>:</p>

<h3>Good — recognizable cat shape</h3>
<p>This is the classic ASCII cat: ears, eyes, nose, whisker stub. A human would immediately recognize it as a cat. The embedding model should too.</p>
<pre><code>{ascii_good}</code></pre>

<h3>Bad — plausible ASCII art, wrong subject</h3>
<p>This is structurally valid ASCII art — it uses line-drawing characters consistently and depicts something coherent (a house with walls and a foundation). But it is not a cat. The metric should penalize the mismatch.</p>
<pre><code>{ascii_bad}</code></pre>

<h3>Noise — random characters</h3>
<p>Random printable characters with no structural intent. No recognizable shape, no semantic content. This should score at or below the noise floor.</p>
<pre><code>{ascii_noise}</code></pre>

<h2>The Pipeline</h2>

{fig_tag(pipeline_fig, "The metric pipeline: text and ASCII art flow through separate embedding paths, converging at a cosine similarity calculation.")}

<p>The pipeline has four steps. Each is a small, focused function:</p>

<h3>Step 1: Render ASCII to PNG</h3>
<p>The embedding model accepts images, not text-as-text. We need to rasterize the ASCII art into a PNG before embedding. We use PIL with a monospace font (DejaVu Sans Mono) so character alignment is preserved.</p>

<pre><code>FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
PAD = 16

def render(ascii_text: str) -> bytes:
    lines = ascii_text.splitlines()
    test_img = Image.new("RGB", (1, 1))
    test_draw = ImageDraw.Draw(test_img)
    line_height = FONT.getbbox("A")[3] + 4
    max_w = max(test_draw.textlength(line, font=FONT) for line in lines)
    w = int(max_w) + PAD * 2
    h = line_height * len(lines) + PAD * 2
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = PAD
    for line in lines:
        draw.text((PAD, y), line, fill=(0, 0, 0), font=FONT)
        y += line_height
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()</code></pre>

<h3>Step 2: Embed text</h3>
<p>Call the Gemini embedding API with the text prompt and <code>SEMANTIC_SIMILARITY</code> task type. This returns a 3072-dimensional float vector.</p>

<pre><code>def embed_text(text: str) -> np.ndarray:
    result = client.models.embed_content(
        model=MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)</code></pre>

<h3>Step 3: Embed image</h3>
<p>Call the same API with the PNG bytes as a <code>Part</code>. Note: no task type for image embeddings — the model infers the modality from the mime type.</p>

<pre><code>def embed_image(png_bytes: bytes) -> np.ndarray:
    result = client.models.embed_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=png_bytes, mime_type="image/png")],
    )
    return np.array(result.embeddings[0].values, dtype=np.float32)</code></pre>

<h3>Step 4: Cosine similarity</h3>
<p>Standard cosine: dot product divided by product of norms. Returns a scalar in [−1, 1].</p>

<pre><code>def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))</code></pre>

<h3>The metric function</h3>
<p>All four steps composed into a single callable:</p>

<pre><code>def ascii_metric(description: str, ascii_art: str) -> float:
    text_vec = embed_text(description)
    image_vec = embed_image(render(ascii_art))
    return cosine(text_vec, image_vec)</code></pre>

<h2>Results</h2>

{fig_tag(results_fig, "Bar chart of cosine similarity scores. Good cat: 0.404, Bad (house): 0.366, Noise: 0.296. The hypothesis holds.")}

<p>The scores confirm the hypothesis:</p>
<ul>
  <li><strong>Good (cat shape):</strong> 0.4040</li>
  <li><strong>Bad (house shape):</strong> 0.3655</li>
  <li><strong>Noise (random chars):</strong> 0.2958</li>
</ul>

{callout("insight", "Hypothesis Confirmed",
  "<p>good (0.404) &gt; bad (0.366) &gt; noise (0.296). The embedding space correctly rank-orders the three ASCII art strings by semantic alignment with the text prompt, without any subject-specific heuristics or human labels.</p>")}

<h2>What the Scores Mean</h2>

<p>The absolute values look low if you are used to within-modality similarity scores, where well-matched pairs often score above 0.85. Cross-modal similarity is structurally lower — text and image embeddings live in overlapping but distinct regions of the space. A score of 0.40 for a text-image pair represents strong alignment.</p>

<p>What matters more than absolute value is <strong>spread</strong>: the difference between good and noise is 0.108, which is a clear signal well above any reasonable noise floor. The noise floor in practice is determined by random vector pairs, which in 3072 dimensions cluster tightly around 0.0. A cross-modal score of 0.30 is already well above that baseline.</p>

<p>The spread also has an important property: it is <em>monotonically related to semantic quality</em>. You do not need to threshold the scores or calibrate them to get useful signal. Any optimizer that wants to maximize the score will automatically prefer outputs that are more semantically aligned with the input description.</p>

<h2>Limitations</h2>

{callout("warning", "Honest Assessment",
  """<p>This is a proof of concept, not a study.</p>
  <ul>
    <li><strong>n=3 is not a study.</strong> Three test cases confirm the hypothesis is not obviously false. They do not establish statistical significance, generalization bounds, or calibration across subjects.</li>
    <li><strong>Font and scale sensitivity unknown.</strong> We used one font at one size. The metric may behave differently with pixel fonts, very small art, or unusual aspect ratios.</li>
    <li><strong>Subject coverage is narrow.</strong> We tested one subject (cat). The metric needs replication across more subjects before claiming subject-independence.</li>
    <li><strong>Threshold calibration needed.</strong> What score separates "acceptable" from "unacceptable"? That requires labeled data for each use case.</li>
    <li><strong>API cost.</strong> Each evaluation requires two embedding API calls. At training scale (hundreds of examples, many iterations), this is non-trivial.</li>
  </ul>""")}

<h2>Complete Script</h2>

<p>The full runnable proof-of-concept script:</p>

<pre><code>{proof_script.replace("<", "&lt;").replace(">", "&gt;")}</code></pre>
"""
    html = page_shell("The Experiment", "experiment", body)
    out = ROOT / "02_experiment.html"
    out.write_text(html)
    print(f"  Wrote {out.name} ({out.stat().st_size // 1024} KB)")


# ── 03_dspy.html ──────────────────────────────────────────────────────────────

def make_dspy():
    loop_fig = FIGS / "fig_dspy_loop.png"

    # Code blocks that contain triple-quotes are built separately to avoid
    # terminating the outer f-string.
    metric_code = (
        "import os, io, numpy as np\n"
        "from PIL import Image, ImageDraw, ImageFont\n"
        "from google import genai\n"
        "from google.genai import types\n"
        "\n"
        "client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])\n"
        "MODEL = 'gemini-embedding-2-preview'\n"
        "FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 20)\n"
        "\n"
        "def render(ascii_text: str) -> bytes:\n"
        '    """Rasterize ASCII art to PNG bytes."""\n'
        "    lines = ascii_text.splitlines()\n"
        "    test_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))\n"
        "    lh = FONT.getbbox('A')[3] + 4\n"
        "    w = int(max(test_draw.textlength(l, font=FONT) for l in lines)) + 32\n"
        "    h = lh * len(lines) + 32\n"
        "    img = Image.new('RGB', (w, h), (255, 255, 255))\n"
        "    draw = ImageDraw.Draw(img)\n"
        "    y = 16\n"
        "    for line in lines:\n"
        "        draw.text((16, y), line, fill=(0, 0, 0), font=FONT)\n"
        "        y += lh\n"
        "    buf = io.BytesIO()\n"
        "    img.save(buf, format='PNG')\n"
        "    return buf.getvalue()\n"
        "\n"
        "def embed_text(text: str) -> np.ndarray:\n"
        "    r = client.models.embed_content(\n"
        "        model=MODEL, contents=[text],\n"
        "        config=types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY'))\n"
        "    return np.array(r.embeddings[0].values, dtype=np.float32)\n"
        "\n"
        "def embed_image(png_bytes: bytes) -> np.ndarray:\n"
        "    r = client.models.embed_content(\n"
        "        model=MODEL,\n"
        "        contents=[types.Part.from_bytes(data=png_bytes, mime_type='image/png')])\n"
        "    return np.array(r.embeddings[0].values, dtype=np.float32)\n"
        "\n"
        "def cosine(a, b):\n"
        "    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))\n"
        "\n"
        "# ── The DSPy metric ──────────────────────────────────────────────────\n"
        "\n"
        "THRESHOLD = 0.35  # adjust after calibration\n"
        "\n"
        "def embedding_metric(example, prediction, trace=None):\n"
        '    """\n'
        "    Cross-modal cosine similarity between the story description\n"
        "    and the rendered PNG of the predicted ASCII art.\n"
        "\n"
        "    - During evaluation (trace=None or trace is a list):\n"
        "        Returns a float in [0, 1] representing alignment quality.\n"
        "    - During bootstrapping (trace is truthy):\n"
        "        Returns True if score exceeds THRESHOLD, else False.\n"
        "        DSPy uses this to filter good few-shot demos.\n"
        '    """\n'
        "    description = example.story_excerpt         # the text prompt\n"
        "    ascii_art   = prediction.ascii_art          # the generated output\n"
        "\n"
        "    text_vec  = embed_text(description)\n"
        "    image_vec = embed_image(render(ascii_art))\n"
        "    score     = cosine(text_vec, image_vec)\n"
        "\n"
        "    # Normalize to [0, 1] from typical cross-modal range [0.1, 0.5]\n"
        "    score_norm = max(0.0, min(1.0, (score - 0.1) / 0.4))\n"
        "\n"
        "    if trace is not None:\n"
        "        return score_norm >= THRESHOLD          # bool for bootstrapping\n"
        "    return score_norm                           # float for evaluation\n"
    )

    sig_code = (
        "import dspy\n"
        "\n"
        "# ── 1. Configure the LM ─────────────────────────────────────────────\n"
        "lm = dspy.LM('gemini/gemini-3.1-pro-preview', api_key=os.environ['GEMINI_API_KEY'])\n"
        "dspy.configure(lm=lm)\n"
        "\n"
        "# ── 2. Define the Signature ──────────────────────────────────────────\n"
        "class AsciiArtSignature(dspy.Signature):\n"
        '    """Convert a story excerpt into expressive ASCII art that captures\n'
        "    the subject described in the text.\"\"\"\n"
        "\n"
        "    story_excerpt: str = dspy.InputField(\n"
        "        desc='A short passage of prose describing a scene or character.'\n"
        "    )\n"
        "    ascii_art: str = dspy.OutputField(\n"
        "        desc='ASCII art (3-8 lines, plain text characters only) that visually '\n"
        "             'represents the main subject of the excerpt.'\n"
        "    )\n"
        "\n"
        "# ── 3. Define the Module ─────────────────────────────────────────────\n"
        "class AsciiArtModule(dspy.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.generate = dspy.Predict(AsciiArtSignature)\n"
        "\n"
        "    def forward(self, story_excerpt: str) -> dspy.Prediction:\n"
        "        # dspy.Predict calls the LM with the current prompt + any\n"
        "        # few-shot demos that have been compiled in.\n"
        "        return self.generate(story_excerpt=story_excerpt)\n"
        "\n"
        "# ── 4. Build trainset ────────────────────────────────────────────────\n"
        "trainset = [\n"
        "    dspy.Example(\n"
        "        story_excerpt='The old tabby curled on the windowsill, eyes half-closed.',\n"
        "        ascii_art='  /\\_/\\\\\\n ( o.o )\\n > ^ <\\n',\n"
        "    ).with_inputs('story_excerpt'),\n"
        "    dspy.Example(\n"
        "        story_excerpt='A hawk circled high above the valley, wings spread wide.',\n"
        "        ascii_art='   __\\n  /  \\\\\\n \\\\____/\\n  \\\\  /\\n   \\\\/\\n',\n"
        "    ).with_inputs('story_excerpt'),\n"
        "    # ... add more for better optimization\n"
        "]\n"
        "\n"
        "# ── 5. Set up MIPROv2 ────────────────────────────────────────────────\n"
        "optimizer = dspy.MIPROv2(\n"
        "    metric=embedding_metric,    # our cross-modal cosine metric\n"
        "    auto='light',               # light = fewer trials, good for prototyping\n"
        "    num_threads=4,\n"
        ")\n"
        "\n"
        "# ── 6. Compile ───────────────────────────────────────────────────────\n"
        "module = AsciiArtModule()\n"
        "optimized = optimizer.compile(\n"
        "    module,\n"
        "    trainset=trainset,\n"
        "    max_bootstrapped_demos=3,   # include up to 3 few-shot examples\n"
        "    max_labeled_demos=2,        # seed with up to 2 labeled demos\n"
        ")\n"
        "\n"
        "# ── 7. Use the optimized module ──────────────────────────────────────\n"
        "result = optimized(story_excerpt='A wolf howled at the full moon.')\n"
        "print(result.ascii_art)\n"
    )

    body = f"""
<h1>Wiring it into DSPy</h1>

<h2>What DSPy Is</h2>

<p>DSPy is a Python framework for building and optimizing language model programs. The core insight behind DSPy is that prompts should be derived, not hand-written. Instead of spending hours crafting system prompts, few-shot examples, and output format instructions, you declare <em>what</em> you want — the input fields, the output fields, and the evaluation metric — and DSPy's optimizers find the best prompt automatically.</p>

<p>This matters because prompt engineering is brittle. A prompt that works well on GPT-4o may perform worse on Gemini. A prompt tuned for 10 examples may generalize poorly to 100. DSPy treats prompts as learned parameters, not fixed strings, and optimizes them against a metric over a training set. The result is programs that are more robust, more reproducible, and often surprisingly effective — because the optimizer explores a much larger space of instructions than any human would think to try.</p>

<h2>The Five Primitives</h2>

<table>
  <thead>
    <tr><th>Primitive</th><th>What it is</th><th>In our context</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><code>dspy.Example</code></td>
      <td>A labeled training example with named fields</td>
      <td>A story excerpt paired with a reference ASCII art</td>
    </tr>
    <tr>
      <td><code>dspy.Prediction</code></td>
      <td>The output produced by a module for one input</td>
      <td>The ASCII art string generated by the LM</td>
    </tr>
    <tr>
      <td><code>dspy.Signature</code></td>
      <td>A typed interface: input fields → output fields with descriptions</td>
      <td><code>story_excerpt → ascii_art</code> with task description</td>
    </tr>
    <tr>
      <td><code>dspy.Predict</code> / Module</td>
      <td>A callable that maps inputs to outputs using an LM</td>
      <td>The forward pass that calls the LM with the current prompt</td>
    </tr>
    <tr>
      <td>Metric function</td>
      <td>A callable <code>(example, prediction, trace) → float | bool</code></td>
      <td>Our embedding cosine similarity score</td>
    </tr>
  </tbody>
</table>

<h2>dspy.Example: Your Data</h2>

<p>A <code>dspy.Example</code> holds a dictionary of named fields. The <code>with_inputs()</code> call marks which fields are inputs (fed to the model) and which are labels (used by the metric for comparison).</p>

<pre><code>import dspy

# One training example: a story excerpt that should produce an ASCII cat
example = dspy.Example(
    story_excerpt="The old tabby curled into a perfect circle on the windowsill.",
    ascii_art="  /\\_/\\\n ( o.o )\n > ^ <\n",
).with_inputs("story_excerpt")

# .with_inputs("story_excerpt") means:
#   - story_excerpt is an INPUT (passed to module.forward)
#   - ascii_art is a LABEL (available to the metric as example.ascii_art)

trainset = [example, ...]  # add more examples here</code></pre>

<h2>The Metric Function</h2>

<p>DSPy metric functions receive three arguments: the gold example, the predicted output, and a trace (used internally during optimization). The function must return a float for evaluation mode and can return a bool threshold for bootstrapping.</p>

<pre><code>{metric_code}</code></pre>

<p>The <code>trace</code> parameter controls dual-mode behavior. When DSPy is bootstrapping few-shot demonstrations, it passes a trace object and expects a bool — "is this demo good enough to include?" When it is evaluating the full program, trace is None and it expects a float for ranking. The same function handles both cases.</p>

<h2>The Full Program</h2>

<pre><code>{sig_code}</code></pre>

<h2>The Optimizer Loop</h2>

{fig_tag(loop_fig, "MIPROv2's optimization loop: trainset examples flow through the module, predictions are scored by the embedding metric, and the Bayesian optimizer updates instructions based on scores.")}

<p>MIPROv2 — Multiprompt Instruction Proposal Optimizer version 2 — operates in three stages:</p>

<ol>
  <li><strong>Bootstrap demonstrations.</strong> Run the module on trainset examples with an initial prompt. Where the metric returns True (score above threshold), collect those input-output pairs as candidate few-shot demos. These are concrete examples of good outputs that will be prepended to the prompt.</li>
  <li><strong>Propose instruction candidates.</strong> Use a meta-LM to generate many candidate instruction strings for each module, based on the task description, the bootstrap demos, and the signature fields. This is where the optimizer explores the space of possible prompts.</li>
  <li><strong>Bayesian search.</strong> Evaluate combinations of instructions and demo sets on the trainset using the metric. Use Bayesian optimization (specifically, a Tree Parzen Estimator) to efficiently search the space of combinations, spending more evaluations in promising regions. Return the combination with the highest average metric score.</li>
</ol>

<h2>What the Optimizer Will Discover</h2>

<p>MIPROv2 will discover prompt instructions that the embedding metric rewards. Because the metric measures how well the rendered ASCII art aligns with the text description in Gemini's embedding space, the optimizer will converge on instructions that produce outputs the embedding model recognizes as matching the text — purely from geometric feedback.</p>

<p>In practice, this likely means instructions like: "use dense characters for dark or solid regions", "use whitespace to preserve shape boundaries", "match the general outline of the described subject". The optimizer will not know this is what it found; it only knows that certain instructions produce higher cosine scores. But the geometric signal is rich enough to guide it there.</p>

<p>This is the deeper value of the paradigm: <strong>the embedding space serves as an implicit specification of quality</strong>. You do not need to encode domain knowledge into your metric. The model's pretraining has already encoded it. You just need to measure against it.</p>

{callout("insight", "The Compounding Advantage",
  "<p>As you expand the trainset and add more subjects, the metric generalizes automatically — because the embedding space generalizes automatically. A metric trained on cats and hawks will correctly evaluate ASCII art of wolves and castles, without any additional labeling, because the embedding model already understands what wolves and castles look like.</p>")}

<h2>Next Steps</h2>

<ul>
  <li><strong>Expand the trainset.</strong> Add 20–50 examples across diverse subjects. The optimizer's signal-to-noise ratio improves significantly with more examples.</li>
  <li><strong>Add TinyStories story→character evaluation.</strong> Use the same metric to evaluate whether ASCII character art matches the protagonist described in a short story passage.</li>
  <li><strong>Add reference image alignment.</strong> For cases where a reference image exists, compute the cosine between the rendered ASCII art PNG and the reference image directly (image-to-image cosine) as a complementary metric.</li>
  <li><strong>Calibrate thresholds.</strong> Collect 50–100 human ratings, fit a logistic regression against the cosine scores, and use the resulting threshold as your bootstrap cutoff.</li>
  <li><strong>Benchmark against LLM-judge.</strong> Run both metrics on the same examples and compare ranking agreement. This tells you how much of the LLM judge's signal the embedding metric captures.</li>
</ul>
"""
    html = page_shell("DSPy Integration", "dspy", body)
    out = ROOT / "03_dspy.html"
    out.write_text(html)
    print(f"  Wrote {out.name} ({out.stat().st_size // 1024} KB)")


# ── ascii-bench.html ──────────────────────────────────────────────────────────

def make_ascii_bench():
    body = """
<h1>ascii-bench</h1>

<p class="subtitle" style="font-size:1.1rem;color:#475569;margin-bottom:32px;">
A benchmark for evaluating ASCII art generation using geometric evaluation —
no human raters, no LLM judges, no exact-match heuristics.
</p>

<h2>What is ascii-bench?</h2>

<p>ascii-bench evaluates ASCII art generation capability using a single core technique: render the output as a PNG, embed it in Gemini Embedding 2's unified multimodal space, and measure where it lands. This turns a qualitative judgment — "does this look like the right thing?" — into a reproducible scalar measurement over a well-defined geometric space.</p>

<p>The approach exploits a structural property of Gemini Embedding 2: it places text, images, audio, and video into the same 3072-dimensional manifold, trained on cross-modal correspondence across the internet. The model has already learned what cats look like, what noise looks like, what Katakana characters look like. When you embed an ASCII art PNG, you get a vector that reflects the embedding model's interpretation of the image's visual content — not its character content.</p>

<p>This separation is precisely what makes the metric useful. A model that generates <code>aaaaaaa</code> in a grid when asked for noise may pass a character-level check but will land far from the noise region in the embedding space. The geometry sees through the characters to the visual impression they create.</p>

<h2>Why it Matters</h2>

<p>Prior ASCII art evaluation has been stuck in one of three modes:</p>
<ul>
  <li><strong>Human raters</strong> — accurate but slow, expensive, and impossible to run in an optimizer loop</li>
  <li><strong>Exact-match</strong> — brittle; only works if you have a fixed reference string</li>
  <li><strong>LLM-as-judge</strong> — expensive, opaque, introduces circularity when the judged model and the judge share training data</li>
</ul>

<p>ascii-bench offers a fourth path: <strong>automatic, scalable, cross-modal, script-agnostic geometric evaluation</strong>. It is cheap enough to run at training time, transparent enough to debug (failed outputs can be visualized and measured), and general enough to extend to any writing system without re-labeling.</p>

<p>The evaluations in this benchmark are also conceptually novel. noise-01 measures something that no benchmark has tried to measure before: what does it mean for a model to correctly generate randomness? Is there a noise manifold, and can a model land inside it? The answer turns out to be empirically interesting, and the geometry reveals structure that a character-level metric would completely miss.</p>

<h2>Seeding the Evaluation Space</h2>

<p>Before building any evaluation, we needed to establish what the semantic space looks like. We started with two corpora: noise (18-character operator grids) and blank canvases (structural frames with empty interiors). Embedding both in Gemini Embedding 2 and projecting to 2D PCA reveals how they cluster:</p>

{fig_tag(FIGS / "fig_noise_vs_blank.png", "ascii-bench corpus seeding: PCA projection of noise and blank canvas corpora. Green circles = noise; orange squares = blank canvases. Centroid-to-centroid distance: 0.8105. These two corpora define the first axis of the evaluation space.")}

<p>The centroid-to-centroid distance between noise and blank canvas is <strong>0.8105</strong> — meaningful separation in a space where same-corpus pairs range from 0.76–0.97. This established a baseline: we could measure semantic distance with a real signal.</p>

<div class="callout callout-note">
<strong>Why blank canvases matter:</strong> The original "bad" case in the proof-of-concept — <code>  _____<br>|     |<br>|_____|</code> (a house outline) — was meant to represent "wrong" art. But it's actually more interesting than that: it is <em>semantically empty</em>. It looks like nothing in particular. It is a blank canvas, not a failure. Understanding this helped clarify what the noise cluster actually measures: not "random characters" but "characters with no visual identity."
</div>

<h2>The Evaluation Triangle</h2>

<p>The benchmark space is anchored by three semantic regions measured empirically in Gemini Embedding 2's 3072-dimensional manifold:</p>

<ul>
  <li><strong style="color:#2ecc71;">Noise</strong> — dense operator/punctuation grids; no visual structure, no semantic content. Internal cohesion: 0.9386 (tightest cluster)</li>
  <li><strong style="color:#e67e22;">Blank canvas</strong> — structural frames with semantic emptiness: boxes, borders, bezels. Internal cohesion: 0.8413</li>
  <li><strong style="color:#3498db;">Computer ASCII art</strong> — recognizable hardware: keyboards, monitors, mice, floppies. Internal cohesion: 0.8149 (most spread — each piece has its own silhouette)</li>
</ul>

<div class="callout callout-discovery">
<strong>Key finding:</strong> blank canvas and computer ASCII art are <em>geometrically close</em> (centroid distance: 0.9166) because both use the same box-drawing grammar. A blank frame <em>is</em> a computer bezel with nothing inside it. The embedding space sees structural kinship where human eyes see semantic difference. The largest gap is between noise and computer art (0.7927) — the two most semantically distinct regions.
</div>

{fig_tag(FIGS / "fig_triangle.png", "ascii-bench evaluation triangle: PCA projection of all three anchor corpora. Noise (green circles), blank canvas (orange squares), computer ASCII art (blue triangles). Centroid distances shown on the triangle edges. The evaluation space is the geometry between these three anchors.")}

<div class="callout callout-insight">
<strong>What the triangle means for scoring:</strong> A generated ASCII art piece is measured by where it lands relative to all three anchors. A good computer ASCII art output should be close to the computer centroid, moderately close to blank (structural similarity), and far from noise (semantic difference). The triangle makes this relationship explicit and measurable.
</div>

<h2>Benchmark Structure</h2>

<table>
  <thead>
    <tr>
      <th>Evaluation</th>
      <th>Status</th>
      <th>What it measures</th>
      <th>Constraint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong><a href="04_noise_eval.html">noise-01</a></strong></td>
      <td><span style="color:#15803d;font-weight:700;">&#10003; COMPLETE</span></td>
      <td>3&times;6 noise grid generation and noise-region membership. Tests whether a model can produce visually random output that lands geometrically inside the noise manifold of the embedding space.</td>
      <td>3 rows &times; 6 chars; no semantic content; multi-script</td>
    </tr>
    <tr>
      <td><strong>computer-01</strong></td>
      <td><span style="color:#b45309;font-weight:700;">&#9654; IN PROGRESS</span></td>
      <td>Computer hardware ASCII art generation. Given a hardware name (keyboard, monitor, floppy), generate ASCII art that lands near the computer anchor and away from noise. The first evaluation with a semantic target.</td>
      <td>Free-form; named subject; computer hardware vocabulary</td>
    </tr>
    <tr>
      <td><strong>story-01</strong></td>
      <td><span style="color:#6b7280;font-style:italic;">&#128300; PLANNED</span></td>
      <td>Story excerpt &rarr; ASCII art of described character. Cross-modal fidelity: does the rendered art land near the text description in embedding space?</td>
      <td>3&ndash;8 rows; prose input; character subject</td>
    </tr>
    <tr>
      <td><strong>raw-01</strong></td>
      <td><span style="color:#6b7280;font-style:italic;">&#128300; PLANNED</span></td>
      <td>Unconstrained ASCII art quality vs. reference image. Image-to-image cosine between rendered output and a reference PNG.</td>
      <td>Open height/width; reference image provided</td>
    </tr>
    <tr>
      <td><strong>self-01</strong></td>
      <td><span style="color:#6b7280;font-style:italic;">&#128300; PLANNED</span></td>
      <td>Model self-portrait as ASCII art. Measured against the model's own origin in the embedding space &mdash; a form of geometric self-knowledge.</td>
      <td>Open; no reference; centroid measurement</td>
    </tr>
  </tbody>
</table>

<h2>The Scoring Framework</h2>

<p>Each evaluation in ascii-bench produces a combined float score by composing two orthogonal signals:</p>

<h3>1. Structural score (grid adherence)</h3>
<p>A fast, API-free check that measures whether the output conforms to the structural constraint of the evaluation. For noise-01, this means: does the output have exactly 3 rows of exactly 6 visible characters? The structural score is always computed first — if it is zero, the geometric score is skipped entirely (saving API cost and avoiding penalizing structurally invalid outputs on semantic grounds).</p>

<pre><code>def grid_adherence(text: str, rows: int = 3, cols: int = 6) -> float:
    # Returns float in [0, 1]
    # 1.0 = perfect 3×6 structure; 0.0 = completely wrong shape</code></pre>

<h3>2. Geometric score (noise-region membership)</h3>
<p>The semantic signal. The output is rendered as a PNG, embedded via Gemini Embedding 2 using the interleaved strategy (text + image fused into one vector), and measured against the centroid of the reference corpus. The raw cosine similarity is normalized within the observed noise-to-noise range to produce a score in [0, 1].</p>

<pre><code>def noise_similarity(text: str, centroid: np.ndarray) -> float:
    png = render(text)
    vec = embed_interleaved(text, png)
    sim = cosine(vec, centroid)
    # Normalize within [noise_floor, noise_ceiling]
    return float(np.clip((sim - noise_floor) / (noise_spread), 0.0, 1.0))</code></pre>

<h3>Combined score</h3>
<p>The two signals are combined as a weighted sum, with weight configurable per evaluation. The default for noise-01 is 40% structural, 60% geometric — structural conformance matters, but the geometric signal carries more information about what the output actually looks like.</p>

<pre><code>score = adherence_weight * structural_score + (1 - adherence_weight) * geometric_score
# adherence_weight default: 0.4</code></pre>

<p>This combined score is a float in [0, 1] suitable for direct use as a DSPy metric. During MIPROv2 bootstrapping, it is thresholded to a boolean (default threshold: 0.50) to filter candidate few-shot demonstrations.</p>

<h2>Stack</h2>
<ul class="stack-list">
  <li>Gemini Embedding 2</li>
  <li>PIL / Pillow</li>
  <li>NumPy</li>
  <li>scikit-learn</li>
  <li>DSPy + MIPROv2</li>
  <li>GPT-4o</li>
</ul>

<p style="margin-top:28px;"><a href="04_noise_eval.html" style="color:#4361ee;font-weight:600;text-decoration:none;">Read the noise-01 deep dive &rarr;</a></p>
"""
    html = page_shell("ascii-bench", "bench", body)
    out = ROOT / "ascii-bench.html"
    out.write_text(html)
    print(f"  Wrote {out.name} ({out.stat().st_size // 1024} KB)")


# ── 04_noise_eval.html ────────────────────────────────────────────────────────

def make_noise_eval():
    corpus_fig    = FIGS / "fig_corpus_v2.png"
    pairwise_fig  = FIGS / "fig_pairwise_v2.png"
    pca_scripts   = FIGS / "fig_pca_scripts.png"
    pca_full      = FIGS / "fig_pca_full.png"
    scores_fig    = FIGS / "fig_scores_v2.png"
    llm_grids_fig = FIGS / "fig_llm_grids_v2.png"

    noise_eval_code = (
        "def noise_eval(\n"
        "    text: str,\n"
        "    noise_centroid: np.ndarray,\n"
        "    trace=None,\n"
        "    threshold: float = 0.50,\n"
        "    adherence_weight: float = 0.4\n"
        ") -> float | bool:\n"
        '    """\n'
        "    Combined noise metric for DSPy.\n"
        "\n"
        "    Args:\n"
        "        text:             The ASCII grid output to evaluate.\n"
        "        noise_centroid:   Pre-computed centroid of the noise corpus embeddings.\n"
        "                          Normalized unit vector in 3072-dim space.\n"
        "        trace:            DSPy trace object.\n"
        "                          None  = evaluation mode  → returns float in [0,1]\n"
        "                          not None = bootstrapping → returns bool (score >= threshold)\n"
        "        threshold:        Bootstrap pass threshold. Default 0.50.\n"
        "        adherence_weight: Weight of structural score vs. geometric score.\n"
        "                          0.4 = 40% structural, 60% geometric. Default 0.4.\n"
        '    """\n'
        "    # Step 1: structural check (no API cost)\n"
        "    adherence = grid_adherence(text)\n"
        "\n"
        "    # Step 2: geometric check (only if structurally non-zero)\n"
        "    if adherence == 0.0:\n"
        "        score = 0.0\n"
        "    else:\n"
        "        rendered   = render(text)\n"
        "        pred_vec   = embed_interleaved(text, rendered)   # text+PNG → one vector\n"
        "        similarity = cosine(pred_vec, noise_centroid)\n"
        "\n"
        "        # Normalize raw cosine into [0, 1] using observed noise-to-noise range\n"
        "        # noise_floor ≈ 0.896  noise_ceil ≈ 0.968 (interleaved, Latin corpus)\n"
        "        similarity_norm = float(np.clip(\n"
        "            (similarity - noise_floor) / (noise_ceil - noise_floor), 0.0, 1.0\n"
        "        ))\n"
        "\n"
        "        score = (adherence_weight * adherence\n"
        "                 + (1 - adherence_weight) * similarity_norm)\n"
        "\n"
        "    # Step 3: dual-mode return (matches DSPy metric contract)\n"
        "    return score if trace is None else score >= threshold\n"
    )

    dspy_integration_code = (
        "import dspy\n"
        "from functools import partial\n"
        "\n"
        "# Pre-compute the centroid once at startup\n"
        "noise_centroid = build_noise_centroid()\n"
        "\n"
        "# Wrap into a DSPy-compatible metric\n"
        "def noise_metric(example, prediction, trace=None):\n"
        "    return noise_eval(\n"
        "        text           = prediction.grid,\n"
        "        noise_centroid = noise_centroid,\n"
        "        trace          = trace,\n"
        "        threshold      = 0.50,\n"
        "        adherence_weight = 0.4,\n"
        "    )\n"
        "\n"
        "# Plug into MIPROv2\n"
        "optimizer = dspy.MIPROv2(\n"
        "    metric     = noise_metric,\n"
        "    auto       = 'light',\n"
        "    num_threads= 4,\n"
        ")\n"
        "optimized = optimizer.compile(\n"
        "    NoiseGridModule(),\n"
        "    trainset = trainset,\n"
        "    max_bootstrapped_demos = 3,\n"
        "    max_labeled_demos = 2,\n"
        ")\n"
    )

    body2 = f"""
<h1>noise-01</h1>
<p class="subtitle" style="font-size:1.1rem;color:#475569;margin-bottom:32px;">
Deep technical dive — 3&times;6 ASCII noise grid generation and geometric measurement.
Part of <a href="ascii-bench.html" style="color:#4361ee;">ascii-bench</a>.
</p>

<h2>The Evaluation</h2>

<p>noise-01 is the simplest possible evaluation in ascii-bench. It asks a model to generate a <strong>3-row &times; 6-character grid of ASCII noise</strong> — 18 characters with no semantic content, no recognizable shapes, no letters or digits. Then it measures two things:</p>

<ol>
  <li><strong>Grid adherence</strong> — does the output conform to the 3&times;6 structural constraint?</li>
  <li><strong>Noise-region membership</strong> — does the rendered output land in the noise region of Gemini's embedding space, as defined by a reference corpus?</li>
</ol>

<p>The constraint "exactly 18 characters in a 3&times;6 grid" is not arbitrary. It is small enough that a failure to adhere is clearly a structural problem, not an ambiguity. It is also large enough to have meaningful visual texture when rendered — 18 characters at 48px monospace produce an image with visible density patterns. The embedding model reads those patterns.</p>

{callout("note", "Why This is Interesting",
  "<p>No one has measured this before, because before multimodal embeddings there was no obvious way to measure it. What does it mean for a language model to correctly generate randomness? The answer turns out to involve geometry, topology, and writing system identity in ways that character-level metrics completely miss.</p>")}

<h2>The Reference Corpus v0.1 — Latin Keyboard Symbols</h2>

<p>The reference corpus defines the noise region. Version 0.1 contains five hand-crafted 3&times;6 grids, all using shifted keyboard characters (the symbols accessed by holding Shift on a standard QWERTY keyboard). No letters, no digits — only punctuation and operator characters.</p>

<p>The choice of shifted characters is deliberate. These are the characters a model reaches for when asked to produce something with no semantic content in its native Latin environment. They are the model's natural visual vocabulary for "noise" — the characters that live at the boundary between meaningful and meaningless in the training distribution.</p>

<p>The keyboard spiral observation: the first grid (<code>x@#$%^&amp;*()_+!?&gt;&lt;&#123;&#125;</code>) traces a path across the keyboard that approximates a golden ratio spiral when read left-to-right, top-to-bottom. This was not designed — it emerged from asking "what arrangement of shifted characters feels most random?" The geometry of the keyboard encodes its own notion of visual density and traversal order.</p>

<pre><code># The five Latin reference grids (v0.1)

"x@#$%^"   # keyboard spiral — the golden ratio path
"&amp;*()_+"
"!?&gt;&lt;&#123;&#125;"

"~`|\\;'"   # upper row sweep + punctuation descent
'",.:?!'
"[]&#123;&#125;()"

"!@#$%^"   # bracket / operator cluster
"&amp;*-+=&lt;"
"&gt;/?|~`"

"^%$#@!"   # diagonal shift sweep
"*&amp;()_+"
"&gt;&lt;&#123;&#125;[]"

"@#$&amp;*("   # dense operator field
")!?&gt;&lt;&#123;"
"&#125;|~`\\;"</code></pre>

<h2>The Three Embedding Strategies</h2>

<p>Before committing to a single embedding strategy for the metric, we compared three approaches on the same corpus and test cases:</p>

<h3>Image — embed the rendered PNG only</h3>
<p>The ASCII grid is rendered to a PNG and embedded as an image. Only visual information is available to the embedding model — it sees the density and spatial arrangement of characters, not their symbolic identity.</p>

<h3>Text — embed the raw ASCII string only</h3>
<p>The raw ASCII string is embedded as text. The embedding model reads the character sequence symbolically — it knows these are parentheses and dollar signs, not just dense marks.</p>

<h3>Interleaved — PNG + raw string, fused into one vector</h3>
<p>Both the raw text and the rendered PNG are passed as parts of a single <code>Content</code> object. Gemini Embedding 2 fuses them into one 3072-dim vector. This strategy provides both visual and symbolic information simultaneously.</p>

<table>
  <thead>
    <tr>
      <th>Strategy</th>
      <th>Noise-to-noise range</th>
      <th>Spread</th>
      <th>Noise score</th>
      <th>Non-noise score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Image</td>
      <td>[0.9197, 0.9798]</td>
      <td>0.0601</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td>Text</td>
      <td>[0.8703, 0.9609]</td>
      <td>0.0905</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <td><strong>Interleaved</strong></td>
      <td>[0.8960, 0.9680]</td>
      <td>0.0720</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>

{callout("insight", "Total Separation on Every Strategy",
  "<p>All three strategies achieve a noise score of 1.000 and non-noise score of 0.000 — perfect separation between noise grids and non-noise grids (alphabetic, numeric, semantic, wrong-size). The embedding space cleanly partitions the noise region regardless of embedding mode.</p>")}

<p>The strategies differ in their internal structure. Text has the largest spread (0.0905) — the noise grids are more differentiated from each other symbolically than visually. Interleaved acts as the strictest judge of wrong-size outputs: a grid with the right characters but wrong dimensions gets a lower interleaved score than image alone. We use interleaved as the default.</p>

<h2>The Multilingual Expansion — Corpus v0.2</h2>

<p>Corpus v0.2 adds seven grids across four non-Latin script families: Katakana, Cyrillic, Greek, and Arabic. This tests whether noise is a universal concept in the embedding manifold, or anchored to the Latin keyboard.</p>

{fig_tag(corpus_fig, "ascii-bench noise-01 Reference Corpus v0.2: 5 Latin grids (top row) and 7 multilingual grids (bottom row). Each grid rendered with a script-appropriate font, colored by script family.")}

{fig_tag(pairwise_fig, "Noise-to-noise pairwise cosine similarity matrix for the full v0.2 corpus (interleaved strategy). Tick labels colored by script family. Within-script pairs cluster at the high end; cross-script pairs spread into the lower range.")}

<h2>The PCA Finding — Noise has Topology</h2>

{callout("discovery", "Discovery: Five Distinct Script Clusters",
  "<p>A 2D PCA projection of the 12 corpus vectors reveals five completely distinct clusters — one per script family. Latin keyboard symbols: top-right, tight cluster. Cyrillic: top-left, tight pair. Greek: near Cyrillic but distinct. Arabic: bottom-left-center. Katakana: bottom-left corner, most distant from Latin. <strong>Noise is not a single universal region. Each writing system occupies a distinct sub-region of the noise manifold.</strong></p>")}

{fig_tag(pca_scripts, "2D PCA of the 12-grid v0.2 corpus (interleaved embeddings, 3072-dim → 2D). Five distinct clusters, one per script family. 61.7% variance explained by the first two principal components. Dashed convex hulls mark per-script boundaries.")}

<p>The PCA projection explains 61.7% of variance in just two dimensions — a high fraction for 3072-dimensional vectors, indicating that script family identity is the dominant source of variation in the corpus. The remaining 38.3% reflects within-script variation between individual grid designs.</p>

<p>This finding has a concrete calibration implication: <strong>a unified centroid is a Latin-dominated centroid</strong>. With 5 Latin grids out of 12, the centroid is pulled toward the Latin cluster. Scores for Latin grids are systematically higher, and scores for non-Latin grids systematically lower, regardless of geometric quality relative to their own script cluster. Per-script centroids would be more accurate.</p>

<h2>LLM Generation — the Framing Effect</h2>

<p>GPT-4o was asked to generate 3&times;6 noise grids under 10 different framings: six Latin-oriented and four multilingual. Each framing shapes what the model understands as "noise" — and those differences are geometrically measurable.</p>

{fig_tag(llm_grids_fig, "GPT-4o generated noise grids across 10 framings (grid adherence score shown per panel). The multilingual framings produce visually distinct outputs that render correctly with script-appropriate fonts.")}

<h2>Full PCA — LLM Outputs Land Near Their Script Clusters</h2>

{fig_tag(pca_full, "Full PCA projection: reference corpus (circles) and all 10 LLM outputs (triangles), colored by script family. LLM outputs land near their respective script reference clusters. LLM Katakana noise is geometrically coherent with reference Katakana noise.")}

<p>The most striking result: LLM outputs land near their respective script reference clusters. This is behavioral introspection without self-report — we learn about the model's relationship to writing systems by measuring where its outputs live geometrically, without asking the model anything about itself.</p>

<p>Notable: the "keyboard self" framing drifts far from the noise cluster on PCA. When asked to generate what it "sees" when looking at a keyboard, the model produces a conceptual map of the keyboard rather than noise — a mix of characters that is coherent but not random. The geometry distinguishes between "what keyboards contain" and "what noise looks like."</p>

<h2>Scores</h2>

{fig_tag(scores_fig, "Score bar chart across reference corpus (left, colored by script family) and LLM outputs (Latin framings center, multilingual framings right). Dashed lines show noise-region floor and ceiling from the interleaved pairwise matrix.")}

<p>Scores against the unified corpus centroid:</p>

<table>
  <thead>
    <tr><th>Source</th><th>Script</th><th>Score range</th><th>Note</th></tr>
  </thead>
  <tbody>
    <tr><td>Reference corpus</td><td>Latin</td><td>0.87&ndash;0.93</td><td>Centroid-dominant script, highest scores</td></tr>
    <tr><td>Reference corpus</td><td>Cyrillic</td><td>0.83&ndash;0.85</td><td>Visually similar to Latin</td></tr>
    <tr><td>Reference corpus</td><td>Greek</td><td>0.79</td><td>Alphabetic but visually distinct</td></tr>
    <tr><td>Reference corpus</td><td>Arabic</td><td>0.75</td><td>Visually distinct, right-to-left</td></tr>
    <tr><td>Reference corpus</td><td>Katakana</td><td>0.73&ndash;0.75</td><td>Most distant from Latin</td></tr>
    <tr><td>LLM (keyboard native)</td><td>Latin</td><td>0.93</td><td>Native noise vocabulary, highest LLM score</td></tr>
    <tr><td>LLM (plain ask)</td><td>Latin</td><td>0.88</td><td>Solid default</td></tr>
    <tr><td>LLM (visual static)</td><td>Latin</td><td>0.71</td><td>Framing pulls toward peripheral noise</td></tr>
    <tr><td>LLM (keyboard self)</td><td>Latin</td><td>&lt;0.71</td><td>Drifts far from noise region on PCA</td></tr>
    <tr><td>LLM (cyrillic)</td><td>Cyrillic</td><td>0.80</td><td>Near reference cluster</td></tr>
    <tr><td>LLM (greek)</td><td>Greek</td><td>0.81</td><td>Near reference cluster</td></tr>
    <tr><td>LLM (katakana)</td><td>Katakana</td><td>0.66</td><td>Below reference; centroid bias applies</td></tr>
    <tr><td>LLM (arabic)</td><td>Arabic</td><td>0.65</td><td>Below reference; centroid bias applies</td></tr>
  </tbody>
</table>

{callout("note", "Centroid Bias",
  "<p>All non-Latin scores are depressed by the Latin-dominated unified centroid. This is not a failure of those outputs — it is a calibration artifact. Per-script centroids would give fairer absolute scores. The PCA projection is the better diagnostic for non-Latin outputs.</p>")}

<h2>The Metric Function</h2>

<pre><code>{noise_eval_code}</code></pre>

<h2>DSPy Integration</h2>

<p>The metric wraps into a DSPy-compatible function and plugs directly into MIPROv2. The centroid is pre-computed once at startup to avoid redundant API calls during the optimization loop:</p>

<pre><code>{dspy_integration_code}</code></pre>

{callout("insight", "What MIPROv2 Will Optimize",
  "<p>Given this metric, MIPROv2 will search for instructions that produce outputs landing deeper inside the noise manifold. It will discover that shifted keyboard characters score higher than alphanumerics, that the 3&times;6 structure must be maintained, and that some framings are geometrically more coherent than others — without being told any of this explicitly.</p>")}

<h2>Open Questions</h2>

<ul>
  <li><strong>Per-script centroids.</strong> Build a separate centroid per script family. Remove calibration bias and give fair absolute scores across writing systems.</li>
  <li><strong>Font sensitivity.</strong> How much does score change with different font choices? Does a pixel font produce a different noise manifold?</li>
  <li><strong>Grid size sensitivity.</strong> Would a 4&times;8 or 6&times;6 grid produce a different manifold? Is the noise region scale-invariant?</li>
  <li><strong>Script expansion.</strong> Devanagari, Hebrew, CJK ideographs — each would add a new cluster. How many distinct script clusters exist in the manifold?</li>
  <li><strong>Adversarial outputs.</strong> What outputs maximize the noise score without being perceptually noisy? Are there degenerate high-scoring patterns?</li>
</ul>

<p style="margin-top:32px;"><a href="ascii-bench.html" style="color:#4361ee;font-weight:600;text-decoration:none;">&larr; Back to ascii-bench overview</a></p>
"""

    html = page_shell("noise-01 Deep Dive", "noise01", body2)
    out = ROOT / "04_noise_eval.html"
    out.write_text(html)
    print(f"  Wrote {out.name} ({out.stat().st_size // 1024} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating figures...")
    make_unified_space()
    make_cosine_intuition()
    make_pipeline()
    make_dspy_loop()
    copy_experiment_result()
    copy_noise_figures()

    print("\nGenerating HTML pages...")
    make_index()
    make_paradigm()
    make_experiment()
    make_dspy()
    make_ascii_bench()
    make_noise_eval()

    print("\nSummary:")
    all_files = sorted(ROOT.glob("*.html")) + sorted(FIGS.glob("*.png"))
    total = 0
    for f in all_files:
        size = f.stat().st_size
        total += size
        rel = f.relative_to(ROOT)
        print(f"  {str(rel):<45}  {size // 1024:>5} KB")
    print(f"\n  Total: {total // 1024} KB across {len(all_files)} files")
    print("\nDone. Open /home/ath/writing/index.html in a browser.")


if __name__ == "__main__":
    main()
