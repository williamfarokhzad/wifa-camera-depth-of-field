"""Render original WebP figures. Optional dependency: matplotlib with Pillow."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dof import evaluate

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#21374d", "text.color": "#21374d",
                     "xtick.color": "#21374d", "ytick.color": "#21374d"})

def load(name):
    return json.loads((ROOT / "examples" / (name + ".json")).read_text())

def save(fig, name):
    fig.savefig(ROOT / "diagrams" / (name + ".webp"), dpi=100,
                pil_kwargs={"quality": 89, "method": 6})
    plt.close(fig)

def render():
    fig, ax = plt.subplots(figsize=(8.4, 5), layout="constrained")
    ax.axvspan(280, 320, color="#e5efef", label="Part depth: 280–320 mm")
    distances = range(270, 336)
    for name, color, style, label in (("f8", "#953700", "-", "f/8"),
                                     ("f11", "#00665c", "--", "f/11")):
        inp = load(name)
        ys = [evaluate(**(inp | {"target_mm": [z, z]}))["endpoint_blur_px"][0]
              for z in distances]
        ax.plot(distances, ys, color=color, ls=style, linewidth=3, label=label)
    ax.axhline(2, color="#32455a", linestyle=":", linewidth=2, label="2 px budget")
    ax.set(xlabel="Object distance from ideal lens (mm)", ylabel="Geometric blur (px)",
           title="A smaller aperture reduces geometric defocus", ylim=(0, 3.8))
    ax.legend(loc="upper center", ncols=2, fontsize=11, framealpha=1)
    ax.grid(axis="y", alpha=.2)
    save(fig, "dof-blur-by-distance")

    fig, ax = plt.subplots(figsize=(8.4, 5), layout="constrained")
    ax.axvspan(280, 320, color="#e5efef", label="Required part depth")
    for y, (name, color) in enumerate((("f8", "#953700"), ("f11", "#00665c"),
                                       ("one-pixel", "#953700"))):
        r = evaluate(**load(name))
        points = [r["near_mm"], r["far_mm"]]
        ax.plot(points, [y, y], "o-", color=color, linewidth=4)
        verdict = "Covers part" if r["geometric_pass"] else "Too narrow"
        ax.text(302, y + .13, verdict, ha="center", color=color, fontsize=12)
    ax.set(xlim=(271, 333), ylim=(-.32, 2.52), xlabel="Object distance from ideal lens (mm)",
           title="A sharper blur budget narrows the usable interval")
    ax.set_yticks(range(3), ["f/8 · 2 px", "f/11 · 2 px", "f/11 · 1 px"])
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="x", alpha=.2)
    save(fig, "dof-working-intervals")

if __name__ == "__main__":
    render()
