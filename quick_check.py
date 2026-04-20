"""Quick visual check of gs_generator outputs.

Usage:
    python quick_check.py

Requires: numpy, matplotlib (and optionally cupy for GPU).
Runs on CPU by default so you can smoke-test anywhere.
"""
import matplotlib.pyplot as plt
from gs_generator import generate

# --- 1. Basic labyrinth batch -----------------------------------------------
A, B, meta = generate(
    batch=8,
    H=192, W=192,
    n_steps=6000,
    region="labyrinth",
    use_gpu=True,   # flip to True once CuPy is installed
    seed=42,
)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(B[i], cmap="gray")
    ax.set_title(f"f={meta['f'][i]:.4f}  k={meta['k'][i]:.4f}", fontsize=9)
    ax.axis("off")
plt.suptitle("Gray-Scott labyrinths")
plt.tight_layout()
plt.savefig("check_labyrinth.png", dpi=100, bbox_inches="tight")
plt.show()   # pops up a window if you're on a desktop

# --- 2. Diverse transient states (great for ML dataset variety) -------------
A, B, meta = generate(
    batch=8, H=192, W=192, n_steps=6000,
    region="labyrinth",
    random_stop=True,
    random_stop_min_frac=0.15,
    use_gpu=False, seed=11,
)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(B[i], cmap="gray")
    ax.set_title(f"stop={int(meta['stop_steps'][i])}", fontsize=9)
    ax.axis("off")
plt.suptitle("Random-stop transient states")
plt.tight_layout()
plt.savefig("check_random_stop.png", dpi=100, bbox_inches="tight")
plt.show()

print("Saved: check_labyrinth.png, check_random_stop.png")
