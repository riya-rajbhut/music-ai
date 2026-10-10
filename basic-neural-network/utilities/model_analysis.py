from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

base_dir = Path(__file__).resolve().parents[1]
resources_dir = base_dir / "resources"
reports_dir = base_dir / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# 1. Load trained model weights from the project resources folder
pt_name = "model_train_ddp_v7_transformer_REMI_57"
checkpoint_path = resources_dir / f"{pt_name}.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# 2. Extract Embedding Weights for pitch tokens (0 to 127)
# Handle different checkpoint formats / key names (e.g., 'model_state_dict', 'state_dict', or direct state dict)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint if isinstance(checkpoint, dict) else {}

# Try to find a plausible embedding weight key
embedding_key = None
for k, v in state_dict.items():
    low = k.lower()
    if "weight" in low and ("token" in low or "embed" in low or "embedding" in low):
        try:
            shape = tuple(v.shape) if hasattr(v, "shape") else (np.array(v).shape)
        except Exception:
            continue
        # prefer keys where first dimension covers at least 128 pitch tokens
        if len(shape) >= 1 and shape[0] >= 128:
            embedding_key = k
            break

if embedding_key is None:
    # fallback: look for any weight with >=128 rows
    for k, v in state_dict.items():
        try:
            shape = tuple(v.shape) if hasattr(v, "shape") else (np.array(v).shape)
        except Exception:
            continue
        if len(shape) >= 1 and shape[0] >= 128 and "weight" in k.lower():
            embedding_key = k
            break

if embedding_key is None:
    raise KeyError(f"Could not find embedding weight in checkpoint. Available keys: {list(state_dict.keys())[:40]}")

print(f"Using embedding key: {embedding_key}")
tensor = state_dict[embedding_key]
if hasattr(tensor, "cpu"):
    tensor = tensor.cpu()
token_weights = tensor.numpy()
token_embeddings = token_weights[:128]  # Only standard pitch tokens (0-127)

# 3. Compute Chroma and Octave labels for pitches 0-127
pitches = np.arange(128)
chroma_labels = pitches % 12    # 0 = C, 1 = C#, ..., 11 = B
octave_labels = pitches // 12   # 0 to 10

# 4. Compute 2D t-SNE Projection
tsne = TSNE(n_components=2, perplexity=15, random_state=53, max_iter=1000)
embeddings_2d = tsne.fit_transform(token_embeddings)

# 5. Plot Side-by-Side
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot A: Colored by Chroma (Pitch Class)
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
scatter1 = axes[0].scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1], 
    c=chroma_labels, 
    cmap='twilight', # Cyclic colormap ideal for pitch classes
    s=70, 
    edgecolors='k', 
    alpha=0.85
)
cbar1 = fig.colorbar(scatter1, ax=axes[0], ticks=range(12))
cbar1.ax.set_yticklabels(note_names)
axes[0].set_title("t-SNE of Pitch Embeddings (Colored by Chroma / Pitch Class)", fontsize=13)
axes[0].grid(True, linestyle="--", alpha=0.5)

# Plot B: Colored by Octave (Register)
scatter2 = axes[1].scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1], 
    c=octave_labels, 
    cmap='viridis', 
    s=70, 
    edgecolors='k', 
    alpha=0.85
)
cbar2 = fig.colorbar(scatter2, ax=axes[1])
cbar2.set_label('Octave Number')
axes[1].set_title("t-SNE of Pitch Embeddings (Colored by Octave)", fontsize=13)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
out_path = reports_dir / f"{pt_name}.png"
plt.savefig(out_path, dpi=300)
print(f"Saved TSNE plot to: {out_path}")
plt.show()