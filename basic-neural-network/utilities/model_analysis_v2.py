"""
Pitch-embedding analysis for the 2D-RoPE music transformer (v8 plain / v9 factored checkpoints).

Changes vs. the original model_analysis.py
  1. Factored (v9) checkpoints: analyses the EFFECTIVE embedding  free + pitch_class[p % 12] + octave[p // 12]
     (what the model actually uses for input lookup and for the tied output head). The original script picked
     the first tensor that looked like an embedding, which is `token_embedding.free.weight` -- only the residual.
  2. Only the piano range (MIDI 21-108) is analysed. MAESTRO is piano-only, so pitch tokens outside it are never
     seen as inputs and stay at their random initialisation; they are pure noise in a plot.
  3. Vectors are mean-centred and L2-normalised before t-SNE (cosine geometry), t-SNE uses PCA initialisation.
  4. A quantitative structure report is printed (nearest neighbours, interval profile, share of the shared part),
     plus a cosine-similarity heatmap / interval profile / PCA figure. t-SNE alone cannot answer "did it learn?".
     An INIT REFERENCE (same metrics on an untrained model) is printed too: a factored model has octave/pitch-class
     similarity by construction, so only the difference from init is evidence of learning.
  5. Chroma colours use a proper 12-step cycle; torch.load uses weights_only=True (the checkpoint is plain tensors).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PT_NAME = "best_model_2DROPE_Contrastive_36"   # edit, or pass --ckpt
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ------------------------------------------------------------------ loading
def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    return {k.removeprefix("module."): v for k, v in sd.items()}


def pitch_embeddings(sd, pitches):
    """Returns (effective, free, shared, description); free/shared are None for plain checkpoints."""
    pitches = np.asarray(pitches)
    if "token_embedding.free.weight" in sd:  # v9 factored model
        free = sd["token_embedding.free.weight"].float().numpy()[pitches]
        pc = sd["token_embedding.pitch_class.weight"].float().numpy()[pitches % 12]
        octv = sd["token_embedding.octave.weight"].float().numpy()[pitches // 12]
        shared = pc + octv
        return free + shared, free, shared, "factored: free + pitch_class[p%12] + octave[p//12]"
    if "token_embedding.weight" in sd:  # v8 / baseline model
        w = sd["token_embedding.weight"].float().numpy()[pitches]
        return w, None, None, "plain: token_embedding.weight"
    raise KeyError(f"No token embedding found. Keys: {list(sd.keys())[:30]}")


# ------------------------------------------------------------------ metrics
def prep(E):
    E = E - E.mean(axis=0, keepdims=True)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def interval_profile(S, max_k=24):
    """Mean cosine similarity between pitches k semitones apart (pitches must be consecutive)."""
    return np.array([np.diagonal(S, offset=k).mean() for k in range(1, max_k + 1)])


def print_report(name, res):
    print(
        f"[{name}] NN within 1-2 semitones: {res['nn_near']*100:5.1f}% (chance {res['chance_near']*100:4.1f}%) | "
        f"NN is an octave-equivalent pitch: {res['nn_oct']*100:5.1f}% (chance {res['chance_oct']*100:4.1f}%) | "
        f"octave bump {res['octave_bump']:+.3f} | fifth bump {res['fifth_bump']:+.3f}"
    )


def structure_report(name, E, pitches, verbose=True):
    X = prep(E)
    S = X @ X.T
    n = len(pitches)
    d = np.abs(pitches[:, None] - pitches[None, :])
    off = ~np.eye(n, dtype=bool)

    S_nn = S.copy()
    np.fill_diagonal(S_nn, -np.inf)
    nn_d = d[np.arange(n), S_nn.argmax(axis=1)]

    near = lambda m: (m & off).sum(axis=1) / (n - 1)
    res = {
        "nn_near": float(np.mean((nn_d >= 1) & (nn_d <= 2))),
        "chance_near": float(near((d >= 1) & (d <= 2)).mean()),
        "nn_oct": float(np.mean((nn_d > 0) & (nn_d % 12 == 0))),
        "chance_oct": float(near(d % 12 == 0).mean()),
    }
    prof = interval_profile(S)
    res["profile"] = prof
    res["octave_bump"] = float(prof[11] - 0.5 * (prof[10] + prof[12]))   # d=12 vs 11/13
    res["fifth_bump"] = float(prof[6] - 0.5 * (prof[5] + prof[7]))       # d=7 vs 6/8
    if verbose:
        print_report(name, res)
    return res, S


def init_reference(kind, dim, pitches, n_seeds=5):
    """
    Same metrics for an UNTRAINED model with the training script's init
    (plain / free: N(0, 0.02); pitch_class and octave: N(0, 0.01)), averaged over a few seeds.
    A factored model has pitch-class/octave similarity BY CONSTRUCTION, so its trained numbers must be compared
    with this reference, not with chance.
    """
    n = len(pitches)
    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        if kind == "factored":
            free = rng.normal(0, 0.02, (n, dim))
            pc = rng.normal(0, 0.01, (12, dim))
            octv = rng.normal(0, 0.01, (int(pitches.max()) // 12 + 1, dim))
            E = free + pc[pitches % 12] + octv[pitches // 12]
        else:
            E = rng.normal(0, 0.02, (n, dim))
        results.append(structure_report("ref", E, pitches, verbose=False)[0])
    avg = {k: float(np.mean([r[k] for r in results])) for k in results[0] if k != "profile"}
    avg["profile"] = np.mean([r["profile"] for r in results], axis=0)
    return avg


# ------------------------------------------------------------------ plots
def tsne_xy(X, seed=53):
    perp = min(15, max(5, (len(X) - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(X)


def plot_tsne(xy, pitches, title, path):
    chroma = pitches % 12
    octave = pitches // 12 - 1  # scientific pitch notation: MIDI 60 = C4
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sc1 = axes[0].scatter(xy[:, 0], xy[:, 1], c=chroma, cmap="hsv", vmin=0, vmax=12, s=70, edgecolors="k", alpha=0.85)
    cb1 = fig.colorbar(sc1, ax=axes[0], ticks=np.arange(12) + 0.0)
    cb1.ax.set_yticklabels(NOTE_NAMES)
    for (x, y), p in zip(xy, pitches):
        if p % 12 == 0:
            axes[0].annotate(f"C{p // 12 - 1}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set_title(f"{title}\nColoured by pitch class (C's labelled)", fontsize=12)

    sc2 = axes[1].scatter(xy[:, 0], xy[:, 1], c=octave, cmap="viridis", s=70, edgecolors="k", alpha=0.85)
    cb2 = fig.colorbar(sc2, ax=axes[1])
    cb2.set_label("Octave (scientific; C4 = MIDI 60)")
    axes[1].set_title(f"{title}\nColoured by octave", fontsize=12)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    return fig


def plot_structure(S, X, profiles, pitches, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    lim = float(np.percentile(np.abs(S[~np.eye(len(S), dtype=bool)]), 99))
    im = axes[0].imshow(S, cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower",
                        extent=[pitches[0], pitches[-1], pitches[0], pitches[-1]])
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Cosine similarity between pitch embeddings\n(centred; octave-equivalence = off-diagonals every 12)")
    axes[0].set_xlabel("MIDI pitch")
    axes[0].set_ylabel("MIDI pitch")

    ks = np.arange(1, len(next(iter(profiles.values()))) + 1)
    for label, prof in profiles.items():
        axes[1].plot(ks, prof, marker="o", ms=4, label=label)
    for k in (12, 24):
        axes[1].axvline(k, color="gray", linestyle="--", alpha=0.6)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("Interval (semitones)")
    axes[1].set_ylabel("Mean cosine similarity")
    axes[1].set_title("Similarity vs interval\n(bumps at 12/24 = octave equivalence, at 7 = fifth)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    pca = PCA(n_components=2).fit(X)
    xy = pca.transform(X)
    axes[2].plot(xy[:, 0], xy[:, 1], color="gray", lw=0.6, alpha=0.6)
    sc = axes[2].scatter(xy[:, 0], xy[:, 1], c=pitches, cmap="viridis", s=45, edgecolors="k")
    fig.colorbar(sc, ax=axes[2]).set_label("MIDI pitch")
    axes[2].set_title(f"PCA of embeddings (variance explained {pca.explained_variance_ratio_[0]*100:.0f}% + "
                      f"{pca.explained_variance_ratio_[1]*100:.0f}%)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    return fig


# ------------------------------------------------------------------ main
def main():
    base_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=base_dir / "resources" / f"{PT_NAME}.pt")
    ap.add_argument("--out-dir", type=Path, default=base_dir / "reports")
    ap.add_argument("--pitch-min", type=int, default=21)   # A0
    ap.add_argument("--pitch-max", type=int, default=108)  # C8
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = args.ckpt.stem

    sd = load_state_dict(args.ckpt)
    pitches = np.arange(args.pitch_min, args.pitch_max + 1)
    eff, free, shared, desc = pitch_embeddings(sd, pitches)
    print(f"Checkpoint: {args.ckpt}\nEmbedding used: {desc}\nPitches analysed: {pitches[0]}..{pitches[-1]} ({len(pitches)})\n")

    if shared is not None:
        sq = lambda a: float((a ** 2).sum(axis=1).mean())
        print(f"Mean squared norm  free: {sq(free):.4f} | shared (pitch_class+octave): {sq(shared):.4f} | "
              f"effective: {sq(eff):.4f}  ->  shared part = {sq(shared) / sq(eff) * 100:.1f}% of embedding energy\n")

    reports, profiles = {}, {}
    reports["effective"], S = structure_report("effective", eff, pitches)
    profiles["effective"] = reports["effective"]["profile"]
    if free is not None:
        reports["free"], _ = structure_report("free-only (what the old script plotted)", free, pitches)
        profiles["free only"] = reports["free"]["profile"]
    ref = init_reference("factored" if shared is not None else "plain", eff.shape[1], pitches)
    print_report("INIT REFERENCE (untrained, mean of 5)", ref)
    profiles["init (untrained)"] = ref["profile"]
    print("-> Judge learning against the INIT REFERENCE line, not against chance. For factored models the octave "
          "numbers are high even untrained; the fifth bump and the change vs init are the learned signal.")
    print("\nMean cosine by interval (effective): " +
          " ".join(f"{k}:{profiles['effective'][k - 1]:+.2f}" for k in (1, 2, 3, 4, 5, 6, 7, 8, 12, 19, 24)))

    X = prep(eff)
    plot_tsne(tsne_xy(X), pitches, f"t-SNE of pitch embeddings ({name})", args.out_dir / f"{name}_tsne.png")
    plot_structure(S, X, profiles, pitches, args.out_dir / f"{name}_structure.png")
    print(f"\nSaved: {args.out_dir / (name + '_tsne.png')}\n       {args.out_dir / (name + '_structure.png')}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()