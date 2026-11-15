"""
theory_over_checkpoints.py  --  how do the music-theory measurements change from checkpoint to checkpoint?

Reads a manifest (JSON list) such as
  [{"label": "ARIA 1k",  "path": "/kaggle/working/ckpts/aria_v1/aria_pretrained_best.pt", "canon": false},
   ...
   {"label": "canon ep3", "path": "/kaggle/working/ckpts/canon_v2/best_maestro_canon.pt",  "canon": true}]
and, for every checkpoint, measures

  embedding tests (no data needed)
    - pitch-class vectors: strength of the circle-of-fifths and chromatic-circle structure (minus the Spearman
      correlation between vector similarity and interval distance; higher = stronger) with a permutation p-value
    - Fourier share of the fifths (k=5) and semitone (k=1) components
  with --with_keys (needs MAESTRO and a GPU; about a minute per checkpoint)
    - key awareness: correlation between the model's key-relative next-pitch-class distribution and the
      Krumhansl-Kessler profiles (major and minor)
    - error structure: excess share of minor-third and tritone errors over what unrelated guesses would give
    - pitch accuracy on the scored notes (second half of each window)

Outputs in --out_dir:  theory_over_checkpoints.png (lines), theory_ckpt_matrices.png (similarity matrix per checkpoint),
                       theory_ckpt_pca.png (2-D map per checkpoint), theory_over_checkpoints.csv

Usage (from the folder that holds music_common.py and theory_plots.py):
  python theory_over_checkpoints.py --manifest /kaggle/working/manifest.json --out_dir /kaggle/working/theory_ckpt
  python theory_over_checkpoints.py --manifest ... --with_keys

"canon": true means the checkpoint was trained on the canonical chord order; it is evaluated on that tokenisation.
Pitch accuracy is NOT comparable across a change of token order, so the figure marks where the order changes.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from theory_plots import (FIFTHS_ORDER, KK_MAJOR, KK_MINOR, NAMES, embedding_stats, error_stats, estimate_keys)


# ------------------------------------------------------------------------------------------------ measurements
def embedding_metrics(X_pc, X_oct, n_perm, seed):
    out = {}
    for tag, X in (("pc", X_pc), ("oct", X_oct)):
        s = embedding_stats(X, n_perm, seed)
        out.update({f"{tag}_fifths": -s["rho5"], f"{tag}_chrom": -s["rho1"], f"{tag}_p_fifths": s["p_rho5"],
                    f"{tag}_p_chrom": s["p_rho1"], f"{tag}_k5": s["shares"][4] * 100, f"{tag}_k1": s["shares"][0] * 100,
                    f"{tag}_p_k5": s["p_share"][4], f"{tag}_p_k1": s["p_share"][0]})
        for k in (2, 3, 4, 6):                   # all six Fourier components (k = 1 and 5 above)
            out[f"{tag}_k{k}"] = s["shares"][k - 1] * 100
            out[f"{tag}_p_k{k}"] = s["p_share"][k - 1]
        if tag == "pc":
            U, S, _ = np.linalg.svd(s["Xc"], full_matrices=False)
            out["_C"] = s["C"]
            out["_xy"] = U[:, :2] * S[:2]
    return out


def prediction_metrics(st):
    out = {}
    for m, nm in enumerate(("major", "minor")):
        kk = (KK_MAJOR if m == 0 else KK_MINOR)
        kk = kk / kk.sum()
        pred = st["rel_pred"][m] / max(st["rel_notes"][m], 1)
        true = st["rel_true"][m] / max(st["rel_true"][m].sum(), 1)
        out[f"r_kk_{nm}"] = float(np.corrcoef(pred, kk)[0, 1])
        out[f"r_true_{nm}"] = float(np.corrcoef(pred, true)[0, 1])
        out[f"windows_{nm}"] = int(st["rel_windows"][m])
    obs, exp = error_stats(st["err_int"], st["err_true_pc"], st["err_pred_pc"])
    out["minor3_excess"] = float((obs[3] + obs[9] - exp[3] - exp[9]) * 100)
    out["tritone_excess"] = float((obs[6] - exp[6]) * 100)
    out["pitch_acc"] = float(1 - st["err_int"].sum() / st["n_notes"]) * 100
    return out


def collect_model_stats(model, songs, dev, min_corr):
    import torch
    import torch.utils.data as data
    from torch.amp import autocast
    from music_common import WindowDataset, concat_songs
    tok, off = concat_songs(songs["validation"])
    ds = WindowDataset(tok, off, np.arange(len(off) - 1), 768, 384)
    loader = data.DataLoader(ds, batch_size=32, num_workers=2)
    N0 = 128
    M = torch.zeros(128, 12, device=dev)
    M[torch.arange(128), torch.arange(128) % 12] = 1.0
    st = dict(rel_pred=np.zeros((2, 12)), rel_true=np.zeros((2, 12)), rel_notes=np.zeros(2), rel_windows=np.zeros(2, int),
              err_int=np.zeros(12), err_true_pc=np.zeros(12), err_pred_pc=np.zeros(12), n_notes=0)
    with torch.no_grad():
        for x, y, a in loader:
            x, y, a = x.to(dev), y.to(dev), a.to(dev)
            with autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits, _ = model(x, a)
            probs = torch.softmax(logits[:, 0::3, :128].float()[:, N0:], -1)
            tgt = y[:, 0::3][:, N0:]
            pred = probs.argmax(-1)
            B, N = tgt.shape
            wrong = pred != tgt
            st["n_notes"] += tgt.numel()
            st["err_int"] += np.bincount(((pred % 12 - tgt % 12) % 12)[wrong].cpu().numpy(), minlength=12)
            st["err_true_pc"] += np.bincount((tgt % 12)[wrong].cpu().numpy(), minlength=12)
            st["err_pred_pc"] += np.bincount((pred % 12)[wrong].cpu().numpy(), minlength=12)
            ctx = (x[:, 1::3][:, :N0] % 12).cpu().numpy()
            hist = np.stack([np.bincount(r, minlength=12) for r in ctx])
            tonic, mode, corr = estimate_keys(hist)
            keep = corr >= min_corr
            if keep.any():
                pc = probs @ M
                tn = torch.as_tensor(tonic, device=dev)
                idx = (torch.arange(12, device=dev)[None] + tn[:, None]) % 12
                rel = pc.gather(2, idx[:, None, :].expand(B, N, 12)).sum(1).cpu().numpy()
                true_rel = ((tgt % 12 - tn[:, None]) % 12).cpu().numpy()
                for m in (0, 1):
                    sel = keep & (mode == m)
                    if sel.any():
                        st["rel_pred"][m] += rel[sel].sum(0)
                        st["rel_notes"][m] += sel.sum() * N
                        st["rel_windows"][m] += int(sel.sum())
                        st["rel_true"][m] += np.bincount(true_rel[sel].reshape(-1), minlength=12)
    return st


# ------------------------------------------------------------------------------------------------ plots
def _series(ax, x, y, p, color, label):
    ax.plot(x, y, "-", color=color, lw=1.5, label=label)
    sig = np.array(p) < 0.05
    ax.scatter(np.array(x)[sig], np.array(y)[sig], color=color, s=36, zorder=3)
    ax.scatter(np.array(x)[~sig], np.array(y)[~sig], facecolors="white", edgecolors=color, s=36, zorder=3)


def _mark_changes(ax, runs):
    for i in range(1, len(runs)):
        if runs[i]["canon"] != runs[i - 1]["canon"]:
            ax.axvline(i - 0.5, color="grey", ls="--", lw=1)


def plot_lines(res, runs, with_keys, path):
    x = list(range(len(runs)))
    labels = [r["label"] for r in runs]
    nrows = 2 if with_keys else 1
    fig, ax = plt.subplots(nrows, 3, figsize=(17, 5 * nrows), squeeze=False)
    blue, orange = "tab:blue", "tab:orange"
    for a, tag, title in ((ax[0, 0], "pc", "Pitch-class component of the embedding"),
                          (ax[0, 1], "oct", "Full pitch vectors averaged over octaves")):
        _series(a, x, [r[f"{tag}_fifths"] for r in res], [r[f"{tag}_p_fifths"] for r in res], blue, "circle of fifths")
        _series(a, x, [r[f"{tag}_chrom"] for r in res], [r[f"{tag}_p_chrom"] for r in res], orange, "chromatic circle")
        a.set_title(title)
        a.set_ylabel("structure strength (minus Spearman)")
        a.axhline(0, color="grey", lw=0.5)
    a = ax[0, 2]
    _series(a, x, [r["pc_k5"] for r in res], [r["pc_p_k5"] for r in res], blue, "fifths component (k=5)")
    _series(a, x, [r["pc_k1"] for r in res], [r["pc_p_k1"] for r in res], orange, "semitone component (k=1)")
    a.set_title("Fourier share, pitch-class component (%)")
    if with_keys:
        a = ax[1, 0]
        a.plot(x, [r["r_kk_major"] for r in res], "o-", color=blue, label="major keys")
        a.plot(x, [r["r_kk_minor"] for r in res], "o-", color=orange, label="minor keys")
        a.set_title("Match with Krumhansl-Kessler profile (r)")
        a = ax[1, 1]
        a.plot(x, [r["minor3_excess"] for r in res], "o-", color=blue, label="minor-third errors")
        a.plot(x, [r["tritone_excess"] for r in res], "o-", color=orange, label="tritone errors")
        a.axhline(0, color="grey", lw=0.5)
        a.set_title("Error share above chance (points)")
        a = ax[1, 2]
        a.plot(x, [r["pitch_acc"] for r in res], "o-", color=blue, label="pitch accuracy")
        a.set_title("Pitch accuracy (%), not comparable across the dashed line")
    for a in ax.ravel():
        _mark_changes(a, runs)
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        a.legend(fontsize=8)
    fig.suptitle("Music-theory measurements across checkpoints (filled marker: p < 0.05 against shuffled labels; "
                 "dashed line: token order changes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}", flush=True)


def plot_matrices(res, runs, path):
    n = len(runs)
    ncols = 4
    nrows = -(-n // ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for i, a in enumerate(ax.ravel()):
        if i >= n:
            a.axis("off")
            continue
        C = res[i]["_C"][np.ix_(FIFTHS_ORDER, FIFTHS_ORDER)].copy()
        np.fill_diagonal(C, np.nan)
        a.imshow(C, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        a.set_xticks(range(12)); a.set_yticks(range(12))
        a.set_xticklabels([NAMES[j] for j in FIFTHS_ORDER], fontsize=6)
        a.set_yticklabels([NAMES[j] for j in FIFTHS_ORDER], fontsize=6)
        a.set_title(f"{runs[i]['label']}\nfifths {res[i]['pc_fifths']:+.2f}, chromatic {res[i]['pc_chrom']:+.2f}", fontsize=9)
    fig.suptitle("Pitch-class similarity in circle-of-fifths order, per checkpoint (diagonal hidden)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}", flush=True)


def plot_pca(res, runs, path):
    n = len(runs)
    ncols = 4
    nrows = -(-n // ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    ring = FIFTHS_ORDER + [FIFTHS_ORDER[0]]
    for i, a in enumerate(ax.ravel()):
        if i >= n:
            a.axis("off")
            continue
        xy = res[i]["_xy"]
        a.plot(xy[ring, 0], xy[ring, 1], "--", color="grey", lw=0.8)
        a.scatter(xy[:, 0], xy[:, 1], c=[plt.cm.hsv(j / 12) for j in range(12)], s=40, zorder=3)
        for j in range(12):
            a.annotate(NAMES[j], xy[j], xytext=(3, 3), textcoords="offset points", fontsize=7)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(runs[i]["label"], fontsize=9)
    fig.suptitle("First two principal components of the pitch-class vectors, per checkpoint "
                 "(dashed: circle of fifths; orientation is arbitrary)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="theory_ckpt")
    ap.add_argument("--maestro_dir", default="data")
    ap.add_argument("--with_keys", action="store_true", help="also run the model on MAESTRO validation (key and error tests)")
    ap.add_argument("--min_corr", type=float, default=0.7)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()
    runs = json.load(open(args.manifest))
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from music_common import build_model, load_or_build_maestro_cache, load_weights
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    songs_cache = {}

    def get_songs(canon):
        if canon not in songs_cache:
            if canon:
                os.environ["CANON_CHORDS"] = "1"
            else:
                os.environ.pop("CANON_CHORDS", None)
            songs_cache[canon] = load_or_build_maestro_cache(args.maestro_dir, workers=4)
        return songs_cache[canon]

    res = []
    for r in runs:
        ck = torch.load(r["path"], map_location="cpu", weights_only=True)
        model = build_model(ck["model_cfg"], dropout=0.0).to(dev)
        load_weights(model, ck["model_state_dict"])
        model.eval()
        with torch.no_grad():
            E = model.embed
            X_pc = E.pitch_class.weight.float().cpu().numpy()
            W = E.weight()[:128].float().cpu().numpy()
        pitches = np.arange(21, 109)
        X_oct = np.stack([W[pitches[pitches % 12 == c]].mean(0) for c in range(12)])
        row = embedding_metrics(X_pc, X_oct, args.n_perm, args.seed)
        if args.with_keys:
            row.update(prediction_metrics(collect_model_stats(model, get_songs(bool(r["canon"])), dev, args.min_corr)))
        res.append(row)
        print(f"{r['label']:>22s} | fifths {row['pc_fifths']:+.2f} (p {row['pc_p_fifths']:.3f}) | "
              f"chromatic {row['pc_chrom']:+.2f} (p {row['pc_p_chrom']:.3f})"
              + (f" | r(K-K) major {row['r_kk_major']:.2f} minor {row['r_kk_minor']:.2f} | acc {row['pitch_acc']:.1f}%"
                 if args.with_keys else ""), flush=True)
        del model, ck
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "theory_over_checkpoints.csv"), "w", newline="") as f:
        cols = ["label"] + [k for k in res[0] if not k.startswith("_")]
        w = csv.writer(f)
        w.writerow(cols)
        for r, row in zip(runs, res):
            w.writerow([r["label"]] + [row[c] for c in cols[1:]])
    plot_lines(res, runs, args.with_keys, os.path.join(args.out_dir, "theory_over_checkpoints.png"))
    plot_matrices(res, runs, os.path.join(args.out_dir, "theory_ckpt_matrices.png"))
    plot_pca(res, runs, os.path.join(args.out_dir, "theory_ckpt_pca.png"))


if __name__ == "__main__":
    main()
