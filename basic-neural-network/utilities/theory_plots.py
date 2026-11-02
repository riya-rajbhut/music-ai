"""
theory_plots.py  --  compare what the trained model has learned with music theory.

Three analyses (choose with --what; default all):

 embeddings : Do the learned pitch-CLASS vectors reflect the circle of fifths?
              Uses the 12 pitch-class vectors (C, C#, ... B) in two versions:
                (a) the factored embedding's pitch-class component,
                (b) the full pitch vector averaged over octaves.
              Figure: cosine-similarity matrix in chromatic order and in circle-of-fifths order, a 2-D PCA map with the
              fifths circle drawn on it, similarity against interval distance, and the 12-point Fourier spectrum
              (component 5 = fifths, 1 = semitones, 3 = minor thirds, 4 = major thirds ...) against a permutation null.
 keys       : Does the model's next-pitch-class distribution, expressed relative to the tonic of the key estimated from
              the FIRST half of each window (Krumhansl-Schmuckler), look like the Krumhansl-Kessler tonal hierarchy?
 errors     : When the model is wrong, what is the pitch-class interval between its guess and the true note, and how far
              is it on the circle of fifths, compared with what independence would give?

Usage (from the folder that holds music_common.py):
  python theory_plots.py --canon --ckpt /kaggle/working/artifacts/best_maestro_canon.pt --out_dir theory_out
  python theory_plots.py --canon --ckpt ... --what embeddings          # no data needed, runs in seconds

Notes
 * Use --canon only for checkpoints trained with CANON_CHORDS=1.
 * A pattern is evidence only if it beats the permutation null printed next to it. A missing pattern is a result too.
 * Krumhansl-Kessler values are listener ratings; the model is trained on note statistics, so a match shows it has
   learned tonality, not that it "knows" music theory.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FIFTHS_ORDER = [(7 * i) % 12 for i in range(12)]               # C G D A E B F# C# G# D# A# F
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
DEGREES = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]
IU = np.triu_indices(12, 1)


def fifths_dist(i, j):
    d = abs((7 * i) % 12 - (7 * j) % 12)
    return min(d, 12 - d)


def chrom_dist(i, j):
    d = abs(i - j) % 12
    return min(d, 12 - d)


D5 = np.array([[fifths_dist(i, j) for j in range(12)] for i in range(12)])
D1 = np.array([[chrom_dist(i, j) for j in range(12)] for i in range(12)])


# ------------------------------------------------------------------------------------------------ embeddings
def cos_matrix(Xc):
    Z = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    return Z @ Z.T


def spearman_vs(C, Dm):
    return spearmanr(C[IU], Dm[IU]).correlation


def fourier_shares(Xc):
    """share of the pitch-class variation carried by each Fourier component k = 1..6 (k and 12-k are merged)"""
    P = (np.abs(np.fft.fft(Xc, axis=0)) ** 2).mean(axis=1)
    tot = P[1:].sum()
    return np.array([(P[k] + P[12 - k]) if k < 6 else P[6] for k in range(1, 7)]) / tot


def embedding_stats(X, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(0, keepdims=True)
    C = cos_matrix(Xc)
    out = dict(C=C, Xc=Xc, shares=fourier_shares(Xc), rho5=spearman_vs(C, D5), rho1=spearman_vs(C, D1))
    null_sh, null5, null1 = [], [], []
    for _ in range(n_perm):
        perm = rng.permutation(12)
        Xp = Xc[perm]
        null_sh.append(fourier_shares(Xp))
        Cp = C[np.ix_(perm, perm)]
        null5.append(spearman_vs(Cp, D5)); null1.append(spearman_vs(Cp, D1))
    out["null_shares"] = np.array(null_sh)
    out["p_share"] = (out["null_shares"] >= out["shares"][None]).mean(0)                 # one-sided, per component
    out["p_rho5"] = (np.abs(np.array(null5)) >= abs(out["rho5"])).mean()
    out["p_rho1"] = (np.abs(np.array(null1)) >= abs(out["rho1"])).mean()
    return out


def profile_by_distance(C, Dm):
    return np.array([C[(Dm == d) & (np.arange(12)[:, None] < np.arange(12)[None])].mean() for d in range(1, 7)])


def embedding_figure(X, title, path, n_perm=2000, seed=0):
    s = embedding_stats(X, n_perm, seed)
    C, Xc = s["C"], s["Xc"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    for a, order, name in ((ax[0, 0], list(range(12)), "chromatic order"), (ax[0, 1], FIFTHS_ORDER, "circle-of-fifths order")):
        im = a.imshow(C[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_xticks(range(12)); a.set_xticklabels([NAMES[i] for i in order], fontsize=8)
        a.set_yticks(range(12)); a.set_yticklabels([NAMES[i] for i in order], fontsize=8)
        a.set_title(f"Cosine similarity of pitch-class vectors\n({name})")
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.03)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    xy = U[:, :2] * S[:2]
    var = (S ** 2 / (S ** 2).sum())[:2] * 100
    a = ax[0, 2]
    ring = FIFTHS_ORDER + [FIFTHS_ORDER[0]]
    a.plot(xy[ring, 0], xy[ring, 1], "--", color="grey", lw=1, label="circle of fifths")
    a.scatter(xy[:, 0], xy[:, 1], c=[plt.cm.hsv(i / 12) for i in range(12)], s=90, zorder=3)
    for i in range(12):
        a.annotate(NAMES[i], xy[i], xytext=(5, 5), textcoords="offset points")
    a.set_title(f"First two principal components ({var[0]:.0f}% + {var[1]:.0f}% of variance)")
    a.legend(loc="best", fontsize=8)
    a = ax[1, 0]
    ks = np.arange(1, 7)
    lab = ["1\nsemitone", "2\nwhole tone", "3\nminor 3rd", "4\nmajor 3rd", "5\nfifth", "6\ntritone"]
    a.bar(ks, s["shares"] * 100, color="tab:blue", alpha=0.85, label="learned vectors")
    lo, hi = np.percentile(s["null_shares"], [2.5, 97.5], axis=0) * 100
    a.vlines(ks, lo, hi, color="black", lw=2, label="permutation null (95%)")
    a.set_xticks(ks); a.set_xticklabels(lab, fontsize=8)
    a.set_ylabel("share of pitch-class variation (%)")
    a.set_title("Fourier spectrum over the 12 pitch classes")
    a.legend(fontsize=8)
    a = ax[1, 1]
    d = np.arange(1, 7)
    a.plot(d, profile_by_distance(C, D5), "o-", label="distance on the circle of fifths")
    a.plot(d, profile_by_distance(C, D1), "s-", label="distance in semitones")
    a.axhline(0, color="grey", lw=0.5)
    a.set_xlabel("interval distance between two pitch classes"); a.set_ylabel("mean cosine similarity")
    a.set_title("Similarity against distance"); a.legend(fontsize=8)
    a = ax[1, 2]
    a.axis("off")
    txt = (f"Similarity vs fifths distance (Spearman):  {s['rho5']:+.2f}  p = {s['p_rho5']:.3f}\n"
           f"Similarity vs semitone distance:           {s['rho1']:+.2f}  p = {s['p_rho1']:.3f}\n\n"
           f"Fourier share, fifths component (k=5):     {s['shares'][4] * 100:.1f}%  p = {s['p_share'][4]:.3f}\n"
           f"Fourier share, semitone component (k=1):   {s['shares'][0] * 100:.1f}%  p = {s['p_share'][0]:.3f}\n\n"
           "Negative Spearman: nearby classes are more\nsimilar. Small p: unlikely under shuffled labels.")
    a.text(0.0, 0.95, txt, va="top", fontsize=9, family="monospace")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}\n  {txt.splitlines()[0]}\n  {txt.splitlines()[1]}\n  {txt.splitlines()[3]}\n  {txt.splitlines()[4]}", flush=True)
    return s


# ------------------------------------------------------------------------------------------------ keys
def kk_profiles():
    prof = np.stack([np.roll(KK_MAJOR, t) for t in range(12)] + [np.roll(KK_MINOR, t) for t in range(12)])
    return (prof - prof.mean(1, keepdims=True)) / prof.std(1, keepdims=True)


def estimate_keys(hist):
    """Krumhansl-Schmuckler: hist (B,12) pitch-class counts -> tonic (B,), mode (B,) 0=major 1=minor, correlation (B,)"""
    h = hist.astype(np.float64)
    sd = h.std(1, keepdims=True)
    z = (h - h.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    corr = z @ kk_profiles().T / 12.0
    best = corr.argmax(1)
    return best % 12, best // 12, corr.max(1)


def keys_figure(st, path):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for m, (a, nm, kk) in enumerate(zip(ax, ("major", "minor"), (KK_MAJOR, KK_MINOR))):
        pred = st["rel_pred"][m] / max(st["rel_notes"][m], 1)
        true = st["rel_true"][m] / max(st["rel_true"][m].sum(), 1)
        kkn = kk / kk.sum()
        x = np.arange(12)
        a.bar(x - 0.2, true, 0.4, color="lightgrey", label="true next note")
        a.bar(x + 0.2, pred, 0.4, color="tab:blue", label="model's predicted distribution")
        a.plot(x, kkn, "o-", color="tab:red", label="Krumhansl-Kessler profile (normalised)")
        a.set_xticks(x); a.set_xticklabels(DEGREES)
        a.set_xlabel("pitch class relative to the estimated tonic (semitones above tonic)")
        r_mk = np.corrcoef(pred, kkn)[0, 1]; r_tk = np.corrcoef(true, kkn)[0, 1]; r_mt = np.corrcoef(pred, true)[0, 1]
        a.set_title(f"{nm.capitalize()} keys ({st['rel_windows'][m]} windows)\n"
                    f"r(model, K-K) = {r_mk:.2f} | r(true, K-K) = {r_tk:.2f} | r(model, true) = {r_mt:.2f}")
        a.legend(fontsize=8)
        print(f"  {nm}: {st['rel_windows'][m]} windows | r(model,K-K) {r_mk:.2f} | r(true,K-K) {r_tk:.2f} | r(model,true) {r_mt:.2f}", flush=True)
    fig.suptitle("Next-pitch-class distribution relative to the key estimated from the first half of each window", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ------------------------------------------------------------------------------------------------ errors
def d5_of_interval(i):
    return fifths_dist(i, 0)


def error_stats(counts_int, t_pc, p_pc):
    n = counts_int.sum()
    obs = counts_int / n
    t, p = t_pc / t_pc.sum(), p_pc / p_pc.sum()
    exp = np.array([sum(t[c] * p[(c + i) % 12] for c in range(12)) for i in range(12)])      # independent guess and truth
    return obs, exp


def errors_figure(st, path):
    obs, exp = error_stats(st["err_int"], st["err_true_pc"], st["err_pred_pc"])
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(12)
    names = ["0\nsame class", "1\nsemitone", "2\nwhole tone", "3\nminor 3rd", "4\nmajor 3rd", "5\nfourth", "6\ntritone",
             "7\nfifth", "8\nminor 6th", "9\nmajor 6th", "10\nminor 7th", "11\nmajor 7th"]
    ax[0].bar(x - 0.2, obs * 100, 0.4, color="tab:blue", label="observed")
    ax[0].bar(x + 0.2, exp * 100, 0.4, color="lightgrey", label="expected if the guess were unrelated to the truth")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8)
    ax[0].set_ylabel("share of wrong predictions (%)")
    ax[0].set_title("Pitch-class interval (guess - truth, mod 12) among wrong predictions")
    ax[0].legend(fontsize=8)
    d = np.arange(0, 7)
    o5 = np.array([sum(obs[i] for i in range(12) if d5_of_interval(i) == k) for k in d])
    e5 = np.array([sum(exp[i] for i in range(12) if d5_of_interval(i) == k) for k in d])
    ax[1].bar(d - 0.2, o5 * 100, 0.4, color="tab:blue", label="observed")
    ax[1].bar(d + 0.2, e5 * 100, 0.4, color="lightgrey", label="expected")
    ax[1].set_xlabel("distance on the circle of fifths (1 = a fourth or fifth, 5 = a semitone)")
    ax[1].set_ylabel("share of wrong predictions (%)")
    ax[1].set_title("The same errors grouped by distance on the circle of fifths")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Structure of the model's errors ({int(st['err_int'].sum())} wrong of {int(st['n_notes'])} notes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    top = np.argsort(-obs)[:4]
    print(f"saved {path}\n  most common error intervals: " +
          ", ".join(f"{i} ({obs[i] * 100:.1f}% vs {exp[i] * 100:.1f}% expected)" for i in top) +
          f"\n  fifths-distance 1 (fourth or fifth): {o5[1] * 100:.1f}% vs {e5[1] * 100:.1f}% expected", flush=True)


# ------------------------------------------------------------------------------------------------ model run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--canon", action="store_true")
    ap.add_argument("--what", choices=["embeddings", "keys", "errors", "all"], default="all")
    ap.add_argument("--out_dir", default="theory_out")
    ap.add_argument("--maestro_dir", default="data")
    ap.add_argument("--min_corr", type=float, default=0.7, help="keep windows whose key estimate correlates at least this well")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()
    if args.canon:
        os.environ["CANON_CHORDS"] = "1"
    import torch
    import torch.utils.data as data
    from torch.amp import autocast
    from music_common import (WindowDataset, build_model, concat_songs, load_or_build_maestro_cache, load_weights)

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = build_model(ck["model_cfg"], dropout=0.0).to(dev)
    load_weights(model, ck["model_state_dict"])
    model.eval()
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    print(f"loaded {args.ckpt} | meta {ck.get('meta')} | tokenisation: "
          f"{'canonical chord order' if args.canon else 'original order'}", flush=True)

    if args.what in ("embeddings", "all"):
        with torch.no_grad():
            E = model.embed
            X_pc = E.pitch_class.weight.float().cpu().numpy()                       # (12, D) pitch-class component
            W = E.weight()[:128].float().cpu().numpy()                              # (128, D) full pitch vectors
        pitches = np.arange(21, 109)
        X_oct = np.stack([W[pitches[pitches % 12 == c]].mean(0) for c in range(12)])
        embedding_figure(X_pc, f"Pitch-class component of the factored embedding ({tag})",
                         os.path.join(args.out_dir, "theory_embeddings_pcrow.png"), args.n_perm, args.seed)
        embedding_figure(X_oct, f"Full pitch vectors averaged over octaves ({tag})",
                         os.path.join(args.out_dir, "theory_embeddings_octavemean.png"), args.n_perm, args.seed)
    if args.what == "embeddings":
        return

    songs = load_or_build_maestro_cache(args.maestro_dir, workers=4)
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
            probs = torch.softmax(logits[:, 0::3, :128].float()[:, N0:], -1)                  # (B,128,128)
            tgt = y[:, 0::3][:, N0:]
            pred = probs.argmax(-1)
            B, N = tgt.shape
            wrong = pred != tgt
            st["n_notes"] += tgt.numel()
            ii = ((pred % 12 - tgt % 12) % 12)[wrong].cpu().numpy()
            st["err_int"] += np.bincount(ii, minlength=12)
            st["err_true_pc"] += np.bincount((tgt % 12)[wrong].cpu().numpy(), minlength=12)
            st["err_pred_pc"] += np.bincount((pred % 12)[wrong].cpu().numpy(), minlength=12)
            ctx = (x[:, 1::3][:, :N0] % 12).cpu().numpy()                                      # first half only
            hist = np.stack([np.bincount(r, minlength=12) for r in ctx])
            tonic, mode, corr = estimate_keys(hist)
            keep = corr >= args.min_corr
            if keep.any():
                pc = probs @ M                                                                  # (B,N,12)
                tn = torch.as_tensor(tonic, device=dev)
                idx = (torch.arange(12, device=dev)[None] + tn[:, None]) % 12
                rel = pc.gather(2, idx[:, None, :].expand(B, N, 12)).sum(1).cpu().numpy()      # (B,12) summed over notes
                true_rel = ((tgt % 12 - tn[:, None]) % 12).cpu().numpy()
                for m in (0, 1):
                    sel = keep & (mode == m)
                    if sel.any():
                        st["rel_pred"][m] += rel[sel].sum(0)
                        st["rel_notes"][m] += sel.sum() * N
                        st["rel_windows"][m] += int(sel.sum())
                        st["rel_true"][m] += np.bincount(true_rel[sel].reshape(-1), minlength=12)
    if args.what in ("keys", "all"):
        keys_figure(st, os.path.join(args.out_dir, "theory_keys.png"))
    if args.what in ("errors", "all"):
        errors_figure(st, os.path.join(args.out_dir, "theory_errors.png"))


if __name__ == "__main__":
    main()
