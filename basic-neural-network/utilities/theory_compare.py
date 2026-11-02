"""
theory_compare.py -- cross-architecture music-theory tests on the MAESTRO models of the music-rnn-ddp project.

Models (key -> training script -> W&B artifact you gave):
  lstm    train_ddp_v4_optimized.py                              run-ksl0xx7l-events:v3
  gru     train_ddp_v6.py                                         run-1k68of2n-events:v3
  tf6     train_ddp_v7_transformer.py   (pitch-only)              run-p4bpo17f-events:v3
  remi12  train_ddp_v7_transformer_REMI.py                        rei_music_transformer_best:v37
  remi6   same script, 6-layer run (pass --ckpt remi6=PATH)       (no artifact given)
  arcface train_ddp_v8_transformer_factorized_arcface.py          factorized_harmonic_transformer_best:v14
  v9      train_ddp_v9_transformer_REMI_2drope_cpc_factored.py    rope_2d_music_transformer_best:v41

IMPORTANT: artifacts called run-<id>-events are W&B's automatic metric-history logs, not checkpoints. The v4, v6 and
v7-pitch-only scripts only torch.save() to artifacts/best_model.pt on the Kaggle disk and never upload it, so for
lstm / gru / tf6 you probably need the .pt from your Kaggle notebook output (--ckpt lstm=PATH ...).

The script IMPORTS your training scripts (all are guarded by `if __name__ == "__main__"`, so nothing trains), builds each
model with your own class, infers sizes from the checkpoint, and evaluates on that model's OWN held-out test split, built
by your own data pipeline and seed 53. Nothing in the pipeline is re-implemented.

Tests per model
  embeddings : output-layer pitch vectors (rows of the 128-way pitch output matrix) averaged over octaves (piano range
               21..108) -> 12 pitch-class vectors -> cosine matrix, structure strength for the circle of fifths and the
               chromatic circle (minus Spearman correlation with interval distance, permutation p-value), Fourier shares
               (k=5 fifths, k=1 semitone), PCA map. Also the INPUT-side pitch-class representation where one exists.
  keys       : Krumhansl-Schmuckler key estimated from the context only; the model's next-pitch-class distribution relative
               to that tonic is correlated with the Krumhansl-Kessler profile (major and minor).
  errors     : pitch-class interval between wrong guesses and the truth, against what unrelated guesses would give.
  --with_random : also measures the embeddings of an UNTRAINED copy of every architecture, so you can say how much of the
               structure is learned and how much is built in by the architecture.

Context used for the key estimate:  lstm / gru: the 128 notes before the predicted note.
                                    transformers: the first 128 notes of the window; notes 129-256 are scored.

Usage (run from the folder that holds data/maestro-v3.0.0, e.g. your Kaggle working dir):
  python theory_compare.py --src_dir /path/to/scripts --models all --download --out_dir theory_compare_out
  python theory_compare.py --src_dir ... --ckpt lstm=/p/best_model.pt --ckpt v9=/p/best.pt --models lstm,v9 --with_random
  python theory_compare.py --src_dir ... --models all --random_only      # smoke test: no checkpoint, no data, seconds

  For --download put the key in the environment (export WANDB_API_KEY=...). The key printed in the training scripts is
  readable by anyone who sees them: rotate it in W&B settings and read the new one from the environment.

Outputs: theory_compare.csv, theory_compare_summary.png, theory_compare_matrices.png, theory_compare_pca.png,
         theory_compare_keys.png, theory_compare_errors.png, and one embedding figure per model.
"""
import argparse
import csv
import glob
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPTS = {
    "lstm": "train_ddp_v4_optimized.py",
    "gru": "train_ddp_v6_GRU.py",
    "tf6": "train_ddp_v7_transformer.py",
    "remi12": "train_ddp_v7_transformer_REMI.py",
    "remi6": "train_ddp_v7_transformer_REMI_v2.py",
    "arcface": "train_ddp_v8_transformer_factorized_arcface.py",
    "v9": "train_ddp_v9_transformer_REMI_2drope_cpc_factored.py",
}
ARTIFACTS = {
    "lstm": "riya-rajbhut-student/music-rnn-ddp/best_model:v39",  #https://forge.coreweave.com/wandb/riya-rajbhut-student/music-rnn-ddp/runs/bfv08du2/overview?nw=nwuserriyarajbhut
    "gru": "riya-rajbhut-student/music-rnn-ddp/best_model:v161", #https://forge.coreweave.com/wandb/riya-rajbhut-student/music-rnn-ddp/runs/93zaxkty/overview?nw=nwuserriyarajbhut
    "tf6": "riya-rajbhut-student/music-rnn-ddp/best_model:v135", #https://forge.coreweave.com/wandb/riya-rajbhut-student/music-rnn-ddp/runs/tokip0b4/overview?nw=nwuserriyarajbhut
    "remi12": "riya-rajbhut-student/music-rnn-ddp/rei_music_transformer_best:v57", #https://forge.coreweave.com/wandb/riya-rajbhut-student/music-rnn-ddp/runs/pbsphwv8/overview?nw=nwuserriyarajbhut
    "remi6": "riya-rajbhut-student/music-rnn-ddp/best_model:v190", #https://forge.coreweave.com/wandb/riya-rajbhut-student/music-rnn-ddp/runs/w3a86ses/overview?nw=nwuserriyarajbhut
    "arcface": "riya-rajbhut-student/music-rnn-ddp/factorized_harmonic_transformer_best:v14", #https://forge.coreweave.com/wandb/riya-rajbhut-student/music-rnn-ddp/runs/vgopk612?nw=nwuserriyarajbhut
    "v9": "riya-rajbhut-student/music-rnn-ddp/rope_2d_music_transformer_best:v46", #https://forge.coreweave.com/wandb/riya-rajbhut-student/music-rnn-ddp/runs/vxtdzv99/overview?nw=nwuserriyarajbhut
}
LABELS = {
    "lstm": "LSTM {layers}L (pitch only)",
    "gru": "GRU {layers}L (pitch + timing)",
    "tf6": "Transformer {layers}L (pitch only)",
    "remi12": "REMI Transformer {layers}L",
    "remi6": "REMI Transformer {layers}L",
    "arcface": "REMI + RoPE + ArcFace",
    "v9": "REMI + 2D-RoPE + contrastive (v9)",
}
# sizes from the hyper-parameters in your scripts; used only for the untrained copies (--with_random / --random_only)
DEFAULTS = {
    "lstm": dict(hidden=512, layers=3),
    "gru": dict(hidden=512, layers=3),
    "tf6": dict(hidden=768, layers=6, seq=256),
    "remi12": dict(hidden=768, layers=12, heads=12, seq=768),
    "remi6": dict(hidden=768, layers=6, heads=8, seq=768),
    "arcface": dict(embed=768, layers=12, heads=12),
    "v9": dict(embed=768, layers=12, heads=12, factored=True),
}
N0 = 128                      # notes of context used for the key estimate (transformers: first half of the window)


def _tp():
    import theory_plots as tp         # your theory_plots.py, must sit in --src_dir
    return tp


# ------------------------------------------------------------------------------------------ loading
def load_module(src_dir, key):
    path = os.path.join(src_dir, SCRIPTS[key])
    if not os.path.exists(path):                       # allow hash-prefixed names such as 848a1dfc-train_ddp_v4_optimized.py
        hits = glob.glob(os.path.join(src_dir, "*" + SCRIPTS[key]))
        if not hits:
            raise FileNotFoundError(path)
        path = hits[0]
    spec = importlib.util.spec_from_file_location(f"mod_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_state_dict(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    return {k.replace("module.", "", 1).replace("_orig_mod.", ""): v for k, v in sd.items()}


def find_ckpt(key, ckpt_args, download, root="wandb_artifacts"):
    if key in ckpt_args:
        return ckpt_args[key]
    if not download or ARTIFACTS.get(key) is None:
        raise SystemExit(f"no checkpoint for '{key}': pass --ckpt {key}=PATH" + ("" if ARTIFACTS.get(key) is None else " or use --download"))
    import wandb
    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])            # key from the environment only
    d = api.artifact(ARTIFACTS[key]).download(os.path.join(root, key))
    pts = sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
    if not pts:
        raise SystemExit(f"{ARTIFACTS[key]} contains no .pt file (files: {os.listdir(d)}). That script logs only metrics: "
                         f"pass --ckpt {key}=PATH from your Kaggle output.")
    print(f"{key}: using {pts[-1]}")
    return pts[-1]


# ------------------------------------------------------------------------------------------ model building
def n_layers(sd, prefix):
    """number of numbered sub-modules under `prefix` (e.g. 'blocks.' or 'transformer.layers.')"""
    return len({int(k[len(prefix):].split(".")[0]) for k in sd if k.startswith(prefix)})


def infer_cfg(key, sd, heads=None):
    if key in ("lstm", "gru"):
        return dict(hidden=sd["fusion_layer.weight"].shape[1] // 2,
                    layers=len([k for k in sd if k.startswith(("lstm.weight_ih_l", "gru.weight_ih_l"))]))
    if key == "tf6":
        return dict(hidden=sd["fusion_layer.weight"].shape[1] // 2, layers=n_layers(sd, "transformer.layers."),
                    seq=sd["pos_embed.weight"].shape[0])
    if key in ("remi12", "remi6"):
        L = n_layers(sd, "transformer.layers.")
        return dict(hidden=sd["token_embed.weight"].shape[1], layers=L, seq=sd["pos_embed.weight"].shape[0],
                    heads=heads or (12 if L == 12 else 8))
    if key == "arcface":
        return dict(embed=sd["token_embedding.weight"].shape[1], layers=n_layers(sd, "blocks."), heads=heads or 12)
    if key == "v9":
        fac = "token_embedding.pitch_class.weight" in sd
        e = sd["token_embedding.pitch_class.weight"] if fac else sd["token_embedding.weight"]
        return dict(embed=e.shape[1], layers=n_layers(sd, "blocks."), heads=heads or 12, factored=fac)
    raise KeyError(key)


def construct(key, mod, cfg):
    if key in ("lstm", "gru"):
        return mod.OptimizedMusicRNN(hidden_size=cfg["hidden"], num_layers=cfg["layers"], dropout_rate=0.0)
    if key == "tf6":
        return mod.OptimizedMusicTransformer(hidden_size=cfg["hidden"], num_layers=cfg["layers"], seq_len=cfg["seq"],
                                             dropout_rate=0.0)
    if key in ("remi12", "remi6"):
        return mod.OptimizedMusicTransformer(num_tokens=384, hidden_size=cfg["hidden"], num_layers=cfg["layers"],
                                             num_heads=cfg["heads"], seq_len=cfg["seq"], dropout_rate=0.0)
    if key == "arcface":
        return mod.FactorizedHarmonic3DMusicTransformer(num_tokens=384, embed_dim=cfg["embed"], num_heads=cfg["heads"],
                                                        num_layers=cfg["layers"], dropout_rate=0.0)
    if key == "v9":
        return mod.Harmonic2DMusicTransformer(num_tokens=384, embed_dim=cfg["embed"], num_heads=cfg["heads"],
                                              num_layers=cfg["layers"], dropout_rate=0.0, cpc_ks=(),
                                              factored_pitch_embed=cfg["factored"])
    raise KeyError(key)


def load_into(key, m, sd):
    if key == "v9":
        sd = {k: v for k, v in sd.items() if not k.startswith("cpc.")}      # contrastive head is training-only
    try:
        m.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise SystemExit(f"checkpoint does not match the '{key}' architecture built from {SCRIPTS[key]}:\n{e}")


def label_for(key, cfg, suffix=""):
    return LABELS[key].format(**cfg) + suffix


# ------------------------------------------------------------------------------------------ pitch vectors
def octave_mean(W):
    p = np.arange(21, 109)
    return np.stack([W[p[p % 12 == c]].mean(0) for c in range(12)])


def pitch_vectors(key, m):
    """dict: out = (128, D) output-layer pitch vectors; inp = (12, d) input-side pitch-class vectors or None; inp_desc"""
    import torch
    f = lambda t: t.detach().float().cpu().numpy()
    with torch.no_grad():
        if key in ("lstm", "gru", "tf6"):
            return dict(out=f(m.fusion_layer.weight), inp=f(m.pitch_class_embed.weight),
                        inp_desc="input pitch-class embedding")
        if key in ("remi12", "remi6"):
            return dict(out=f(m.output_head.weight[:128]), inp=octave_mean(f(m.token_embed.weight[:128])),
                        inp_desc="input token embedding, octave-averaged")
        if key == "arcface":
            W = torch.nn.functional.normalize(m.pitch_head.weight.float(), dim=1)
            return dict(out=f(W), inp=octave_mean(f(m.token_embedding.weight[:128])),
                        inp_desc="input token embedding, octave-averaged")
        if key == "v9":
            if getattr(m, "factored", False):
                E = m.token_embedding
                return dict(out=f(E.full_weight()[:128]), inp=f(E.pitch_class.weight),
                            inp_desc="pitch-class component of the factored embedding")
            return dict(out=f(m.token_embedding.weight[:128]), inp=None, inp_desc="")
    raise KeyError(key)


# ------------------------------------------------------------------------------------------ data / inference
def make_dataset(key, mod, data_root):
    """returns (dataset used for sampling, number of test windows with the TRAINING hop, True if the cache was rebuilt)"""
    import pathlib
    root = pathlib.Path(data_root)
    if not root.exists():
        mod.download_maestro_dataset(str(root.parent))
    before = set(glob.glob(str(root / "converted_notes_*.pkl")))
    if key == "arcface":
        songs = mod.load_or_create_note_cache(root, True)
        _, _, test = mod.get_official_split_songs(root, songs)
        ds = mod.RemiTokenDataset(test, seq_len=768, hop_length=768, augment=False)
        ref = len(mod.RemiTokenDataset(test, seq_len=768, hop_length=384, augment=False))
    else:
        notes = mod.load_or_create_note_cache(root, True, None)
        train, _, test = mod.split_song_arrays(notes, seed=53)
        if key == "lstm":
            ds = mod.BasicRNNForMusic(test, seq_len=128, augment=False, hop_length=64)
            ref = len(mod.BasicRNNForMusic(test, seq_len=128, augment=False, hop_length=3))
        elif key == "gru":
            smax, dmax = mod.compute_time_clip_bounds(train, 99.5)
            train = mod.clip_extreme_time_values(train, smax, dmax)
            test = mod.clip_extreme_time_values(test, smax, dmax)
            mean, std = mod.compute_time_feature_stats(train)
            ds = mod.BasicRNNForMusic(test, seq_len=128, time_feature_mean=mean, time_feature_std=std, augment=False)
            ref = len(ds)
        elif key == "tf6":
            ds = mod.BasicRNNForMusic(test, seq_len=256, augment=False, hop_length=256)
            ref = len(ds)
        elif key in ("remi12", "remi6"):
            ds = mod.BasicRNNForMusic(test, seq_len=768, hop_length=768, augment=False)
            # remi12 trained/evaluated with hop 384; the 6-layer script (train_ddp_v7_transformer_REMI_v2.py) uses hop = seq_len
            ref = len(mod.BasicRNNForMusic(test, seq_len=768, hop_length=768 if key == "remi6" else 384, augment=False))
        else:                                                                         # v9
            ds = mod.RemiTokenDataset(test, seq_len=768, hop_length=768, augment=False)
            ref = len(mod.RemiTokenDataset(test, seq_len=768, hop_length=384, augment=False))
    rebuilt = bool(set(glob.glob(str(root / "converted_notes_*.pkl"))) - before) and key != "arcface"   # arcface uses the official CSV split: order-proof
    return ds, ref, rebuilt


def batches(key, m, ds, dev, max_windows, bs, seed=0):
    """yields ctx (B,K) pitch classes of the context, probs (B,N,128) over pitches, tgt (B,N) true pitches"""
    import torch
    import torch.utils.data as data
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:max_windows]
    loader = data.DataLoader(data.Subset(ds, idx.tolist()), batch_size=bs, num_workers=0)
    with torch.no_grad():
        for b in loader:
            if key == "lstm":
                x, y = b
                x, y = x.to(dev), y.to(dev)
                logits = m(x)["pitch"]                                                  # (B,128)
                yield (x % 12).cpu().numpy(), torch.softmax(logits.float(), -1)[:, None], y[:, None]
            elif key == "gru":
                (x, t), (y, _) = b
                x, t, y = x.to(dev), t.to(dev), y.to(dev)
                logits = m(x, t)["pitch"]
                yield (x % 12).cpu().numpy(), torch.softmax(logits.float(), -1)[:, None], y[:, None]
            elif key == "tf6":
                x, y = b
                x, y = x.to(dev), y.to(dev)
                logits = m(x)["pitch"]                                                  # (B,T,128)
                h = x.shape[1] // 2
                yield (x[:, :h] % 12).cpu().numpy(), torch.softmax(logits[:, h:].float(), -1), y[:, h:]
            else:                                                                        # REMI family: 3 tokens per note
                if key in ("remi12", "remi6"):
                    x, y = b
                    x, y = x.to(dev), y.to(dev)
                    logits = m(x)["pitch"]
                elif key == "v9":
                    x, y, a = b
                    x, y, a = x.to(dev), y.to(dev), a.to(dev)
                    logits = m(x, a)["pitch"]
                else:                                                                    # arcface
                    x, y, a, s = b
                    x, y, a, s = x.to(dev), y.to(dev), a.to(dev), s.to(dev)
                    logits = m(x, a, start_indices=s)["logits"]
                probs = torch.softmax(logits[:, 0::3, :128].float(), -1)[:, N0:]         # grammar: pitch only
                tgt = y[:, 0::3][:, N0:]
                ctx = (x[:, 1::3][:, :N0] % 12).cpu().numpy()
                yield ctx, probs, tgt


def new_stats():
    return dict(rel_pred=np.zeros((2, 12)), rel_true=np.zeros((2, 12)), rel_notes=np.zeros(2), rel_windows=np.zeros(2, int),
                err_int=np.zeros(12), err_true_pc=np.zeros(12), err_pred_pc=np.zeros(12), n_notes=0, n_correct=0)


def accumulate(st, ctx, probs, tgt, min_corr):
    import torch
    tp = _tp()
    B, N, _ = probs.shape
    pred = probs.argmax(-1)
    ok = pred == tgt
    wrong = ~ok
    st["n_notes"] += tgt.numel()
    st["n_correct"] += int(ok.sum())
    st["err_int"] += np.bincount(((pred % 12 - tgt % 12) % 12)[wrong].cpu().numpy(), minlength=12)
    st["err_true_pc"] += np.bincount((tgt % 12)[wrong].cpu().numpy(), minlength=12)
    st["err_pred_pc"] += np.bincount((pred % 12)[wrong].cpu().numpy(), minlength=12)
    hist = np.stack([np.bincount(r, minlength=12) for r in ctx])
    tonic, mode, corr = tp.estimate_keys(hist)
    keep = corr >= min_corr
    if not keep.any():
        return
    M = torch.zeros(128, 12, device=probs.device)
    M[torch.arange(128), torch.arange(128) % 12] = 1.0
    pc = probs @ M                                                                       # (B,N,12)
    tn = torch.as_tensor(tonic, device=probs.device)
    idx = (torch.arange(12, device=probs.device)[None] + tn[:, None]) % 12
    rel = pc.gather(2, idx[:, None, :].expand(B, N, 12)).sum(1).cpu().numpy()           # (B,12) summed over notes
    true_rel = ((tgt % 12 - tn[:, None]) % 12).cpu().numpy()
    for mo in (0, 1):
        sel = keep & (mode == mo)
        if sel.any():
            st["rel_pred"][mo] += rel[sel].sum(0)
            st["rel_notes"][mo] += sel.sum() * N
            st["rel_windows"][mo] += int(sel.sum())
            st["rel_true"][mo] += np.bincount(true_rel[sel].reshape(-1), minlength=12)


# ------------------------------------------------------------------------------------------ numpy-only analysis
def _r(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def embedding_row(X, tag, n_perm, seed=53):
    """structure strength = minus Spearman (higher = stronger), as in the checkpoint figures"""
    tp = _tp()
    s = tp.embedding_stats(X, n_perm, seed)
    U, S, _ = np.linalg.svd(s["Xc"], full_matrices=False)
    row = {f"{tag}_fifths": -s["rho5"], f"{tag}_chrom": -s["rho1"], f"{tag}_p_fifths": s["p_rho5"],
           f"{tag}_p_chrom": s["p_rho1"], f"{tag}_k5": s["shares"][4] * 100, f"{tag}_k1": s["shares"][0] * 100,
           f"{tag}_p_k5": s["p_share"][4], f"{tag}_p_k1": s["p_share"][0]}
    return row, s["C"], U[:, :2] * S[:2]


def key_row(st):
    tp = _tp()
    out = {}
    for mo, nm in enumerate(("major", "minor")):
        kk = (tp.KK_MAJOR, tp.KK_MINOR)[mo]
        kkn = kk / kk.sum()
        pred = st["rel_pred"][mo] / max(st["rel_notes"][mo], 1)
        true = st["rel_true"][mo] / max(st["rel_true"][mo].sum(), 1)
        out[f"key_r_model_kk_{nm}"] = _r(pred, kkn)
        out[f"key_r_true_kk_{nm}"] = _r(true, kkn)
        out[f"key_r_model_true_{nm}"] = _r(pred, true)
        out[f"key_windows_{nm}"] = int(st["rel_windows"][mo])
    obs, exp = tp.error_stats(st["err_int"], st["err_true_pc"], st["err_pred_pc"])
    out["err_minor3rd_obs"], out["err_minor3rd_exp"] = obs[3] * 100, exp[3] * 100
    out["err_tritone_obs"], out["err_tritone_exp"] = obs[6] * 100, exp[6] * 100
    out["err_fifths1_obs"] = sum(obs[i] for i in range(12) if tp.d5_of_interval(i) == 1) * 100
    out["err_fifths1_exp"] = sum(exp[i] for i in range(12) if tp.d5_of_interval(i) == 1) * 100
    out["pitch_acc_%"] = 100 * st["n_correct"] / max(st["n_notes"], 1)
    out["n_scored_notes"] = int(st["n_notes"])
    return out


# ------------------------------------------------------------------------------------------ plots
def _bars(ax, x, vals, ps, color, dx, untrained):
    for xi, v, p, u in zip(x, vals, ps, untrained):
        if v is None or not np.isfinite(v):
            continue
        sig = p is None or (np.isfinite(p) and p < 0.05)             # no p-value given -> solid bar
        ax.bar(xi + dx, v, 0.38, facecolor=color if sig else "white", edgecolor=color, linewidth=1.3,
               hatch="///" if u else None)


def _get(rows, k):
    return [r.get(k, float("nan")) for r in rows]


def summary_figure(rows, path):
    x = np.arange(len(rows))
    lab = [r["label"].replace(" (", "\n(") for r in rows]
    unt = [bool(r.get("untrained")) for r in rows]
    nones = [None] * len(rows)
    blue, orange, green, red, purple = "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))

    def panel(a, title, specs, ylabel=None, zero=True):
        for dx, (k, pk, color, name) in zip((-0.2, 0.2) if len(specs) == 2 else (-0.26, 0, 0.26), specs):
            _bars(a, x, _get(rows, k), _get(rows, pk) if pk else nones, color, dx, unt)
        a.legend(handles=[Patch(color=s[2], label=s[3]) for s in specs], fontsize=8)
        a.set_title(title, fontsize=10)
        if ylabel:
            a.set_ylabel(ylabel)
        if zero:
            a.axhline(0, color="grey", lw=0.6)

    panel(ax[0, 0], "Output-layer pitch vectors, octave-averaged",
          [("out_fifths", "out_p_fifths", blue, "circle of fifths"), ("out_chrom", "out_p_chrom", orange, "chromatic circle")],
          "structure strength (minus Spearman)")
    panel(ax[0, 1], "Input-side pitch-class representation",
          [("inp_fifths", "inp_p_fifths", blue, "circle of fifths"), ("inp_chrom", "inp_p_chrom", orange, "chromatic circle")],
          "structure strength (minus Spearman)")
    panel(ax[0, 2], "Fourier share, output-layer pitch classes (%)",
          [("out_k5", "out_p_k5", blue, "fifths component (k=5)"), ("out_k1", "out_p_k1", orange, "semitone component (k=1)")],
          zero=False)
    panel(ax[1, 0], "Key awareness: r(model's next-pitch-class distribution, Krumhansl-Kessler)",
          [("key_r_model_kk_major", None, blue, "major keys"), ("key_r_model_kk_minor", None, red, "minor keys")], zero=False)
    ax[1, 0].set_ylim(0, 1)
    for k, color, dx in (("err_minor3rd", green, -0.26), ("err_tritone", purple, 0), ("err_fifths1", orange, 0.26)):
        d = [o - e for o, e in zip(_get(rows, k + "_obs"), _get(rows, k + "_exp"))]
        _bars(ax[1, 1], x, d, nones, color, dx, unt)
    ax[1, 1].axhline(0, color="grey", lw=0.6)
    ax[1, 1].legend(handles=[Patch(color=green, label="minor third"), Patch(color=purple, label="tritone"),
                             Patch(color=orange, label="fourth / fifth")], fontsize=8)
    ax[1, 1].set_title("Error share above chance (observed minus expected, points)", fontsize=10)
    _bars(ax[1, 2], x, _get(rows, "pitch_acc_%"), nones, blue, 0, unt)
    ax[1, 2].set_title("Next-pitch accuracy (%) on the scored notes of each model's own test split", fontsize=10)
    for a in ax.ravel():
        a.set_xticks(x)
        a.set_xticklabels(lab, fontsize=7)
    fig.suptitle("Music-theory measurements per architecture (filled: p < 0.05 against shuffled labels; "
                 "hatched: untrained network of the same architecture)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path, flush=True)


def _grid(n, ncols=4, cell=4):
    nrows = -(-n // ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows), squeeze=False)
    for i, a in enumerate(ax.ravel()):
        if i >= n:
            a.axis("off")
    return fig, ax.ravel()


def plot_matrices(items, path):
    """items: (label, cosine matrix C, strength fifths, strength chromatic)"""
    tp = _tp()
    fig, ax = _grid(len(items))
    for a, (label, C, s5, s1) in zip(ax, items):
        M = C[np.ix_(tp.FIFTHS_ORDER, tp.FIFTHS_ORDER)].copy()
        np.fill_diagonal(M, np.nan)
        a.imshow(M, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        a.set_xticks(range(12))
        a.set_yticks(range(12))
        a.set_xticklabels([tp.NAMES[j] for j in tp.FIFTHS_ORDER], fontsize=6)
        a.set_yticklabels([tp.NAMES[j] for j in tp.FIFTHS_ORDER], fontsize=6)
        a.set_title(f"{label}\nfifths {s5:+.2f}, chromatic {s1:+.2f}", fontsize=9)
    fig.suptitle("Output-layer pitch-class similarity in circle-of-fifths order, per architecture (diagonal hidden)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path, flush=True)


def plot_pca(items, path):
    """items: (label, xy (12,2))"""
    tp = _tp()
    fig, ax = _grid(len(items))
    ring = tp.FIFTHS_ORDER + [tp.FIFTHS_ORDER[0]]
    for a, (label, xy) in zip(ax, items):
        a.plot(xy[ring, 0], xy[ring, 1], "--", color="grey", lw=0.8)
        a.scatter(xy[:, 0], xy[:, 1], c=[plt.cm.hsv(j / 12) for j in range(12)], s=40, zorder=3)
        for j in range(12):
            a.annotate(tp.NAMES[j], xy[j], xytext=(3, 3), textcoords="offset points", fontsize=7)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(label, fontsize=9)
    fig.suptitle("First two principal components of the output-layer pitch-class vectors, per architecture "
                 "(dashed: circle of fifths; orientation is arbitrary)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path, flush=True)


def plot_keys_grid(items, path):
    """items: (label, stats dict). One row per model, major and minor side by side."""
    tp = _tp()
    n = len(items)
    fig, ax = plt.subplots(n, 2, figsize=(13, 2.7 * n), squeeze=False)
    x = np.arange(12)
    for i, (label, st) in enumerate(items):
        for mo, (nm, kk) in enumerate((("major", tp.KK_MAJOR), ("minor", tp.KK_MINOR))):
            a = ax[i, mo]
            kkn = kk / kk.sum()
            pred = st["rel_pred"][mo] / max(st["rel_notes"][mo], 1)
            true = st["rel_true"][mo] / max(st["rel_true"][mo].sum(), 1)
            a.bar(x - 0.2, true, 0.4, color="lightgrey", label="true next note")
            a.bar(x + 0.2, pred, 0.4, color="tab:blue", label="model")
            a.plot(x, kkn, "o-", color="tab:red", ms=3, lw=1, label="Krumhansl-Kessler")
            a.set_xticks(x)
            a.set_xticklabels(tp.DEGREES, fontsize=7)
            a.set_title(f"{label} | {nm} ({st['rel_windows'][mo]} windows) | r(model, K-K) = {_r(pred, kkn):.2f}", fontsize=8)
            if i == 0 and mo == 0:
                a.legend(fontsize=7)
    fig.suptitle("Next-pitch-class distribution relative to the estimated tonic (scale degree, semitones above tonic)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path, flush=True)


def plot_errors_grid(items, path):
    tp = _tp()
    n = len(items)
    fig, ax = plt.subplots(n, 1, figsize=(10, 2.5 * n), squeeze=False)
    x = np.arange(12)
    names = ["same\nclass", "semi-\ntone", "whole\ntone", "minor\n3rd", "major\n3rd", "4th", "tri-\ntone", "5th", "minor\n6th",
             "major\n6th", "minor\n7th", "major\n7th"]
    for a, (label, st) in zip(ax.ravel(), items):
        obs, exp = tp.error_stats(st["err_int"], st["err_true_pc"], st["err_pred_pc"])
        a.bar(x - 0.2, obs * 100, 0.4, color="tab:blue", label="observed")
        a.bar(x + 0.2, exp * 100, 0.4, color="lightgrey", label="expected if guess unrelated to truth")
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=7)
        a.set_ylabel("% of wrong guesses", fontsize=8)
        a.set_title(f"{label} ({int(st['err_int'].sum())} wrong of {int(st['n_notes'])} notes)", fontsize=9)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Pitch-class interval of wrong predictions (guess minus truth, mod 12)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path, flush=True)


def write_csv(rows, path):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("saved", path, flush=True)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="folder with your training scripts, theory_plots.py")
    ap.add_argument("--models", default="all", help="comma list of: " + ",".join(SCRIPTS))
    ap.add_argument("--ckpt", action="append", default=[], help="key=path (repeatable)")
    ap.add_argument("--download", action="store_true", help="fetch artifacts from W&B (WANDB_API_KEY in the environment)")
    ap.add_argument("--what", choices=["embeddings", "keys", "errors", "all"], default="all")
    ap.add_argument("--out_dir", default="theory_compare_out")
    ap.add_argument("--data_root", default="data/maestro-v3.0.0")
    ap.add_argument("--max_windows", type=int, default=3000, help="test windows sampled per model")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min_corr", type=float, default=0.7, help="keep windows whose key estimate correlates at least this well")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--heads", type=int, default=None, help="override attention heads (only if the inferred value is wrong)")
    ap.add_argument("--with_random", action="store_true", help="also measure an untrained copy of each architecture")
    ap.add_argument("--random_only", action="store_true", help="only untrained copies (smoke test; no checkpoint or data needed)")
    args = ap.parse_args()
    sys.path.insert(0, args.src_dir)
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))      # theory_plots.py may sit next to this script
    import torch
    tp = _tp()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_args = dict(a.split("=", 1) for a in args.ckpt)
    keys = list(SCRIPTS) if args.models == "all" else args.models.split(",")
    if args.models == "all" and not args.random_only:
        keys = [k for k in keys if k in ckpt_args or ARTIFACTS.get(k)]          # skip remi6 unless a path is given
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows, mats, pcas, keyst = [], [], [], []

    failures = []

    def run_one(key):
        print(f"\n=== {key}", flush=True)
        mod = load_module(args.src_dir, key)
        variants = []                                           # (row-label suffix, model, cfg, trained?)
        if not args.random_only:
            sd = get_state_dict(find_ckpt(key, ckpt_args, args.download))
            cfg = infer_cfg(key, sd, args.heads)
            m = construct(key, mod, cfg)
            load_into(key, m, sd)
            variants.append(("", m.eval(), cfg, True))
        if args.with_random or args.random_only:
            torch.manual_seed(0)
            cfg0 = dict(DEFAULTS[key])
            if args.heads:
                cfg0["heads"] = args.heads
            variants.append((" [untrained]", construct(key, mod, cfg0).eval(), cfg0, False))

        for suffix, m, cfg, trained in variants:
            label = label_for(key, cfg, suffix)
            print(f"  {label}: {cfg}", flush=True)
            row = dict(model=key, label=label, untrained=not trained, config=str(cfg))
            V = pitch_vectors(key, m)
            r, C, xy = embedding_row(octave_mean(V["out"]), "out", args.n_perm)
            row.update(r)
            if trained:
                mats.append((label, C, r["out_fifths"], r["out_chrom"]))
                pcas.append((label, xy))
            if V["inp"] is not None:
                r2, _, _ = embedding_row(V["inp"], "inp", args.n_perm)
                row.update(r2)
                row["inp_desc"] = V["inp_desc"]
            print(f"    output layer: fifths {row['out_fifths']:+.2f} (p {row['out_p_fifths']:.3f}) | "
                  f"chromatic {row['out_chrom']:+.2f} (p {row['out_p_chrom']:.3f})", flush=True)
            if trained and args.what != "embeddings":
                try:
                    m = m.to(dev)
                    ds, ref, rebuilt = make_dataset(key, mod, args.data_root)
                    print(f"    test windows with the training hop: {ref} -> compare with the 'Dataset split ... Test:' line of "
                          f"your training log" + ("  | WARNING: note cache was rebuilt here, the split may differ from training "
                                                  "(glob order); use the original .pkl cache if you still have it" if rebuilt else ""),
                          flush=True)
                    st = new_stats()
                    for ctx, probs, tgt in batches(key, m, ds, dev, args.max_windows, args.batch):
                        accumulate(st, ctx, probs, tgt, args.min_corr)
                    row.update(key_row(st))
                    row["test_windows_training_hop"] = ref
                    row["cache_rebuilt"] = rebuilt
                    keyst.append((label, st))
                    print(f"    next-pitch accuracy on {st['n_notes']} scored notes: {row['pitch_acc_%']:.1f}% "
                          f"(pitch only; compare with W&B test/val PITCH accuracy, not all-token accuracy. "
                          f"Near-chance (<10%) means wrong head count, split or checkpoint)", flush=True)
                    m.cpu()
                except (Exception, SystemExit) as e:
                    failures.append((key, "keys/errors stage", repr(e)[:300]))
                    print(f"    !! keys/errors stage failed for {key}: {e!r} (embedding results are kept)", flush=True)
                    m.cpu()
            rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # per-model embedding figures for the trained network
        if variants and variants[0][3] and args.what in ("embeddings", "all"):
            _, m, cfg, _ = variants[0]
            V = pitch_vectors(key, m)
            tp.embedding_figure(octave_mean(V["out"]), f"{label_for(key, cfg)} - output-layer pitch vectors, octave-averaged",
                                os.path.join(args.out_dir, f"{key}_emb_out.png"), args.n_perm)
            if V["inp"] is not None:
                tp.embedding_figure(V["inp"], f"{label_for(key, cfg)} - {V['inp_desc']}",
                                    os.path.join(args.out_dir, f"{key}_emb_inp.png"), args.n_perm)

    for key in keys:
        try:
            run_one(key)
        except (Exception, SystemExit) as e:           # one bad checkpoint must not cost the other models
            failures.append((key, "model", repr(e)[:300]))
            print(f"!! {key} failed: {e!r}", flush=True)

    write_csv(rows, os.path.join(args.out_dir, "theory_compare.csv"))
    summary_figure(rows, os.path.join(args.out_dir, "theory_compare_summary.png"))
    if mats:
        plot_matrices(mats, os.path.join(args.out_dir, "theory_compare_matrices.png"))
        plot_pca(pcas, os.path.join(args.out_dir, "theory_compare_pca.png"))
    if keyst:
        plot_keys_grid(keyst, os.path.join(args.out_dir, "theory_compare_keys.png"))
        plot_errors_grid(keyst, os.path.join(args.out_dir, "theory_compare_errors.png"))
    if failures:
        print("\nFAILED (rerun these with --models KEY after fixing):")
        for k, stage, msg in failures:
            print(f"  {k} [{stage}]: {msg}")


if __name__ == "__main__":
    main()