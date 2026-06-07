"""Bootstrap analysis of paper_assets/real_5fold.json + real_5fold_preds.npz.

Produces:
  - paper_assets/real_5fold_summary.json   (means, 95% CIs, paired-test p-value)
  - prints LaTeX-ready strings for the paper's H3 section

Usage:
    python analyze_5fold.py
    python analyze_5fold.py --n-boot 10000
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "paper_assets")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="42,43",
                   help="Comma-separated list of training seeds to aggregate")
    args = p.parse_args()

    train_seeds = [int(s) for s in args.seeds.split(",")]
    runs = {}
    preds_data_per_seed = {}
    for s in train_seeds:
        json_path = os.path.join(ASSETS, f"real_5fold_seed{s}.json")
        npz_path  = os.path.join(ASSETS, f"real_5fold_preds_seed{s}.npz")
        if not os.path.exists(json_path):
            # back-compat: fall back to legacy unsuffixed names
            legacy = os.path.join(ASSETS, "real_5fold.json")
            legacy_npz = os.path.join(ASSETS, "real_5fold_preds.npz")
            if os.path.exists(legacy):
                json_path = legacy
                npz_path  = legacy_npz
            else:
                print(f"WARNING: {json_path} not found, skipping seed {s}")
                continue
        with open(json_path) as f:
            runs[s] = json.load(f)
        preds_data_per_seed[s] = np.load(npz_path)
        print(f"Loaded seed {s}: {len(runs[s]['results'])} fold-runs")
    if not runs:
        print("No data found.")
        return

    # for the first-seed-only legacy code path
    run = runs[train_seeds[0]] if train_seeds[0] in runs else next(iter(runs.values()))
    preds_data = preds_data_per_seed[train_seeds[0]] if train_seeds[0] in preds_data_per_seed \
        else next(iter(preds_data_per_seed.values()))

    # Per-(seed, n_reg) accuracies
    by_seed_n: dict[tuple, list[dict]] = defaultdict(list)
    for s, r_obj in runs.items():
        for r in r_obj["results"]:
            by_seed_n[(s, int(r["n_reg"]))].append(r)
    # Aggregated by n_reg only
    by_n: dict[int, list[dict]] = defaultdict(list)
    for (s, n_reg), rows in by_seed_n.items():
        for r in rows:
            r2 = dict(r); r2["__seed__"] = s
            by_n[n_reg].append(r2)

    summary = {"by_n": {}, "by_seed_n": {}, "paired_test": {},
               "n_boot": args.n_boot, "seed": args.seed,
               "training_seeds": list(train_seeds)}
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 64)
    print(f"Bootstrap (B = {args.n_boot}, seed = {args.seed})")
    print("=" * 64)

    # --- per-(seed, n_reg) summary -----------------------------------
    print("\n--- per-seed breakdown ---")
    for (s, n_reg), rows in sorted(by_seed_n.items()):
        per_fold = [r["best_test_acc"] * 100 for r in rows]
        summary["by_seed_n"][f"seed={s},n={n_reg}"] = {
            "per_fold_best_acc": per_fold,
            "mean": float(np.mean(per_fold)),
            "std":  float(np.std(per_fold)),
            "n_folds": len(per_fold),
        }
        print(f"  seed={s} n={n_reg}: mean={np.mean(per_fold):.2f}%  "
              f"std={np.std(per_fold):.2f}  folds={[f'{a:.2f}' for a in per_fold]}")

    # --- per-architecture summary (aggregated across seeds) ----------
    print("\n--- aggregated across all seeds ---")
    correct_by_n: dict[int, np.ndarray] = {}
    for n_reg, rows in sorted(by_n.items()):
        correct_concat = []
        for r in rows:
            s = r["__seed__"]
            pd = preds_data_per_seed[s]
            try:
                preds = pd[f"preds_n{n_reg}_fold{r['fold']}"]
                targets = pd[f"targets_n{n_reg}_fold{r['fold']}"]
            except KeyError:
                # legacy file might use a different key scheme
                continue
            correct_concat.append((preds == targets).astype(np.int8))
        if not correct_concat:
            print(f"  n={n_reg}: no preds data")
            continue
        correct = np.concatenate(correct_concat)
        correct_by_n[n_reg] = correct

        idx = rng.integers(0, len(correct), size=(args.n_boot, len(correct)))
        boot_acc = correct[idx].mean(axis=1) * 100
        lo, hi = np.percentile(boot_acc, [2.5, 97.5])
        mean = correct.mean() * 100
        per_fold = [r["best_test_acc"] * 100 for r in rows]
        summary["by_n"][f"n={n_reg}"] = {
            "n_test_clips": int(len(correct)),
            "n_folds": len(rows),
            "per_fold_best_acc": per_fold,
            "mean_acc": float(mean),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
            "fold_mean": float(np.mean(per_fold)),
            "fold_std": float(np.std(per_fold)),
        }
        print(f"  n={n_reg:>2}:  acc={mean:.2f}%   95% CI=[{lo:.2f}, {hi:.2f}]   "
              f"({len(rows)} fold-runs across {len(set(r['__seed__'] for r in rows))} seeds)")

    # --- paired test n=0 vs n=4 (or first vs second arch in sweep) ---
    if 0 in correct_by_n and 4 in correct_by_n:
        a, b = 0, 4
    else:
        keys = sorted(correct_by_n)
        if len(keys) < 2:
            print("\n(only one architecture; skipping paired test)")
            with open(os.path.join(ASSETS, "real_5fold_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            return
        a, b = keys[0], keys[1]

    ca, cb = correct_by_n[a], correct_by_n[b]
    if len(ca) != len(cb):
        print(f"\nWARNING: n={a} and n={b} have different test sizes "
              f"({len(ca)} vs {len(cb)}); paired test invalid.")
    else:
        diffs = (cb - ca).astype(np.float32)
        observed = diffs.mean() * 100  # accuracy delta in pp

        # paired bootstrap CI on the delta
        idx = rng.integers(0, len(diffs), size=(args.n_boot, len(diffs)))
        boot_diffs = diffs[idx].mean(axis=1) * 100
        lo, hi = np.percentile(boot_diffs, [2.5, 97.5])

        # paired permutation test (sign-flip)
        signs = rng.choice([-1, 1], size=(args.n_boot, len(diffs))).astype(np.float32)
        perm_diffs = (signs * diffs).mean(axis=1) * 100
        p_two = float(np.mean(np.abs(perm_diffs) >= np.abs(observed)))
        p_one = float(np.mean(perm_diffs >= observed)) if observed >= 0 else \
                float(np.mean(perm_diffs <= observed))

        summary["paired_test"] = {
            "a": f"n={a}", "b": f"n={b}",
            "delta_pp": float(observed),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
            "p_two_sided": p_two,
            "p_one_sided": p_one,
            "n_pairs": int(len(diffs)),
        }
        print(f"\nPaired n={b} - n={a}:")
        print(f"  delta              : {observed:+.3f} pp")
        print(f"  95% CI (bootstrap) : [{lo:+.3f}, {hi:+.3f}] pp")
        print(f"  permutation p (2s) : {p_two:.4f}")

    # --- write summary -----------------------------------------------
    out = os.path.join(ASSETS, "real_5fold_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")

    # --- LaTeX-ready strings -----------------------------------------
    print("\n" + "=" * 64)
    print("Drop-in strings for the paper's H3 section:")
    print("=" * 64)
    if "paired_test" in summary and summary["paired_test"]:
        pt = summary["paired_test"]
        b_n = sorted(by_n)
        a_summary = summary["by_n"][f"n={b_n[0]}"]
        b_summary = summary["by_n"][f"n={b_n[1]}"]
        print(f"""
% --- paste into Section V-C / Discussion / abstract ---
Top-1 accuracy on the full ESC-50 5-fold cross-validation:
  baseline (n=0): {a_summary['mean_acc']:.2f}\\% (95\\% CI [{a_summary['ci95_lo']:.2f}, {a_summary['ci95_hi']:.2f}])
  modified (n=4): {b_summary['mean_acc']:.2f}\\% (95\\% CI [{b_summary['ci95_lo']:.2f}, {b_summary['ci95_hi']:.2f}])
  paired delta  : {pt['delta_pp']:+.3f} pp, 95\\% CI [{pt['ci95_lo']:+.3f}, {pt['ci95_hi']:+.3f}]
  permutation p : {pt['p_two_sided']:.4f} (two-sided, 10k permutations)
""")

    # --- Regenerate training-dynamics figure -------------------------
    try:
        import matplotlib.pyplot as plt
        plt.rcParams["font.size"] = 10
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.0))
        for n_reg, color in [(0, "#1565C0"), (4, "#C62828")]:
            if n_reg not in by_n:
                continue
            runs = by_n[n_reg]
            train_curves = np.array([r["history"]["train_acc"] for r in runs]) * 100
            test_curves = np.array([r["history"]["test_acc"] for r in runs]) * 100
            xs = np.arange(1, train_curves.shape[1] + 1)
            for cv, label, panel in [(train_curves, "train", 0), (test_curves, "test", 1)]:
                m = cv.mean(axis=0); s = cv.std(axis=0)
                ax[panel].plot(xs, m, "o-", color=color, linewidth=1.5,
                               label=f"n={n_reg}")
                ax[panel].fill_between(xs, m - s, m + s, alpha=0.15, color=color)
        ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Train acc (%)")
        ax[0].set_title("(a) Training accuracy"); ax[0].grid(alpha=0.3); ax[0].legend()
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Test acc (%)")
        ax[1].set_title("(b) Held-out test accuracy"); ax[1].grid(alpha=0.3); ax[1].legend()
        plt.tight_layout()
        out_pdf = os.path.join(ASSETS, "fig_training_dynamics.pdf")
        plt.savefig(out_pdf); plt.close(fig)
        print(f"Wrote {out_pdf}")
    except Exception as e:
        print(f"figure generation skipped: {e}")

    # --- Per-class figure (delta) ------------------------------------
    if 0 in correct_by_n and 4 in correct_by_n:
        try:
            label_names = run.get("label_names")
            if not label_names:
                from datasets import load_dataset
                ds = load_dataset("ashraq/esc50", split="train")
                id2 = {int(t): n for t, n in zip(ds["target"], ds["category"])}
                label_names = [id2[i] for i in range(50)]
            # gather predictions and targets across folds for n=0 and n=4
            def per_class_acc(n_reg):
                acc = np.zeros(50); cnt = np.zeros(50)
                for r in by_n[n_reg]:
                    p = preds_data[f"preds_n{n_reg}_fold{r['fold']}"]
                    t = preds_data[f"targets_n{n_reg}_fold{r['fold']}"]
                    for tgt in range(50):
                        mask = t == tgt
                        if mask.sum():
                            acc[tgt] += (p[mask] == tgt).sum()
                            cnt[tgt] += mask.sum()
                return np.where(cnt > 0, acc / cnt, np.nan)
            a0, a4 = per_class_acc(0), per_class_acc(4)
            delta = (a4 - a0) * 100
            order = np.argsort(delta)
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ["#C62828" if d > 0 else "#1565C0" for d in delta[order]]
            ax.barh(range(50), delta[order], color=colors)
            ax.set_yticks(range(50))
            ax.set_yticklabels([label_names[i] for i in order], fontsize=7)
            ax.set_xlabel("Per-class accuracy delta (n=4 - n=0), pp")
            ax.set_title("Per-class effect of register tokens (5-fold CV)")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(alpha=0.3, axis="x")
            plt.tight_layout()
            out_pdf = os.path.join(ASSETS, "fig_perclass.pdf")
            plt.savefig(out_pdf); plt.close(fig)
            print(f"Wrote {out_pdf}")
        except Exception as e:
            print(f"per-class figure skipped: {e}")


if __name__ == "__main__":
    main()
