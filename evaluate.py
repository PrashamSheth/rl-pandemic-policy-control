"""
Evaluate every policy, including the saved DQN+PI v3 model, on the same seeds.

Each policy is run for NUM_SEEDS episodes; episode s of every policy starts
from np.random.seed(BASE_SEED + s), so all policies see the same initial
population. Writes results/per_episode.csv, results/summary.csv and the
figures in images/.

    python evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandemic_sim as ps

NUM_SEEDS = 20
BASE_SEED = 42

LABELS = {"Always_Open": "No policy", "Manual_Reactive": "Manual threshold",
          "Always_Lockdown": "Lockdown", "Dynamic_PI": "Dynamic PI",
          "PID": "PID", "DQN_PI": "DQN + PI (saved model)"}
COLORS = {"Always_Open": "#E24B4A", "Manual_Reactive": "#BA7517",
          "Always_Lockdown": "#1D9E75", "Dynamic_PI": "#9B59B6",
          "PID": "#FF6B35", "DQN_PI": "#2196F3"}


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    net = ps.load_trained_model("saved_models/dqn_pi_v3.pth")

    rows, curves = [], {}
    for p in ps.POLICIES:
        hs, ms = [], []
        for s in range(NUM_SEEDS):
            np.random.seed(BASE_SEED + s)
            h, m = ps.run_policy(p, net)
            hs.append(h); ms.append(m)
            rows.append({"policy": p, "seed": BASE_SEED + s, **ps.episode_metrics(h, m)})
        curves[p] = (np.array(hs), np.array(ms))
        print(f"{p:16s} done")

    df = pd.DataFrame(rows)
    df.to_csv("results/per_episode.csv", index=False)
    summary = (df.groupby("policy")[["Peak", "Mobility", "Days Over Cap", "Efficiency"]]
                 .agg(["mean", "std"]).reindex(ps.POLICIES))
    summary.columns = [f"{a} {b}" for a, b in summary.columns]
    summary.round(3).to_csv("results/summary.csv")

    print(f"\n{'Policy':<24} {'Peak':>13} {'Mobility':>14} {'Days over cap':>15} {'Efficiency':>16}")
    for p in ps.POLICIES:
        r = summary.loc[p]
        print(f"{LABELS[p]:<24} {r['Peak mean']:7.0f} ± {r['Peak std']:<4.0f}"
              f"{r['Mobility mean']:8.2f} ± {r['Mobility std']:<4.2f}"
              f"{r['Days Over Cap mean']:9.1f} ± {r['Days Over Cap std']:<4.1f}"
              f"{r['Efficiency mean']:9.4f} ± {r['Efficiency std']:.4f}")

    # Paired comparison: same seed, DQN minus each controller
    print("\nDQN + PI minus controller, paired by seed (mean ± std over seeds):")
    piv = df.pivot(index="seed", columns="policy")
    for ref in ["Dynamic_PI", "PID"]:
        dp = piv["Peak"]["DQN_PI"] - piv["Peak"][ref]
        dm = piv["Mobility"]["DQN_PI"] - piv["Mobility"][ref]
        wins = int(((dp <= 0) & (dm >= 0)).sum())
        print(f"  vs {LABELS[ref]:<11}: peak {dp.mean():+7.1f} ± {dp.std():5.1f}, "
              f"mobility {dm.mean():+.3f} ± {dm.std():.3f}, "
              f"better on both in {wins}/{NUM_SEEDS} seeds")

    days = np.arange(1, ps.DAYS + 1)

    # Figure 1: infection curves
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for p in ps.POLICIES:
        h = curves[p][0]
        ax.plot(days, h.mean(0), color=COLORS[p], lw=2.2 if p == "DQN_PI" else 1.6, label=LABELS[p])
        ax.fill_between(days, h.mean(0) - h.std(0), h.mean(0) + h.std(0), color=COLORS[p], alpha=0.08)
    ax.axhline(ps.CAPACITY_LIMIT, color="grey", ls=":", label="Capacity (500)")
    ax.set(xlabel="Day", ylabel="Infected agents", xlim=(1, ps.DAYS),
           title=f"Infection curves, mean ± 1 std over {NUM_SEEDS} seeds")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig("images/infection_curves.png", dpi=130)

    # Figure 2: mobility (step size) chosen by the adaptive policies
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for p in ["Dynamic_PI", "PID", "DQN_PI"]:
        ax.plot(days, curves[p][1].mean(0), color=COLORS[p], lw=2, label=LABELS[p])
    ax.axhline(ps.L_OPEN, color="grey", ls="--", lw=1, label="Fully open")
    ax.axhline(ps.L_LOCKDOWN, color="#1D9E75", ls="--", lw=1, label="Lockdown")
    ax.set(xlabel="Day", ylabel="Step size L (mobility)", xlim=(1, ps.DAYS),
           title=f"Mobility chosen by the adaptive policies (mean over {NUM_SEEDS} seeds)")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig("images/mobility.png", dpi=130)

    # Figure 3: bar charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, title in zip(axes, ["Peak", "Days Over Cap", "Mobility"],
                              ["Peak infections", "Days over capacity", "Total mobility"]):
        means = [summary.loc[p, f"{col} mean"] for p in ps.POLICIES]
        stds = [summary.loc[p, f"{col} std"] for p in ps.POLICIES]
        ax.bar(range(len(ps.POLICIES)), means, yerr=stds, capsize=4,
               color=[COLORS[p] for p in ps.POLICIES])
        ax.set_xticks(range(len(ps.POLICIES)))
        ax.set_xticklabels([LABELS[p] for p in ps.POLICIES], rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
    axes[0].axhline(ps.CAPACITY_LIMIT, color="grey", ls=":")
    fig.suptitle(f"Policy comparison, mean ± 1 std over {NUM_SEEDS} seeds")
    fig.tight_layout(); fig.savefig("images/policy_comparison.png", dpi=130)

    # Figure 4: trade-off between peak and mobility
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for p in ps.POLICIES:
        d = df[df.policy == p]
        ax.scatter(d["Mobility"], d["Peak"], color=COLORS[p], alpha=0.35, s=18)
        ax.scatter(d["Mobility"].mean(), d["Peak"].mean(), color=COLORS[p], s=120,
                   edgecolor="black", label=LABELS[p])
    ax.set(xlabel="Total mobility (higher is better)", ylabel="Peak infections (lower is better)",
           title="Trade-off: each small dot is one seed, large dot is the mean")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("images/tradeoff.png", dpi=130)


if __name__ == "__main__":
    main()
