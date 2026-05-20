import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import AdaptiveCPEnv
from DQN import DQN

NUM_STEPS  = 30
NLOS_START = 10
NLOS_END   = 18

ACTION_MAP = {0: "Decrease CP", 1: "Maintain CP", 2: "Increase CP"}


def test():
    env    = AdaptiveCPEnv()
    device = torch.device("cpu")

    model = DQN(env.state_dim, env.action_dim).to(device)
    model.load_state_dict(torch.load("dqn_adaptive_cp.pth", map_location=device))
    model.eval()

    print("=" * 70)
    print("  Adaptive CP Algorithm — Test Phase (v6)")
    print(f"  NLOS 급변 구간: Step {NLOS_START} ~ {NLOS_END}")
    print("=" * 70)

    steps            = list(range(1, NUM_STEPS + 1))
    step_rewards     = []
    sigma_ds_list    = []
    nlos_flags       = []
    theo_cp_list     = []
    fixed_cp_list    = []
    adapt_cp_list    = []
    theo_se_list     = []
    fixed_se_list    = []
    adapt_se_list    = []
    theo_power_list  = []
    fixed_power_list = []
    adapt_power_list = []

    state = env.reset()

    for step in range(1, NUM_STEPS + 1):
        is_nlos = NLOS_START <= step <= NLOS_END

        # NLOS 타이밍 수정: 행동 전 채널 먼저 갱신 (v5)
        if is_nlos:
            env._set_nlos_channel()
            env.sigma_ds = max(env._calculate_ds(), 1e-12)
            state        = env._make_state()

        with torch.no_grad():
            q_vals = model(torch.FloatTensor(state).unsqueeze(0).to(device))
        action = q_vals.argmax().item()

        next_state, reward, done, info = env.step(action, nlos_event=False)

        step_rewards.append(reward)
        sigma_ds_list.append(info["sigma_ds"] * 1e9)
        nlos_flags.append(is_nlos)
        theo_cp_list.append(info["theo_cp"]      * 1e9)
        fixed_cp_list.append(info["fixed_cp"]    * 1e9)
        adapt_cp_list.append(info["adaptive_cp"] * 1e9)
        theo_se_list.append(info["theo_se"])
        fixed_se_list.append(info["fixed_se"])
        adapt_se_list.append(info["adaptive_se"])
        theo_power_list.append(info["theo_power_eff"])
        fixed_power_list.append(info["fixed_power_eff"])
        adapt_power_list.append(info["power_eff"])

        nlos_tag = " [NLOS!]" if is_nlos else ""
        print(f"\nStep {step:2d}{nlos_tag} | DS: {state[0]:.2f} ns | "
              f"SNR: {state[1]:.1f} dB | Δθ: {state[2]:.1f}° | M: {int(state[3])}")
        print(f"  [이론 기준]  CP: {info['theo_cp']*1e9:6.2f} ns | "
              f"SE: {info['theo_se']:.4f} bps/Hz | Eff: {info['theo_power_eff']:.2f}%")
        print(f"  [고정 CP]    CP: {info['fixed_cp']*1e9:6.2f} ns | "
              f"SE: {info['fixed_se']:.4f} bps/Hz | Eff: {info['fixed_power_eff']:.2f}%")
        print(f"  [Adaptive]   CP: {info['adaptive_cp']*1e9:6.2f} ns | "
              f"SE: {info['adaptive_se']:.4f} bps/Hz | "
              f"Eff: {info['power_eff']:.2f}% | Action: {ACTION_MAP[action]}")
        print(f"  RSSI: {info['rssi']:.2f} | Reward: {reward:.4f}")
        print("-" * 70)

        state = next_state

    # ── 대시보드 시각화 ───────────────────────────────────────────────────
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(
        "Adaptive CP Algorithm — Performance Dashboard (v6)\n"
        f"(NLOS 급변 구간: Step {NLOS_START}~{NLOS_END}, 음영 표시)",
        fontsize=15, fontweight="bold"
    )

    def shade_nlos(ax):
        ax.axvspan(NLOS_START - 0.5, NLOS_END + 0.5,
                   alpha=0.25, color="red", label="NLOS Zone")

    # ① 학습 보상 추이
    ax = axs[0, 0]
    if os.path.exists("train_rewards.npy"):
        tr     = np.load("train_rewards.npy")
        window = max(1, len(tr) // 50)
        ma     = np.convolve(tr, np.ones(window) / window, mode="valid")
        ax.plot(tr, color="steelblue", alpha=0.6, linewidth=0.8)
        ax.plot(range(window - 1, len(tr)), ma, color="red", linewidth=1.5,
                label=f"Moving Avg (w={window})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "train_rewards.npy 없음\n(먼저 train.py 실행)",
                ha="center", va="center", fontsize=10)
    ax.set_title("① Training: Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True, linestyle="--", alpha=0.5)

    # ② 스텝별 보상
    ax = axs[0, 1]
    shade_nlos(ax)
    ax.plot(steps, step_rewards, marker="o", color="orange",
            linewidth=2, markersize=4, label="Step Reward")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("② Test: Step-by-Step Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ③ σ_DS 변화
    ax = axs[0, 2]
    shade_nlos(ax)
    ax.plot(steps, sigma_ds_list, marker="D", color="purple",
            linewidth=2, markersize=4, label="σ_DS (ns)")
    ax.set_title("③ RMS Delay Spread (σ_DS)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Delay Spread (ns)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ④ CP 길이 3-way 비교
    ax = axs[1, 0]
    shade_nlos(ax)
    ax.plot(steps, theo_cp_list,  marker="s", linestyle="--", color="gray",
            linewidth=1.5, markersize=4, label="Theoretical CP (ρ·σ_DS)")
    ax.plot(steps, fixed_cp_list, marker="x", linestyle=":",  color="red",
            linewidth=1.5, markersize=5, label="Fixed CP (75ns)")
    ax.plot(steps, adapt_cp_list, marker="^", color="green",
            linewidth=2,   markersize=4, label="Adaptive CP (DQN)")
    ax.set_title("④ CP Length Comparison (3-way)")
    ax.set_xlabel("Step")
    ax.set_ylabel("CP Length (ns)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ⑤ SE 3-way 비교
    ax = axs[1, 1]
    shade_nlos(ax)
    ax.plot(steps, theo_se_list,  marker="s", linestyle="--", color="gray",
            linewidth=1.5, markersize=4, label="Theoretical SE")
    ax.plot(steps, fixed_se_list, marker="x", linestyle=":",  color="red",
            linewidth=1.5, markersize=5, label="Fixed CP SE")
    ax.plot(steps, adapt_se_list, marker="^", color="green",
            linewidth=2,   markersize=4, label="Adaptive SE (DQN)")
    ax.set_title("⑤ Spectral Efficiency Comparison (3-way)")
    ax.set_xlabel("Step")
    ax.set_ylabel("SE (bps/Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ⑥ 전력 효율 3-way 비교
    ax = axs[1, 2]
    shade_nlos(ax)
    ax.plot(steps, theo_power_list,  marker="s", linestyle="--", color="gray",
            linewidth=1.5, markersize=4, label="Theoretical Eff")
    ax.plot(steps, fixed_power_list, marker="x", linestyle=":",  color="red",
            linewidth=1.5, markersize=5, label="Fixed CP Eff")
    ax.plot(steps, adapt_power_list, marker="d", color="purple",
            linewidth=2,   markersize=4, label="Adaptive Eff (DQN)")
    all_vals = theo_power_list + fixed_power_list + adapt_power_list
    ax.set_ylim(0, max(all_vals) + 15)
    ax.set_title("⑥ Power Efficiency Comparison (3-way)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Efficiency (%)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ⑦ CP/σ_DS 비율
    ax = axs[2, 0]
    shade_nlos(ax)
    theo_r  = [tc/sd if sd > 0 else 0 for tc, sd in zip(theo_cp_list, sigma_ds_list)]
    fixed_r = [fc/sd if sd > 0 else 0 for fc, sd in zip(fixed_cp_list, sigma_ds_list)]
    adapt_r = [ac/sd if sd > 0 else 0 for ac, sd in zip(adapt_cp_list, sigma_ds_list)]
    ax.plot(steps, theo_r,  marker="s", linestyle="--", color="gray",
            linewidth=1.5, markersize=4, label="Theoretical CP/σ_DS")
    ax.plot(steps, fixed_r, marker="x", linestyle=":",  color="red",
            linewidth=1.5, markersize=5, label="Fixed CP/σ_DS")
    ax.plot(steps, adapt_r, marker="^", color="green",
            linewidth=2,   markersize=4, label="Adaptive CP/σ_DS")
    ax.axhline(1.0, color="black", linestyle="-",  linewidth=0.8,
               alpha=0.5, label="ISI 임계 (CP=σ_DS)")
    ax.axhline(1.5, color="blue",  linestyle="--", linewidth=0.8,
               alpha=0.5, label="Sweet Spot 상한 (1.5×)")
    ax.set_title("⑦ CP / σ_DS Ratio (ISI 안전 마진)")
    ax.set_xlabel("Step")
    ax.set_ylabel("CP / σ_DS")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ⑧ 평균 SE 요약
    ax      = axs[2, 1]
    labels  = ["Theoretical", "Fixed CP", "Adaptive (DQN)"]
    mean_se = [np.mean(theo_se_list), np.mean(fixed_se_list), np.mean(adapt_se_list)]
    colors  = ["gray", "red", "green"]
    bars    = ax.bar(labels, mean_se, color=colors, alpha=0.75, edgecolor="black")
    for bar, val in zip(bars, mean_se):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_title("⑧ Average SE Summary (전체 구간)")
    ax.set_ylabel("Mean SE (bps/Hz)")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # ⑨ Normal vs NLOS SE 비교
    ax       = axs[2, 2]
    n_idx    = [i for i, f in enumerate(nlos_flags) if not f]
    nlos_idx = [i for i, f in enumerate(nlos_flags) if f]
    cats     = ["Normal\n(Theoretical)", "NLOS\n(Theoretical)",
                "Normal\n(Fixed CP)",    "NLOS\n(Fixed CP)",
                "Normal\n(Adaptive)",    "NLOS\n(Adaptive)"]
    means = []
    for idx_list, se_list in [
        (n_idx, theo_se_list),   (nlos_idx, theo_se_list),
        (n_idx, fixed_se_list),  (nlos_idx, fixed_se_list),
        (n_idx, adapt_se_list),  (nlos_idx, adapt_se_list)
    ]:
        means.append(np.mean([se_list[i] for i in idx_list]) if idx_list else 0.0)

    bar_colors = ["#AAAAAA","#666666","#FF8888","#CC0000","#88CC88","#008800"]
    bars2      = ax.bar(cats, means, color=bar_colors, alpha=0.85, edgecolor="black")
    for bar, val in zip(bars2, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
    ax.set_title("⑨ Normal vs NLOS 구간 SE 비교")
    ax.set_ylabel("Mean SE (bps/Hz)")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = "adaptive_cp_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Test] 대시보드 저장 완료 → {out_path}")
    plt.show()

    print("\n" + "=" * 70)
    print("  [최종 성능 요약]")
    print(f"  평균 SE  | 이론기준: {np.mean(theo_se_list):.4f} | "
          f"고정CP: {np.mean(fixed_se_list):.4f} | "
          f"Adaptive: {np.mean(adapt_se_list):.4f} bps/Hz")
    print(f"  평균 Eff | 이론기준: {np.mean(theo_power_list):.2f}% | "
          f"고정CP: {np.mean(fixed_power_list):.2f}% | "
          f"Adaptive: {np.mean(adapt_power_list):.2f}%")
    if nlos_idx:
        print(f"\n  [NLOS 구간 ({len(nlos_idx)} steps) SE]")
        print(f"    이론기준: {np.mean([theo_se_list[i] for i in nlos_idx]):.4f} | "
              f"고정CP: {np.mean([fixed_se_list[i] for i in nlos_idx]):.4f} | "
              f"Adaptive: {np.mean([adapt_se_list[i] for i in nlos_idx]):.4f} bps/Hz")
    print("=" * 70)


if __name__ == "__main__":
    test()
