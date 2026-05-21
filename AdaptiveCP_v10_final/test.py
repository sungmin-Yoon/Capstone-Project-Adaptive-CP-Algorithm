import os
import platform
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from env import AdaptiveCPEnv
from DQN import DQN

# ═══════════════════════════════════════════════════════════════════════════
# 안테나 배열 선택
# [v10] None = 랜덤 | 16 / 64 / 256 = 해당 배열로 고정
# ═══════════════════════════════════════════════════════════════════════════
FIXED_M = None

# ── 테스트 설정 ────────────────────────────────────────────────────────────
NUM_STEPS  = 30
NLOS_START = 10
NLOS_END   = 18
CHARTS_DIR = "charts"
DPI        = 200

ACTION_MAP = {0: "Decrease CP", 1: "Maintain CP", 2: "Increase CP"}

# ── 색상 팔레트 (전 차트 통일) ─────────────────────────────────────────────
C_ADAPTIVE = "#2ca02c"   # 초록 — Adaptive DQN
C_FIXED    = "#d62728"   # 빨강 — Fixed CP
C_THEO     = "#636363"   # 진회색 — Theoretical
C_NLOS_BG  = "#fee0d2"   # 연살색 — NLOS 배경
C_TRAIN    = "#3182bd"   # 파랑 — Training curve
C_MA       = "#e6550d"   # 주황 — Moving average


# ═══════════════════════════════════════════════════════════════════════════
# 한국어 폰트 설정
# ═══════════════════════════════════════════════════════════════════════════
def setup_font():
    """OS별 한글 폰트를 자동 감지하여 matplotlib에 적용"""
    candidates = {
        "Windows": ["Malgun Gothic", "NanumGothic", "Gulim"],
        "Darwin":  ["AppleGothic", "NanumGothic", "Helvetica"],
        "Linux":   ["NanumGothic", "UnDotum", "DejaVu Sans"],
    }
    installed = {f.name for f in fm.fontManager.ttflist}
    system    = platform.system()
    for font in candidates.get(system, ["DejaVu Sans"]):
        if font in installed:
            matplotlib.rcParams["font.family"] = font
            break
    # 마이너스 기호 깨짐 방지
    matplotlib.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════════════════
# 공통 유틸리티
# ═══════════════════════════════════════════════════════════════════════════
def make_antenna_label(fixed_m: int | None, actual_m: int) -> str:
    """차트에 삽입할 안테나 배열 표기 문자열"""
    if fixed_m is None:
        return f"Antenna Array: Random Selection → M = {actual_m}"
    return f"Antenna Array: M = {fixed_m} (Fixed)"


def shade_nlos(ax):
    ax.axvspan(NLOS_START - 0.5, NLOS_END + 0.5,
               alpha=0.35, color=C_NLOS_BG, zorder=0, label="NLOS Zone")


def add_footer(fig, antenna_label: str):
    """그림 하단 우측에 안테나 정보 표기"""
    fig.text(
        0.99, 0.005, antenna_label,
        ha="right", va="bottom", fontsize=9,
        color="#636363", style="italic"
    )


def save_fig(fig, filename: str):
    fig.savefig(os.path.join(CHARTS_DIR, filename),
                dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  저장: {filename}")


def new_fig(title: str, antenna_label: str) -> tuple:
    """단일 차트용 Figure/Axes 생성"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    add_footer(fig, antenna_label)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 개별 차트 함수
# ═══════════════════════════════════════════════════════════════════════════
def chart_01_training(antenna_label: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Training Episode Reward",
                 fontsize=14, fontweight="bold", y=0.98)
    add_footer(fig, antenna_label)

    if os.path.exists("train_rewards.npy"):
        tr     = np.load("train_rewards.npy")
        window = max(1, len(tr) // 50)
        ma     = np.convolve(tr, np.ones(window) / window, mode="valid")
        ax.plot(tr, color=C_TRAIN, alpha=0.35, linewidth=0.6, label="Episode Reward")
        ax.plot(range(window - 1, len(tr)), ma, color=C_MA, linewidth=2.0,
                label=f"Moving Average (w={window})")
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, "train_rewards.npy not found\nRun train.py first",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "01_training_reward.png")


def chart_02_step_reward(steps, step_rewards, antenna_label: str):
    fig, ax = new_fig("Step-by-Step Reward (Test Phase)", antenna_label)
    shade_nlos(ax)
    ax.plot(steps, step_rewards, marker="o", color="#e6550d",
            linewidth=2.0, markersize=5, label="Step Reward", zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "02_step_reward.png")


def chart_03_sigma_ds(steps, sigma_ds_list, antenna_label: str):
    fig, ax = new_fig("RMS Delay Spread (σ_DS)", antenna_label)
    shade_nlos(ax)
    ax.plot(steps, sigma_ds_list, marker="D", color="#756bb1",
            linewidth=2.0, markersize=5, label="σ_DS (ns)", zorder=3)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Delay Spread (ns)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "03_sigma_ds.png")


def chart_04_cp_comparison(steps, theo_cp_list, fixed_cp_list,
                            adapt_cp_list, antenna_label: str):
    fig, ax = new_fig("Cyclic Prefix Length — 3-way Comparison", antenna_label)
    shade_nlos(ax)
    ax.plot(steps, theo_cp_list,  marker="s", linestyle="--", color=C_THEO,
            linewidth=1.5, markersize=4, label="Theoretical CP (3.0·σ_DS)", zorder=3)
    ax.plot(steps, fixed_cp_list, marker="x", linestyle=":",  color=C_FIXED,
            linewidth=1.5, markersize=6, label="Fixed CP (75 ns)", zorder=3)
    ax.plot(steps, adapt_cp_list, marker="^", color=C_ADAPTIVE,
            linewidth=2.0, markersize=5, label="Adaptive CP (DQN)", zorder=4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("CP Length (ns)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "04_cp_comparison.png")


def chart_05_se_comparison(steps, theo_se_list, fixed_se_list,
                            adapt_se_list, antenna_label: str):
    fig, ax = new_fig("Spectral Efficiency Comparison", antenna_label)
    shade_nlos(ax)
    ax.plot(steps, theo_se_list,  marker="s", linestyle="--", color=C_THEO,
            linewidth=1.5, markersize=4, label="Theoretical SE (σ×3.0)", zorder=3)
    ax.plot(steps, fixed_se_list, marker="x", linestyle=":",  color=C_FIXED,
            linewidth=1.5, markersize=6, label="Fixed CP SE", zorder=3)
    ax.plot(steps, adapt_se_list, marker="^", color=C_ADAPTIVE,
            linewidth=2.0, markersize=5, label="Adaptive SE (DQN)", zorder=4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("SE (bps/Hz)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "05_se_comparison.png")


def chart_06_power_efficiency(steps, theo_power_list, fixed_power_list,
                               adapt_power_list, antenna_label: str):
    fig, ax = new_fig("Power Efficiency Comparison (T_s / (T_s + T_CP))", antenna_label)
    shade_nlos(ax)
    ax.plot(steps, theo_power_list,  marker="s", linestyle="--", color=C_THEO,
            linewidth=1.5, markersize=4, label="Theoretical Eff.", zorder=3)
    ax.plot(steps, fixed_power_list, marker="x", linestyle=":",  color=C_FIXED,
            linewidth=1.5, markersize=6, label="Fixed CP Eff.", zorder=3)
    ax.plot(steps, adapt_power_list, marker="d", color="#7b2d8b",
            linewidth=2.0, markersize=5, label="Adaptive Eff. (DQN)", zorder=4)
    all_vals = theo_power_list + fixed_power_list + adapt_power_list
    ax.set_ylim(0, max(all_vals) * 1.2)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Efficiency (%)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "06_power_efficiency.png")


def chart_07_cp_ratio(steps, cp_sigma_ratio, theo_cp_list,
                      fixed_cp_list, sigma_ds_list, antenna_label: str):
    fig, ax = new_fig("CP / σ_DS Ratio — ISI Safety Margin", antenna_label)
    shade_nlos(ax)
    theo_r  = [tc / sd if sd > 0 else 0 for tc, sd in zip(theo_cp_list, sigma_ds_list)]
    fixed_r = [fc / sd if sd > 0 else 0 for fc, sd in zip(fixed_cp_list, sigma_ds_list)]
    ax.plot(steps, theo_r,          marker="s", linestyle="--", color=C_THEO,
            linewidth=1.5, markersize=4, label="Theoretical CP / σ_DS", zorder=3)
    ax.plot(steps, fixed_r,         marker="x", linestyle=":",  color=C_FIXED,
            linewidth=1.5, markersize=6, label="Fixed CP / σ_DS", zorder=3)
    ax.plot(steps, cp_sigma_ratio,  marker="^", color=C_ADAPTIVE,
            linewidth=2.0, markersize=5, label="Adaptive CP / σ_DS", zorder=4)
    ax.axhline(1.0, color="black",   linestyle="-",  linewidth=1.0,
               alpha=0.6, label="ISI Threshold (CP = σ_DS)")
    ax.axhline(1.1, color="#fd8d3c", linestyle="--", linewidth=1.2,
               alpha=0.9, label="CP Lower Bound (1.1·σ_DS)")
    ax.axhline(3.0, color=C_THEO,   linestyle=":",  linewidth=0.8,
               alpha=0.6, label="Paper T_CP (3.0·σ_DS)")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("CP / σ_DS", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "07_cp_sigma_ratio.png")


def chart_08_avg_se(theo_se_list, fixed_se_list,
                    adapt_se_list, antenna_label: str):
    fig, ax = new_fig("Average Spectral Efficiency Summary", antenna_label)

    labels  = ["Theoretical\n(σ×3.0)", "Fixed CP\n(75 ns)", "Adaptive\n(DQN)"]
    mean_se = [np.mean(theo_se_list), np.mean(fixed_se_list), np.mean(adapt_se_list)]
    colors  = [C_THEO, C_FIXED, C_ADAPTIVE]
    bars    = ax.bar(labels, mean_se, color=colors, alpha=0.82,
                     edgecolor="white", linewidth=1.2, width=0.5)

    # 수치 레이블
    for bar, val in zip(bars, mean_se):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Mean SE (bps/Hz)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_ylim(0, max(mean_se) * 1.35)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "08_avg_se_summary.png")


def chart_09_nlos_se(theo_se_list, fixed_se_list, adapt_se_list,
                     nlos_flags, antenna_label: str):
    fig, ax = new_fig("SE Comparison: Normal vs. NLOS", antenna_label)

    n_idx    = [i for i, f in enumerate(nlos_flags) if not f]
    nlos_idx = [i for i, f in enumerate(nlos_flags) if f]

    cats = ["Normal\n(Theoretical)", "NLOS\n(Theoretical)",
            "Normal\n(Fixed CP)",    "NLOS\n(Fixed CP)",
            "Normal\n(Adaptive)",    "NLOS\n(Adaptive)"]
    means = []
    for idx_list, se_list in [
        (n_idx, theo_se_list),   (nlos_idx, theo_se_list),
        (n_idx, fixed_se_list),  (nlos_idx, fixed_se_list),
        (n_idx, adapt_se_list),  (nlos_idx, adapt_se_list),
    ]:
        means.append(np.mean([se_list[i] for i in idx_list]) if idx_list else 0.0)

    bar_colors = ["#969696", "#525252",
                  "#fc9272", "#de2d26",
                  "#74c476", "#238b45"]
    bars2 = ax.bar(cats, means, color=bar_colors, alpha=0.88,
                   edgecolor="white", linewidth=1.2, width=0.6)
    for bar, val in zip(bars2, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean SE (bps/Hz)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "09_nlos_se_comparison.png")


def chart_summary_top4(steps, adapt_se_list, fixed_se_list, theo_se_list,
                       theo_se_mean, fixed_se_mean, adapt_se_mean,
                       nlos_flags, antenna_label: str):
    """보고서용 핵심 4-chart 요약 (2×2)"""
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Adaptive CP Algorithm — Performance Summary (v10)\n"
        f"Theoretical = σ_DS × ρ = 3·σ_DS  [Paper T_CP Definition]  |  {antenna_label}",
        fontsize=14, fontweight="bold", y=0.99
    )

    # ── 좌상: Training ──────────────────────────────────────────────────
    ax = axs[0, 0]
    if os.path.exists("train_rewards.npy"):
        tr     = np.load("train_rewards.npy")
        window = max(1, len(tr) // 50)
        ma     = np.convolve(tr, np.ones(window) / window, mode="valid")
        ax.plot(tr, color=C_TRAIN, alpha=0.3, linewidth=0.6)
        ax.plot(range(window - 1, len(tr)), ma, color=C_MA, linewidth=2.0,
                label=f"Moving Average (w={window})")
        ax.legend(fontsize=9)
    ax.set_title("Training Reward Convergence", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── 우상: SE 비교 ────────────────────────────────────────────────────
    ax = axs[0, 1]
    ax.axvspan(NLOS_START - 0.5, NLOS_END + 0.5,
               alpha=0.35, color=C_NLOS_BG, zorder=0, label="NLOS Zone")
    ax.plot(steps, theo_se_list,  marker="s", linestyle="--", color=C_THEO,
            linewidth=1.5, markersize=4, label="Theoretical SE", zorder=3)
    ax.plot(steps, fixed_se_list, marker="x", linestyle=":",  color=C_FIXED,
            linewidth=1.5, markersize=5, label="Fixed CP SE", zorder=3)
    ax.plot(steps, adapt_se_list, marker="^", color=C_ADAPTIVE,
            linewidth=2.0, markersize=4, label="Adaptive SE (DQN)", zorder=4)
    ax.set_title("Spectral Efficiency Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("SE (bps/Hz)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # ── 좌하: 평균 SE 바 ──────────────────────────────────────────────────
    ax = axs[1, 0]
    labels  = ["Theoretical\n(σ×3.0)", "Fixed CP\n(75 ns)", "Adaptive\n(DQN)"]
    means   = [theo_se_mean, fixed_se_mean, adapt_se_mean]
    bars    = ax.bar(labels, means, color=[C_THEO, C_FIXED, C_ADAPTIVE],
                     alpha=0.82, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    improve = (adapt_se_mean / fixed_se_mean - 1.0) * 100.0
    ax.annotate(f"+{improve:.0f}%\nvs Fixed CP",
                xy=(2, adapt_se_mean),
                xytext=(1.55, adapt_se_mean * 0.65),
                fontsize=9, color=C_ADAPTIVE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ADAPTIVE, lw=1.4))
    ax.set_title("Average SE Summary", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean SE (bps/Hz)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # ── 우하: Normal/NLOS SE ──────────────────────────────────────────────
    ax = axs[1, 1]
    n_idx    = [i for i, f in enumerate(nlos_flags) if not f]
    nlos_idx = [i for i, f in enumerate(nlos_flags) if f]
    cats = ["Normal\n(Theoretical)", "NLOS\n(Theoretical)",
            "Normal\n(Fixed CP)",    "NLOS\n(Fixed CP)",
            "Normal\n(Adaptive)",    "NLOS\n(Adaptive)"]
    nlos_means = []
    for idx_list, se_list in [
        (n_idx, theo_se_list),   (nlos_idx, theo_se_list),
        (n_idx, fixed_se_list),  (nlos_idx, fixed_se_list),
        (n_idx, adapt_se_list),  (nlos_idx, adapt_se_list),
    ]:
        nlos_means.append(np.mean([se_list[i] for i in idx_list]) if idx_list else 0.0)
    bar_colors = ["#969696", "#525252", "#fc9272", "#de2d26", "#74c476", "#238b45"]
    bars2 = ax.bar(cats, nlos_means, color=bar_colors, alpha=0.88,
                   edgecolor="white", linewidth=1.2, width=0.6)
    for bar, val in zip(bars2, nlos_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_title("SE: Normal vs. NLOS", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean SE (bps/Hz)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, max(nlos_means) * 1.3)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    save_fig(fig, "summary_top4.png")


# ═══════════════════════════════════════════════════════════════════════════
# 메인 테스트 함수
# ═══════════════════════════════════════════════════════════════════════════
def test():
    setup_font()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    env    = AdaptiveCPEnv(fixed_m=FIXED_M)
    device = torch.device("cpu")

    model = DQN(env.state_dim, env.action_dim).to(device)
    model.load_state_dict(torch.load("dqn_adaptive_cp.pth", map_location=device))
    model.eval()

    # 초기 reset 후 실제 M 값 확인
    state     = env.reset()
    actual_m  = env.M
    ant_label = make_antenna_label(FIXED_M, actual_m)

    m_print = f"M = {FIXED_M} (고정)" if FIXED_M is not None else f"Random → M = {actual_m}"
    print("=" * 70)
    print("  Adaptive CP Algorithm — Test Phase (v10)")
    print(f"  안테나 배열: {m_print}")
    print(f"  NLOS 급변 구간: Step {NLOS_START} ~ {NLOS_END}")
    print(f"  출력 폴더: {CHARTS_DIR}/")
    print("=" * 70)

    # ── 데이터 수집 ────────────────────────────────────────────────────────
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
    cp_sigma_ratio   = []

    for step in range(1, NUM_STEPS + 1):
        is_nlos = NLOS_START <= step <= NLOS_END

        with torch.no_grad():
            q_vals = model(torch.FloatTensor(state).unsqueeze(0).to(device))
        action = q_vals.argmax().item()

        next_state, reward, done, info = env.step(action, nlos_event=is_nlos)

        ratio = info["adaptive_cp"] / info["sigma_ds"] if info["sigma_ds"] > 0 else 0
        cp_sigma_ratio.append(ratio)

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
        print(f"Step {step:2d}{nlos_tag} | DS: {info['sigma_ds']*1e9:.1f} ns | "
              f"SNR: {state[1]:.1f} dB | Δθ: {state[2]:.1f}° | M: {int(state[3])} | "
              f"CP: {info['adaptive_cp']*1e9:.1f} ns | "
              f"SE: {info['adaptive_se']:.4f} | Action: {ACTION_MAP[action]}")

        state = next_state

    # ── ISI 통계 ───────────────────────────────────────────────────────────
    isi_violations = sum(1 for r in cp_sigma_ratio if r < 1.0)

    # ── 평균 지표 ──────────────────────────────────────────────────────────
    n_idx    = [i for i, f in enumerate(nlos_flags) if not f]
    nlos_idx = [i for i, f in enumerate(nlos_flags) if f]
    t_mean = np.mean(theo_se_list)
    f_mean = np.mean(fixed_se_list)
    a_mean = np.mean(adapt_se_list)

    print("\n" + "=" * 70)
    print("  [최종 성능 요약]")
    print(f"  평균 SE  | 이론기준(σ×3): {t_mean:.4f} | 고정CP: {f_mean:.4f} | Adaptive: {a_mean:.4f}")
    if nlos_idx:
        print(f"  NLOS SE  | 이론기준: {np.mean([theo_se_list[i] for i in nlos_idx]):.4f} | "
              f"고정CP: {np.mean([fixed_se_list[i] for i in nlos_idx]):.4f} | "
              f"Adaptive: {np.mean([adapt_se_list[i] for i in nlos_idx]):.4f}")
    print(f"  ISI 위반: {isi_violations}/{NUM_STEPS} ({isi_violations/NUM_STEPS*100:.1f}%)")
    print(f"  CP/σ_DS: min={min(cp_sigma_ratio):.2f}  mean={np.mean(cp_sigma_ratio):.2f}  "
          f"max={max(cp_sigma_ratio):.2f}")
    print("=" * 70)

    # ── 개별 차트 저장 ─────────────────────────────────────────────────────
    print(f"\n[Charts] {CHARTS_DIR}/ 폴더에 개별 차트 저장 중...")

    chart_01_training(ant_label)
    chart_02_step_reward(steps, step_rewards, ant_label)
    chart_03_sigma_ds(steps, sigma_ds_list, ant_label)
    chart_04_cp_comparison(steps, theo_cp_list, fixed_cp_list, adapt_cp_list, ant_label)
    chart_05_se_comparison(steps, theo_se_list, fixed_se_list, adapt_se_list, ant_label)
    chart_06_power_efficiency(steps, theo_power_list, fixed_power_list, adapt_power_list, ant_label)
    chart_07_cp_ratio(steps, cp_sigma_ratio, theo_cp_list, fixed_cp_list, sigma_ds_list, ant_label)
    chart_08_avg_se(theo_se_list, fixed_se_list, adapt_se_list, ant_label)
    chart_09_nlos_se(theo_se_list, fixed_se_list, adapt_se_list, nlos_flags, ant_label)
    chart_summary_top4(
        steps, adapt_se_list, fixed_se_list, theo_se_list,
        t_mean, f_mean, a_mean, nlos_flags, ant_label
    )

    print(f"\n[Charts] 완료 — {CHARTS_DIR}/ 에 총 10개 파일 저장")
    print(f"  개별 차트 : 01~09_*.png (10 × 6 inch, {DPI} dpi)")
    print(f"  요약 차트 : summary_top4.png (16 × 10 inch, {DPI} dpi)")


if __name__ == "__main__":
    test()
