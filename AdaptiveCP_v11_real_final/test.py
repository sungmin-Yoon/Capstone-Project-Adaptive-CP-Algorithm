import os
import platform
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import seaborn as sns

from env import AdaptiveCPEnv
from DQN import DQN

# ═══════════════════════════════════════════════════════════════════════════
# 테스트 설정
# ═══════════════════════════════════════════════════════════════════════════
NUM_STEPS  = 30
NLOS_START = 10
NLOS_END   = 18
CHARTS_DIR = "charts"          # 루트 charts 폴더
DPI        = 200

# M=16, 64, 256 세 가지를 한 번에 실행
M_VALUES   = [16, 64, 256]

ACTION_MAP = {0: "Decrease CP", 1: "Maintain CP", 2: "Increase CP"}

# ═══════════════════════════════════════════════════════════════════════════
# ── 통일 색상 팔레트 ─────────────────────────────────────────────────────────
#   모든 차트에서 동일한 색상 사용 (절대 변경 금지)
# ═══════════════════════════════════════════════════════════════════════════
C_ADAPTIVE = "#2ca02c"   # 초록  — Adaptive DQN
C_FIXED    = "#d62728"   # 빨강  — Fixed CP
C_THEO     = "#636363"   # 진회색 — Theoretical (3·σ_DS)
C_NO_CP    = "#1f77b4"   # 파랑  — No CP (CP = 0)
C_NLOS_BG  = "#fee0d2"   # 연살색 — NLOS 배경 음영
C_TRAIN    = "#3182bd"   # 파랑  — Training curve
C_MA       = "#e6550d"   # 주황  — Moving average


# ═══════════════════════════════════════════════════════════════════════════
# 한국어 폰트 설정
# ═══════════════════════════════════════════════════════════════════════════
def setup_font():
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.2)
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
    matplotlib.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════════════════
# 공통 유틸리티
# ═══════════════════════════════════════════════════════════════════════════
def make_antenna_label(fixed_m: int, actual_m: int) -> str:
    return f"Antenna Array: M = {fixed_m} (Fixed)"


def shade_nlos(ax):
    ax.axvspan(NLOS_START - 0.5, NLOS_END + 0.5,
               alpha=0.35, color=C_NLOS_BG, zorder=0, label="NLOS Zone")


def add_footer(fig, antenna_label: str):
    fig.text(0.99, 0.005, antenna_label,
             ha="right", va="bottom", fontsize=9,
             color="#636363", style="italic")


def save_fig(fig, filename: str, charts_dir: str):
    path = os.path.join(charts_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  저장: {os.path.relpath(path)}")


def new_fig(title: str, antenna_label: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    add_footer(fig, antenna_label)
    sns.despine(ax=ax)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 테스트 1회 실행 → 결과 dict 반환
# ═══════════════════════════════════════════════════════════════════════════
def run_single(fixed_m: int, model, device) -> dict:
    """fixed_m 안테나 배열로 NUM_STEPS 스텝 실행, 결과 dict 반환"""
    env   = AdaptiveCPEnv(fixed_m=fixed_m)
    state = env.reset()

    steps            = list(range(1, NUM_STEPS + 1))
    step_rewards     = []
    sigma_ds_list    = []
    snr_list         = []
    delta_theta_list = []
    nlos_flags       = []

    theo_cp_list, fixed_cp_list, adapt_cp_list = [], [], []
    theo_se_list, fixed_se_list, adapt_se_list, no_cp_se_list = [], [], [], []
    theo_power_list, fixed_power_list, adapt_power_list = [], [], []
    cp_sigma_ratio   = []

    ber_adapt_list,  ber_fixed_list  = [], []
    ber_theo_list,   ber_no_cp_list  = [], []
    sinr_adapt_list, sinr_fixed_list = [], []
    sinr_theo_list,  sinr_no_cp_list = [], []

    for step in range(1, NUM_STEPS + 1):
        is_nlos = NLOS_START <= step <= NLOS_END

        with torch.no_grad():
            q = model(torch.FloatTensor(state).unsqueeze(0).to(device))
        action = q.argmax().item()

        next_state, reward, done, info = env.step(action, nlos_event=is_nlos)

        ratio = info["adaptive_cp"] / max(info["sigma_ds"], 1e-12)

        step_rewards.append(reward)
        sigma_ds_list.append(info["sigma_ds"] * 1e9)
        snr_list.append(info["snr"])
        delta_theta_list.append(next_state[2])
        nlos_flags.append(is_nlos)
        cp_sigma_ratio.append(ratio)

        theo_cp_list.append(info["theo_cp"]      * 1e9)
        fixed_cp_list.append(info["fixed_cp"]    * 1e9)
        adapt_cp_list.append(info["adaptive_cp"] * 1e9)

        theo_se_list.append(info["theo_se"])
        fixed_se_list.append(info["fixed_se"])
        adapt_se_list.append(info["adaptive_se"])
        no_cp_se_list.append(info["no_cp_se"])

        theo_power_list.append(info["theo_power_eff"])
        fixed_power_list.append(info["fixed_power_eff"])
        adapt_power_list.append(info["power_eff"])

        sinr_adapt_list.append(info["sinr_adapt_db"])
        sinr_fixed_list.append(info["sinr_fixed_db"])
        sinr_theo_list.append(info["sinr_theo_db"])
        sinr_no_cp_list.append(info["sinr_no_cp_db"])

        ber_adapt_list.append(info["ber_adapt"])
        ber_fixed_list.append(info["ber_fixed"])
        ber_theo_list.append(info["ber_theo"])
        ber_no_cp_list.append(info["ber_no_cp"])

        nlos_tag = " [NLOS!]" if is_nlos else ""
        print(f"  Step {step:2d}{nlos_tag} | DS:{info['sigma_ds']*1e9:.1f}ns | "
              f"SNR:{info['snr']:.1f}dB | CP:{info['adaptive_cp']*1e9:.1f}ns | "
              f"SE:{info['adaptive_se']:.4f} | "
              f"SINR:{info['sinr_adapt_db']:.1f}dB | BER:{info['ber_adapt']:.2e} | "
              f"{ACTION_MAP[action]}")

        state = next_state

    return {
        "fixed_m"         : fixed_m,
        "ant_label"       : make_antenna_label(fixed_m, fixed_m),
        "steps"           : steps,
        "step_rewards"    : step_rewards,
        "sigma_ds_list"   : sigma_ds_list,
        "snr_list"        : snr_list,
        "delta_theta_list": delta_theta_list,
        "nlos_flags"      : nlos_flags,
        "theo_cp_list"    : theo_cp_list,
        "fixed_cp_list"   : fixed_cp_list,
        "adapt_cp_list"   : adapt_cp_list,
        "theo_se_list"    : theo_se_list,
        "fixed_se_list"   : fixed_se_list,
        "adapt_se_list"   : adapt_se_list,
        "no_cp_se_list"   : no_cp_se_list,
        "theo_power_list" : theo_power_list,
        "fixed_power_list": fixed_power_list,
        "adapt_power_list": adapt_power_list,
        "cp_sigma_ratio"  : cp_sigma_ratio,
        "sinr_adapt_list" : sinr_adapt_list,
        "sinr_fixed_list" : sinr_fixed_list,
        "sinr_theo_list"  : sinr_theo_list,
        "sinr_no_cp_list" : sinr_no_cp_list,
        "ber_adapt_list"  : ber_adapt_list,
        "ber_fixed_list"  : ber_fixed_list,
        "ber_theo_list"   : ber_theo_list,
        "ber_no_cp_list"  : ber_no_cp_list,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 개별 차트 함수
# ═══════════════════════════════════════════════════════════════════════════

def chart_01_training(charts_dir: str, antenna_label: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Training Episode Reward", fontsize=14, fontweight="bold", y=0.98)
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
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "01_training_reward.png", charts_dir)


def chart_02_step_reward(r: dict, charts_dir: str):
    fig, ax = new_fig("Step-by-Step Reward (Test Phase)", r["ant_label"])
    shade_nlos(ax)
    ax.plot(r["steps"], r["step_rewards"], marker="o", color=C_MA,
            linewidth=2.0, markersize=5, label="Step Reward", zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "02_step_reward.png", charts_dir)


def chart_03_sigma_ds(r: dict, charts_dir: str):
    fig, ax = new_fig("RMS Delay Spread (σ_DS)", r["ant_label"])
    shade_nlos(ax)
    ax.plot(r["steps"], r["sigma_ds_list"], marker="D", color="#756bb1",
            linewidth=2.0, markersize=5, label="σ_DS (ns)", zorder=3)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Delay Spread (ns)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "03_sigma_ds.png", charts_dir)


def chart_04_cp_comparison(r: dict, charts_dir: str):
    fig, ax = new_fig("Cyclic Prefix Length — 3-way Comparison", r["ant_label"])
    shade_nlos(ax)
    ax.plot(r["steps"], r["theo_cp_list"],  marker="s", linestyle="--", color=C_THEO,
            linewidth=1.5, markersize=4, label="Theoretical CP (3.0·σ_DS)", zorder=3)
    ax.plot(r["steps"], r["fixed_cp_list"], marker="x", linestyle=":",  color=C_FIXED,
            linewidth=1.5, markersize=6, label="Fixed CP (75 ns)", zorder=3)
    ax.plot(r["steps"], r["adapt_cp_list"], marker="^", color=C_ADAPTIVE,
            linewidth=2.0, markersize=5, label="Adaptive CP (DQN)", zorder=4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("CP Length (ns)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "04_cp_comparison.png", charts_dir)


def chart_05_se_comparison(r: dict, charts_dir: str):
    """
    [v11] No CP 케이스 추가
    논문 핵심 포인트: CP=0 일 때도 Fixed/Theo CP 대비 SE가 높을 수 있음
      - No CP: 오버헤드 0 + ISI 패널티(×0.5)
      - Fixed CP: 오버헤드 75ns 로 SE 손실 큼
    """
    fig, ax = new_fig(
        "Spectral Efficiency Comparison  [v11: No CP added]",
        r["ant_label"],
    )
    shade_nlos(ax)
    ax.plot(r["steps"], r["no_cp_se_list"],  marker="o", linestyle="-.",
            color=C_NO_CP,    linewidth=1.5, markersize=4, label="No CP (CP=0)", zorder=2)
    ax.plot(r["steps"], r["theo_se_list"],   marker="s", linestyle="--",
            color=C_THEO,     linewidth=1.5, markersize=4, label="Theoretical SE (3·σ_DS)", zorder=3)
    ax.plot(r["steps"], r["fixed_se_list"],  marker="x", linestyle=":",
            color=C_FIXED,    linewidth=1.5, markersize=6, label="Fixed CP SE (75 ns)", zorder=3)
    ax.plot(r["steps"], r["adapt_se_list"],  marker="^",
            color=C_ADAPTIVE, linewidth=2.0, markersize=5, label="Adaptive SE (DQN)", zorder=4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("SE (bps/Hz)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "05_se_comparison.png", charts_dir)


def chart_06_power_efficiency(r: dict, charts_dir: str):
    fig, ax = new_fig("Power Efficiency (T_s / (T_s + T_CP))", r["ant_label"])
    shade_nlos(ax)
    ax.plot(r["steps"], r["theo_power_list"],  marker="s", linestyle="--", color=C_THEO,
            linewidth=1.5, markersize=4, label="Theoretical Eff.", zorder=3)
    ax.plot(r["steps"], r["fixed_power_list"], marker="x", linestyle=":",  color=C_FIXED,
            linewidth=1.5, markersize=6, label="Fixed CP Eff.", zorder=3)
    ax.plot(r["steps"], r["adapt_power_list"], marker="d", color=C_ADAPTIVE,
            linewidth=2.0, markersize=5, label="Adaptive Eff. (DQN)", zorder=4)
    all_vals = r["theo_power_list"] + r["fixed_power_list"] + r["adapt_power_list"]
    ax.set_ylim(0, max(all_vals) * 1.2)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Efficiency (%)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "06_power_efficiency.png", charts_dir)


def chart_07_cp_ratio(r: dict, charts_dir: str):
    fig, ax = new_fig("CP / σ_DS Ratio — ISI Safety Margin", r["ant_label"])
    shade_nlos(ax)
    sigma = r["sigma_ds_list"]
    theo_ratio  = [tc / sd if sd > 0 else 0 for tc, sd in zip(r["theo_cp_list"],  sigma)]
    fixed_ratio = [fc / sd if sd > 0 else 0 for fc, sd in zip(r["fixed_cp_list"], sigma)]
    ax.plot(r["steps"], theo_ratio,          marker="s", linestyle="--", color=C_THEO,
            linewidth=1.5, markersize=4, label="Theoretical CP / σ_DS", zorder=3)
    ax.plot(r["steps"], fixed_ratio,         marker="x", linestyle=":",  color=C_FIXED,
            linewidth=1.5, markersize=6, label="Fixed CP / σ_DS", zorder=3)
    ax.plot(r["steps"], r["cp_sigma_ratio"], marker="^", color=C_ADAPTIVE,
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
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "07_cp_sigma_ratio.png", charts_dir)


def chart_08_avg_se(r: dict, charts_dir: str):
    """[v11] No CP 바 추가"""
    fig, ax = new_fig("Average Spectral Efficiency Summary  [v11: No CP added]",
                      r["ant_label"])
    labels = ["No CP\n(CP=0)", "Theoretical\n(σ×3.0)", "Fixed CP\n(75 ns)", "Adaptive\n(DQN)"]
    means  = [
        np.mean(r["no_cp_se_list"]),
        np.mean(r["theo_se_list"]),
        np.mean(r["fixed_se_list"]),
        np.mean(r["adapt_se_list"]),
    ]
    colors = [C_NO_CP, C_THEO, C_FIXED, C_ADAPTIVE]
    bars   = ax.bar(labels, means, color=colors, alpha=0.82,
                    edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean SE (bps/Hz)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.set_ylim(0, max(means) * 1.35)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "08_avg_se_summary.png", charts_dir)


def chart_09_nlos_se(r: dict, charts_dir: str):
    """[v11] No CP 바 추가"""
    fig, ax = new_fig("SE Comparison: Normal vs. NLOS  [v11: No CP added]",
                      r["ant_label"])
    flags    = r["nlos_flags"]
    n_idx    = [i for i, f in enumerate(flags) if not f]
    nlos_idx = [i for i, f in enumerate(flags) if f]

    cats   = ["Normal\n(No CP)", "NLOS\n(No CP)",
              "Normal\n(Theo.)", "NLOS\n(Theo.)",
              "Normal\n(Fixed)", "NLOS\n(Fixed)",
              "Normal\n(Adapt.)", "NLOS\n(Adapt.)"]
    pairs  = [
        (n_idx,    r["no_cp_se_list"]),  (nlos_idx, r["no_cp_se_list"]),
        (n_idx,    r["theo_se_list"]),   (nlos_idx, r["theo_se_list"]),
        (n_idx,    r["fixed_se_list"]),  (nlos_idx, r["fixed_se_list"]),
        (n_idx,    r["adapt_se_list"]),  (nlos_idx, r["adapt_se_list"]),
    ]
    means = [np.mean([sl[i] for i in il]) if il else 0.0 for il, sl in pairs]
    bar_colors = [
        "#6baed6", "#2171b5",   # No CP  (파랑 계열)
        "#969696", "#525252",   # Theo.  (회색 계열)
        "#fc9272", "#de2d26",   # Fixed  (빨강 계열)
        "#74c476", "#238b45",   # Adapt. (초록 계열)
    ]
    bars = ax.bar(cats, means, color=bar_colors, alpha=0.88,
                  edgecolor="white", linewidth=1.2, width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean SE (bps/Hz)", fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, max(means) * 1.3)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "09_nlos_se_comparison.png", charts_dir)


def chart_10_ber(r: dict, charts_dir: str):
    """[v11] BER 비교 (4가지 CP 케이스, log 스케일)"""
    fig, ax = new_fig("BER Comparison — BPSK Approximation  (log scale)",
                      r["ant_label"])
    shade_nlos(ax)
    ax.semilogy(r["steps"], r["ber_no_cp_list"],   marker="o", linestyle="-.",
                color=C_NO_CP,    linewidth=1.5, markersize=4, label="No CP (CP=0)", zorder=2)
    ax.semilogy(r["steps"], r["ber_theo_list"],    marker="s", linestyle="--",
                color=C_THEO,     linewidth=1.5, markersize=4, label="Theoretical (3·σ_DS)", zorder=3)
    ax.semilogy(r["steps"], r["ber_fixed_list"],   marker="x", linestyle=":",
                color=C_FIXED,    linewidth=1.5, markersize=6, label="Fixed CP (75 ns)", zorder=3)
    ax.semilogy(r["steps"], r["ber_adapt_list"],   marker="^",
                color=C_ADAPTIVE, linewidth=2.0, markersize=5, label="Adaptive CP (DQN)", zorder=4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "10_ber_comparison.png", charts_dir)


def chart_11_sinr(r: dict, charts_dir: str):
    """[v11] SINR 비교 (4가지 CP 케이스)"""
    fig, ax = new_fig("SINR Comparison (dB)  [4 CP Cases]", r["ant_label"])
    shade_nlos(ax)
    ax.plot(r["steps"], r["sinr_no_cp_list"],   marker="o", linestyle="-.",
            color=C_NO_CP,    linewidth=1.5, markersize=4, label="No CP (CP=0)", zorder=2)
    ax.plot(r["steps"], r["sinr_theo_list"],    marker="s", linestyle="--",
            color=C_THEO,     linewidth=1.5, markersize=4, label="Theoretical (3·σ_DS)", zorder=3)
    ax.plot(r["steps"], r["sinr_fixed_list"],   marker="x", linestyle=":",
            color=C_FIXED,    linewidth=1.5, markersize=6, label="Fixed CP (75 ns)", zorder=3)
    ax.plot(r["steps"], r["sinr_adapt_list"],   marker="^",
            color=C_ADAPTIVE, linewidth=2.0, markersize=5, label="Adaptive CP (DQN)", zorder=4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("SINR (dB)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "11_sinr_comparison.png", charts_dir)


def chart_12_param_table(r: dict, charts_dir: str):
    """
    [v11] 평균 파라미터 비교표
    고정 CP 대비 Adaptive CP의 지표 변화를 한눈에 확인.
    행: 각 지표 / 열: No CP · Theoretical · Fixed CP · Adaptive DQN · Δ vs Fixed
    """
    flags    = r["nlos_flags"]
    n_idx    = [i for i, f in enumerate(flags) if not f]
    nlos_idx = [i for i, f in enumerate(flags) if f]

    def m(lst):       return float(np.mean(lst))
    def mn(lst):      return float(np.mean([lst[i] for i in n_idx]))   if n_idx    else 0.0
    def mnlos(lst):   return float(np.mean([lst[i] for i in nlos_idx])) if nlos_idx else 0.0
    def delta_pct(a, f): return f"{(a/f - 1)*100:+.1f}%" if abs(f) > 1e-10 else "—"
    def delta_abs(a, f): return f"{a - f:+.3f}"
    def ber_ratio(a, f):
        # a=adaptive BER, f=fixed BER
        # 비율 = adaptive/fixed: >1이면 Adaptive가 더 나쁨, <1이면 더 좋음
        if a < 1e-15 or f < 1e-15:
            return "—"
        ratio = a / f
        if ratio > 1.0:
            return f"×{ratio:.1f} worse"
        else:
            return f"×{1/ratio:.1f} better"

    no_cp = r["no_cp_se_list"]; theo = r["theo_se_list"]
    fixed = r["fixed_se_list"]; adpt = r["adapt_se_list"]

    # 각 지표 행 구성
    # [지표명, No CP, Theoretical, Fixed CP, Adaptive, Δ vs Fixed]
    rows = [
        ["지표", "No CP\n(CP=0)", "Theoretical\n(3·σ_DS)", "Fixed CP\n(75 ns)", "Adaptive\n(DQN)", "Δ vs Fixed CP"],
        # ── 채널 파라미터 (CP 무관) ────────────────────────────────────────
        ["σ_DS mean (ns)",
         "—", f"{m(r['sigma_ds_list']):.2f}", f"{m(r['sigma_ds_list']):.2f}", f"{m(r['sigma_ds_list']):.2f}", "—"],
        ["SNR mean (dB)",
         "—", f"{m(r['snr_list']):.2f}", f"{m(r['snr_list']):.2f}", f"{m(r['snr_list']):.2f}", "—"],
        # ── CP 파라미터 ────────────────────────────────────────────────────
        ["CP mean (ns)",
         "0",
         f"{m(r['theo_cp_list']):.2f}",
         f"{m(r['fixed_cp_list']):.2f}",
         f"{m(r['adapt_cp_list']):.2f}",
         delta_abs(m(r["adapt_cp_list"]), m(r["fixed_cp_list"])) + " ns"],
        ["CP / σ_DS mean",
         "0",
         f"{m(r['theo_cp_list'])/m(r['sigma_ds_list']):.2f}",
         f"{m(r['fixed_cp_list'])/m(r['sigma_ds_list']):.2f}",
         f"{m(r['cp_sigma_ratio']):.2f}",
         delta_abs(m(r["cp_sigma_ratio"]),
                   m(r["fixed_cp_list"]) / m(r["sigma_ds_list"]))],
        # ── SE ─────────────────────────────────────────────────────────────
        ["SE mean (bps/Hz)",
         f"{m(no_cp):.4f}",
         f"{m(theo):.4f}",
         f"{m(fixed):.4f}",
         f"{m(adpt):.4f}",
         delta_pct(m(adpt), m(fixed))],
        ["SE Normal (bps/Hz)",
         f"{mn(no_cp):.4f}", f"{mn(theo):.4f}", f"{mn(fixed):.4f}", f"{mn(adpt):.4f}",
         delta_pct(mn(adpt), mn(fixed))],
        ["SE NLOS (bps/Hz)",
         f"{mnlos(no_cp):.4f}" if nlos_idx else "—",
         f"{mnlos(theo):.4f}"  if nlos_idx else "—",
         f"{mnlos(fixed):.4f}" if nlos_idx else "—",
         f"{mnlos(adpt):.4f}"  if nlos_idx else "—",
         delta_pct(mnlos(adpt), mnlos(fixed)) if nlos_idx else "—"],
        # ── SINR ───────────────────────────────────────────────────────────
        ["SINR mean (dB)",
         f"{m(r['sinr_no_cp_list']):.2f}",
         f"{m(r['sinr_theo_list']):.2f}",
         f"{m(r['sinr_fixed_list']):.2f}",
         f"{m(r['sinr_adapt_list']):.2f}",
         delta_abs(m(r["sinr_adapt_list"]), m(r["sinr_fixed_list"])) + " dB"],
        # ── BER ────────────────────────────────────────────────────────────
        ["BER mean",
         f"{m(r['ber_no_cp_list']):.2e}",
         f"{m(r['ber_theo_list']):.2e}",
         f"{m(r['ber_fixed_list']):.2e}",
         f"{m(r['ber_adapt_list']):.2e}",
         ber_ratio(m(r["ber_adapt_list"]), m(r["ber_fixed_list"]))],
        # ── Power Efficiency ───────────────────────────────────────────────
        ["Power Eff. mean (%)",
         "100.0",
         f"{m(r['theo_power_list']):.2f}",
         f"{m(r['fixed_power_list']):.2f}",
         f"{m(r['adapt_power_list']):.2f}",
         delta_abs(m(r["adapt_power_list"]), m(r["fixed_power_list"])) + " pp"],
        # ── ISI ────────────────────────────────────────────────────────────
        ["ISI Violations",
         f"{NUM_STEPS}/{NUM_STEPS}",
         "0",
         "0",
         f"{sum(1 for v in r['cp_sigma_ratio'] if v < 1.0)}/{NUM_STEPS}",
         "—"],
    ]

    # ─── matplotlib Table ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(17, 7))
    ax.axis("off")
    fig.suptitle(
        f"Parameter Comparison Table  (M = {r['fixed_m']}, "
        f"NLOS Steps {NLOS_START}–{NLOS_END})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    add_footer(fig, r["ant_label"])

    header = rows[0]
    data   = rows[1:]

    tbl = ax.table(
        cellText=data,
        colLabels=header,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.6)

    # 헤더 스타일
    col_colors = {
        0: "#f0f0f0",   # 지표 열
        1: "#dbeafe",   # No CP  (파랑 계열)
        2: "#e5e7eb",   # Theoretical (회색)
        3: "#fee2e2",   # Fixed CP (빨강)
        4: "#d1fae5",   # Adaptive (초록)
        5: "#fef3c7",   # Δ (노랑)
    }
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            # 헤더 행
            bg = [None, "#1f77b4", "#636363", "#d62728", "#2ca02c", "#b45309"]
            fc = "white" if col_idx > 0 else "#333333"
            if col_idx > 0:
                cell.set_facecolor(bg[col_idx])
            else:
                cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold", color=fc)
        else:
            cell.set_facecolor(col_colors.get(col_idx, "white"))
            if col_idx == 0:
                cell.set_text_props(fontweight="bold")
            # Δ 열 양수/음수 색상
            if col_idx == 5:
                txt = cell.get_text().get_text()
                if txt.startswith("+"):
                    cell.set_text_props(color="#166534")   # 짙은 초록
                elif txt.startswith("-"):
                    cell.set_text_props(color="#991b1b")   # 짙은 빨강

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    save_fig(fig, "12_param_table.png", charts_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 요약 4-차트 (보고서용)
# ═══════════════════════════════════════════════════════════════════════════
def chart_summary(r: dict, charts_dir: str):
    """보고서용 핵심 4-chart 요약 (2×2)"""
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Adaptive CP Algorithm — Performance Summary (v11)  |  M = {r['fixed_m']}  |  "
        f"Theoretical = σ_DS × 3  [Paper T_CP]",
        fontsize=13, fontweight="bold", y=0.99,
    )
    add_footer(fig, r["ant_label"])

    steps = r["steps"]

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

    # ── 우상: SE 비교 (No CP 포함) ────────────────────────────────────────
    ax = axs[0, 1]
    ax.axvspan(NLOS_START - 0.5, NLOS_END + 0.5,
               alpha=0.35, color=C_NLOS_BG, zorder=0, label="NLOS Zone")
    ax.plot(steps, r["no_cp_se_list"],  marker="o", linestyle="-.",
            color=C_NO_CP,    linewidth=1.2, markersize=3, label="No CP", zorder=2)
    ax.plot(steps, r["theo_se_list"],   marker="s", linestyle="--",
            color=C_THEO,     linewidth=1.2, markersize=3, label="Theoretical SE", zorder=3)
    ax.plot(steps, r["fixed_se_list"],  marker="x", linestyle=":",
            color=C_FIXED,    linewidth=1.2, markersize=4, label="Fixed CP SE", zorder=3)
    ax.plot(steps, r["adapt_se_list"],  marker="^",
            color=C_ADAPTIVE, linewidth=2.0, markersize=4, label="Adaptive SE", zorder=4)
    ax.set_title("Spectral Efficiency (incl. No CP)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("SE (bps/Hz)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8)

    # ── 좌하: 평균 SE (No CP 포함) ────────────────────────────────────────
    ax = axs[1, 0]
    labels = ["No CP\n(CP=0)", "Theo.\n(σ×3.0)", "Fixed CP\n(75 ns)", "Adaptive\n(DQN)"]
    means  = [
        np.mean(r["no_cp_se_list"]), np.mean(r["theo_se_list"]),
        np.mean(r["fixed_se_list"]), np.mean(r["adapt_se_list"]),
    ]
    bars = ax.bar(labels, means, color=[C_NO_CP, C_THEO, C_FIXED, C_ADAPTIVE],
                  alpha=0.82, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_title("Average SE Summary", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean SE (bps/Hz)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_ylim(0, max(means) * 1.35)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # ── 우하: SINR 비교 ───────────────────────────────────────────────────
    ax = axs[1, 1]
    ax.axvspan(NLOS_START - 0.5, NLOS_END + 0.5,
               alpha=0.35, color=C_NLOS_BG, zorder=0, label="NLOS Zone")
    ax.plot(steps, r["sinr_no_cp_list"],   marker="o", linestyle="-.",
            color=C_NO_CP,    linewidth=1.2, markersize=3, label="No CP SINR", zorder=2)
    ax.plot(steps, r["sinr_theo_list"],    marker="s", linestyle="--",
            color=C_THEO,     linewidth=1.2, markersize=3, label="Theo. SINR", zorder=3)
    ax.plot(steps, r["sinr_fixed_list"],   marker="x", linestyle=":",
            color=C_FIXED,    linewidth=1.2, markersize=4, label="Fixed CP SINR", zorder=3)
    ax.plot(steps, r["sinr_adapt_list"],   marker="^",
            color=C_ADAPTIVE, linewidth=2.0, markersize=4, label="Adaptive SINR", zorder=4)
    ax.set_title("SINR Comparison (dB)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("SINR (dB)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8)

    for a in axs.flat:
        sns.despine(ax=a)

    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    save_fig(fig, "summary.png", charts_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 모든 차트 저장
# ═══════════════════════════════════════════════════════════════════════════
def save_all_charts(r: dict, charts_dir: str):
    ant = r["ant_label"]
    chart_01_training(charts_dir, ant)
    chart_02_step_reward(r, charts_dir)
    chart_03_sigma_ds(r, charts_dir)
    chart_04_cp_comparison(r, charts_dir)
    chart_05_se_comparison(r, charts_dir)
    chart_06_power_efficiency(r, charts_dir)
    chart_07_cp_ratio(r, charts_dir)
    chart_08_avg_se(r, charts_dir)
    chart_09_nlos_se(r, charts_dir)
    chart_10_ber(r, charts_dir)
    chart_11_sinr(r, charts_dir)
    chart_12_param_table(r, charts_dir)
    chart_summary(r, charts_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 요약 출력
# ═══════════════════════════════════════════════════════════════════════════
def print_summary(r: dict):
    flags    = r["nlos_flags"]
    nlos_idx = [i for i, f in enumerate(flags) if f]
    n_idx    = [i for i, f in enumerate(flags) if not f]
    viols    = sum(1 for v in r["cp_sigma_ratio"] if v < 1.0)

    t_mean = np.mean(r["theo_se_list"])
    f_mean = np.mean(r["fixed_se_list"])
    a_mean = np.mean(r["adapt_se_list"])
    nc_mean= np.mean(r["no_cp_se_list"])

    print(f"\n  [M={r['fixed_m']} 최종 성능 요약]")
    print(f"  ── SE ────────────────────────────────────────────────────────────")
    print(f"  No CP:      {nc_mean:.4f} bps/Hz")
    print(f"  이론 기준:  {t_mean:.4f} bps/Hz")
    print(f"  고정 CP:    {f_mean:.4f} bps/Hz")
    print(f"  Adaptive:   {a_mean:.4f} bps/Hz  ({(a_mean/f_mean-1)*100:+.1f}% vs Fixed)")
    if nlos_idx:
        fn = np.mean([r["fixed_se_list"][i] for i in nlos_idx])
        an = np.mean([r["adapt_se_list"][i] for i in nlos_idx])
        print(f"  NLOS SE:    Fixed={fn:.4f}  Adaptive={an:.4f}  ({(an/fn-1)*100:+.1f}% vs Fixed)")
    print(f"  ── SINR ──────────────────────────────────────────────────────────")
    print(f"  Adaptive:   {np.mean(r['sinr_adapt_list']):.2f} dB    "
          f"Fixed: {np.mean(r['sinr_fixed_list']):.2f} dB    "
          f"No CP: {np.mean(r['sinr_no_cp_list']):.2f} dB")
    print(f"  ── BER ───────────────────────────────────────────────────────────")
    print(f"  Adaptive:   {np.mean(r['ber_adapt_list']):.2e}    "
          f"Fixed: {np.mean(r['ber_fixed_list']):.2e}    "
          f"No CP: {np.mean(r['ber_no_cp_list']):.2e}")
    print(f"  ── ISI ───────────────────────────────────────────────────────────")
    print(f"  ISI 위반: {viols}/{NUM_STEPS}  "
          f"CP/σ_DS: min={min(r['cp_sigma_ratio']):.2f}  "
          f"mean={np.mean(r['cp_sigma_ratio']):.2f}  "
          f"max={max(r['cp_sigma_ratio']):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════
def test():
    setup_font()

    device = torch.device("cpu")
    model  = DQN(6, 3).to(device)
    model.load_state_dict(torch.load("dqn_adaptive_cp.pth", map_location=device))
    model.eval()

    print("=" * 70)
    print("  Adaptive CP Algorithm — Test Phase (v11)")
    print(f"  M 배열: {M_VALUES} (각각 별도 실행)")
    print(f"  NLOS 구간: Step {NLOS_START} ~ {NLOS_END}")
    print(f"  결과 저장: {CHARTS_DIR}/M=XX/")
    print("=" * 70)

    for m_val in M_VALUES:
        subdir = os.path.join(CHARTS_DIR, f"M={m_val}")
        os.makedirs(subdir, exist_ok=True)

        print(f"\n{'─'*70}")
        print(f"  ▶  M = {m_val} 실행 중...")
        print(f"{'─'*70}")

        results = run_single(m_val, model, device)
        print_summary(results)

        print(f"\n  [M={m_val}] 차트 저장 → {subdir}/")
        save_all_charts(results, subdir)
        print(f"  [M={m_val}] 완료 — 총 13개 파일 (01~12 + summary)")

    print("\n" + "=" * 70)
    print(f"  전체 완료: {CHARTS_DIR}/M=16/, M=64/, M=256/ 각 13개 차트")
    print("=" * 70)


if __name__ == "__main__":
    test()
