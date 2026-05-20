"""
v7 환경 경량 분석 — scipy.integrate 없이 tau_hat 샘플링만 사용
sigma_DS 분포, ISI 위반 가능성, 정책별 SE 비교
"""
import sys, random, math
sys.path.insert(0, '.')
import numpy as np

random.seed(42)
np.random.seed(42)

# 3GPP TR 38.900 파라미터 (env.py와 동일)
LOS_MU, LOS_SIG   = -7.03, 0.66
NLOS_MU, NLOS_SIG = -6.44, 0.39
TAU_REF = 25e-9        # 논문 기준 τ̂
TAU_MAX = 100e-9       # v6 이후 클리핑 상한
LOS_MED = 10 ** LOS_MU
FIXED_CP = TAU_REF * 3.0   # 75ns
RHO      = 3.0
B        = 100e6
Ts       = 1.0 / B         # 10ns

def sample_tau(is_nlos):
    mu  = NLOS_MU  if is_nlos else LOS_MU
    sig = NLOS_SIG if is_nlos else LOS_SIG
    ds  = 10 ** random.gauss(mu, sig)
    tau = ds * (TAU_REF / LOS_MED)
    return float(np.clip(tau, 5e-9, TAU_MAX))

# sigma_DS ≈ tau_hat / 2  (keyhole 모델, 빔 완전정렬 가정)
# 실제 env._calculate_ds()는 빔 파라미터에 따라 달라지지만
# 분포 특성 파악에는 이 근사가 충분
def approx_sigma(tau): return tau * 0.5

def se(cp, sigma, snr_db):
    snr = 10 ** (snr_db / 10.0)
    s   = (Ts / (Ts + cp)) * math.log2(1 + snr)
    return s * 0.5 if cp < sigma else s

N = 10000

# ── 1. sigma_DS 분포 ────────────────────────────────────────────────────
los_sig  = [approx_sigma(sample_tau(False)) * 1e9 for _ in range(N)]
nlos_sig = [approx_sigma(sample_tau(True))  * 1e9 for _ in range(N)]

print("=" * 65)
print("  [1] 근사 sigma_DS 분포 (tau_hat/2, ns) — N=10000")
print("=" * 65)
for lbl, d in [("LOS ", los_sig), ("NLOS", nlos_sig)]:
    print(f"  {lbl} | mean={np.mean(d):.1f}  std={np.std(d):.1f}  "
          f"p10={np.percentile(d,10):.1f}  p50={np.percentile(d,50):.1f}  "
          f"p90={np.percentile(d,90):.1f}  max={np.max(d):.1f}")

# ── 2. 정책별 SE & ISI 위반율 (전체 환경, 40% NLOS 노출) ─────────────────
print(f"\n  [2] 정책별 SE & ISI 위반율 (N={N} 샘플, 40% NLOS)")
print("=" * 65)

se_fixed, se_theo, se_sweet = [], [], []
isi_fixed, isi_theo, isi_sweet = 0, 0, 0

for _ in range(N):
    is_nlos = random.random() < 0.4
    tau   = sample_tau(is_nlos)
    sigma = approx_sigma(tau)
    snr   = random.uniform(3.0, 12.0) if is_nlos else random.uniform(10.0, 25.0)

    cp_f = FIXED_CP
    cp_t = sigma * RHO
    cp_s = sigma * 1.25

    se_fixed.append(se(cp_f, sigma, snr))
    se_theo.append(se(cp_t, sigma, snr))
    se_sweet.append(se(cp_s, sigma, snr))
    isi_fixed += (cp_f < sigma)
    isi_theo  += (cp_t < sigma)
    isi_sweet += (cp_s < sigma)

for lbl, sv, ic in [
    ("Fixed CP(75ns) ", se_fixed, isi_fixed),
    ("Theo  (3×sigma)", se_theo,  isi_theo),
    ("Sweet (1.25×σ) ", se_sweet, isi_sweet),
]:
    print(f"  {lbl}  SE={np.mean(sv):.4f}  ISI={ic/N*100:.2f}%")

# ── 3. NLOS 한정 분석 ───────────────────────────────────────────────────
print(f"\n  [3] NLOS 한정 분석 (N={N})")
print("=" * 65)
nlos_se_f, nlos_se_t, nlos_se_s = [], [], []
isi_inevitable = 0

for _ in range(N):
    tau   = sample_tau(True)
    sigma = approx_sigma(tau)
    snr   = random.uniform(3.0, 12.0)
    nlos_se_f.append(se(FIXED_CP,     sigma, snr))
    nlos_se_t.append(se(sigma*RHO,    sigma, snr))
    nlos_se_s.append(se(sigma*1.25,   sigma, snr))
    if FIXED_CP < sigma: isi_inevitable += 1

pct = isi_inevitable / N * 100
print(f"  Fixed CP(75ns) < sigma_DS, ISI 필연 비율: {pct:.1f}%")
print(f"  Fixed  SE: {np.mean(nlos_se_f):.4f}")
print(f"  Theo   SE: {np.mean(nlos_se_t):.4f}")
print(f"  Sweet  SE: {np.mean(nlos_se_s):.4f}")

# ── 4. v7 step_size 복귀 속도 계산 ─────────────────────────────────────
print(f"\n  [4] NLOS 이후 CP 복귀 속도 (v6 vs v7)")
print("=" * 65)

# 시나리오: NLOS 종료 직후 current_cp=150ns, sigma=12ns (LOS 복귀)
sigma_post = 12e-9
cp_start   = 150e-9

def steps_to_recover_v6(cp, sigma):
    steps = 0
    while cp > sigma * 1.5 and steps < 500:
        ratio = cp / sigma
        if ratio < 0.8:   ss = sigma * 0.5
        elif ratio < 1.5: ss = sigma * 0.2
        else:             ss = max(sigma * 0.05, 1e-9)
        cp = max(cp - ss, sigma)
        steps += 1
    return steps

def steps_to_recover_v7(cp, sigma):
    steps = 0
    while cp > sigma * 1.5 and steps < 500:
        ratio = cp / sigma
        if ratio < 0.8:   ss = sigma * 0.5
        elif ratio < 1.5: ss = sigma * 0.2
        elif ratio < 5.0: ss = sigma * 0.3
        else:             ss = sigma * 0.5
        cp = max(cp - ss, sigma)
        steps += 1
    return steps

for cp_init_ns, label in [(100, "100ns"), (150, "150ns"), (200, "200ns"), (250, "250ns")]:
    cp_init = cp_init_ns * 1e-9
    v6 = steps_to_recover_v6(cp_init, sigma_post)
    v7 = steps_to_recover_v7(cp_init, sigma_post)
    print(f"  CP={label} → sigma=12ns:  v6 {v6:3d}스텝  v7 {v7:3d}스텝  ({v6/max(v7,1):.1f}배 단축)")

print("\n분석 완료")
