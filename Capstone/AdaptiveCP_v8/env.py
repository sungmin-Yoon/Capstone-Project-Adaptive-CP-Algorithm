import numpy as np
import random
import math
from scipy import integrate

class AdaptiveCPEnv:
    """
    논문: "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"
    (Akgun, Krunz, Manzi — ICNC 2020)

    ── DS 모델링 설계 원칙 ───────────────────────────────────────────────────
    Layer 1 — 3GPP TR 38.900 Table 7.5-6 UMa (채널 환경의 절대적 DS 스케일)
      LOS:  tau_hat ~ LogNormal(μ=-7.03, σ=0.66) → 논문 τ̂=25ns 기준 정규화
      NLOS: tau_hat ~ LogNormal(μ=-6.44, σ=0.39) → LOS 대비 평균 3.89배 증가
      클리핑 상한 100ns (v6 이후 유지)

    Layer 2 — 논문 keyhole closed-form (빔포밍 파라미터에 따른 DS 감소 효과)
      M↑, θb↓, Δθ↓ → σ_DS 감소  (논문 Fig.3 재현)
      최종 σ_DS = _calculate_ds() [논문 수식 (6)(7) 수치 적분]

    ── v8 변경 사항 ──────────────────────────────────────────────────────────
    [수정 1] CP 하한 강화: sigma_ds → sigma_ds × 1.1 (v7 문제 1 수정)
      - ISI 발생 원천 차단: action=0(Decrease) 이후 clamp가 σ×1.1 보장
      - SE 계산 시 ISI 페널티(×0.5) 구간에 절대 진입 불가
      - 10% 안전 마진으로 채널 추정 오차 흡수

    [수정 2] is_nlos 플래그를 state 6번째 차원으로 추가 (v7 문제 3 수정)
      - 기존: state = [σ_DS, SNR, Δθ, M, CP]  (5차원)
      - 변경: state = [σ_DS, SNR, Δθ, M, CP, is_nlos]  (6차원)
      - 효과: 에이전트가 NLOS 진입을 즉시 인식 → 선제적 CP 상향 정책 학습 가능

    [수정 3] theo_cp 재정의: σ×3.0 → σ×1.1 (v7 문제 2 수정)
      - 기존: theo_cp = σ_DS × ρ(=3.0) — ISI 안전 마진 기준, SE 최적이 아님
      - 변경: theo_cp = σ_DS × 1.1  — SE 최적 최소 CP (ISI 경계 + 10% 마진)
      - 효과: Theoretical SE가 실질적인 SE 상한이 되어 의미 있는 비교 가능

    [수정 4] 보상 클리핑 추가 (v7 문제 4 수정)
      - reward = np.clip(reward, -20.0, 20.0)
      - 효과: 이상치 보상으로 인한 학습 불안정 억제, 분산 감소

    [v7 유지] 논문 수식 및 3GPP 파라미터 절대 준수
    [v7 유지] Fixed SE 기반 보상 함수
    [v7 유지] step_size 구간 세분화 (빠른 CP 복귀)
    """

    # 3GPP TR 38.900 Table 7.5-6 UMa DS 파라미터
    THREEGPP_LOS_MU   = -7.03
    THREEGPP_LOS_SIG  =  0.66
    THREEGPP_NLOS_MU  = -6.44
    THREEGPP_NLOS_SIG =  0.39

    # 논문 기준 τ̂
    TAU_HAT_REF = 25e-9
    # 클리핑 상한 100ns
    TAU_HAT_MAX = 100e-9

    # [v8] CP 하한 안전 마진
    CP_SAFETY_MARGIN = 1.1

    # [v8] State 정규화 기준값 — 6차원 (is_nlos 추가)
    STATE_NORM = np.array([
        200.0,   # σ_DS 최대 기준 (ns)
        30.0,    # SNR 최대 (dB)
        45.0,    # Δθ 최대 (°)
        256.0,   # M 최대
        600.0,   # CP 최대 (ns)
        1.0,     # is_nlos (0 or 1) [v8 신규]
    ], dtype=np.float32)

    def __init__(self):
        self.f_options  = [28e9, 60e9]
        self.B          = 100e6
        self.kappa_deg  = 5.0
        self.M_options  = [16, 64, 256]
        self.rho        = 3.0

        self.Ts       = 1.0 / self.B
        self.fixed_cp = self.TAU_HAT_REF * self.rho   # 고정 CP = 75ns

        # [v8] state_dim 5 → 6
        self.state_dim  = 6
        self.action_dim = 3

        self.reset()

    # =========================================================================
    # 3GPP TR 38.900 기반 tau_hat 샘플링
    # =========================================================================
    def _sample_tau_hat_3gpp(self, is_nlos: bool) -> float:
        mu  = self.THREEGPP_NLOS_MU  if is_nlos else self.THREEGPP_LOS_MU
        sig = self.THREEGPP_NLOS_SIG if is_nlos else self.THREEGPP_LOS_SIG
        ds_3gpp    = 10 ** random.gauss(mu, sig)
        los_median = 10 ** self.THREEGPP_LOS_MU
        tau_hat    = ds_3gpp * (self.TAU_HAT_REF / los_median)
        return float(np.clip(tau_hat, 5e-9, self.TAU_HAT_MAX))

    # =========================================================================
    # Step 1. θ_minus(τ), θ_plus(τ)  [논문 채널 모델]
    # =========================================================================
    def _theta_minus(self, tau: float) -> float:
        kappa_rad   = math.radians(self.kappa_deg)
        theta_c_rad = math.radians(self.theta_c_deg)
        return theta_c_rad - kappa_rad * math.sqrt(max(tau, 0.0) / self.tau_hat)

    def _theta_plus(self, tau: float) -> float:
        kappa_rad   = math.radians(self.kappa_deg)
        theta_c_rad = math.radians(self.theta_c_deg)
        return theta_c_rad + kappa_rad * math.sqrt(max(tau, 0.0) / self.tau_hat)

    # =========================================================================
    # Step 2. G(θn|θb)  [논문 수식 (4)]
    # =========================================================================
    def _antenna_gain(self, theta_n_rad: float) -> float:
        theta_b_rad = math.radians(self.theta_b_deg)
        omega_b     = self._get_omega_b()
        G_m, g_m    = self._get_antenna_gains()
        return G_m if abs(theta_b_rad - theta_n_rad) <= omega_b / 2.0 else g_m

    # =========================================================================
    # Step 3. Pc(τ) — PDP  [논문 수식 (5)]
    # =========================================================================
    def _pdp(self, tau: float) -> float:
        if tau < 0:
            return 0.0
        power_decay = (1.0 / (2.0 * self.tau_hat)) * math.exp(-2.0 * tau / self.tau_hat)
        return power_decay * (
            self._antenna_gain(self._theta_minus(tau)) +
            self._antenna_gain(self._theta_plus(tau))
        )

    # =========================================================================
    # Step 4 & 5. μ_DS, σ_DS  [논문 수식 (6)(7)]
    # =========================================================================
    def _calculate_ds(self) -> float:
        tau_max  = self.tau_hat * 20.0
        denom, _ = integrate.quad(lambda t: self._pdp(t), 0, tau_max)
        if denom < 1e-30:
            return self.tau_hat
        num_mu, _   = integrate.quad(lambda t: t * self._pdp(t), 0, tau_max)
        mu_ds       = num_mu / denom
        num_sq, _   = integrate.quad(lambda t: (t**2) * self._pdp(t), 0, tau_max)
        sigma_ds_sq = num_sq / denom - mu_ds**2
        return math.sqrt(abs(sigma_ds_sq)) if sigma_ds_sq > 0 else self.tau_hat

    # =========================================================================
    # Monte Carlo 교차검증
    # =========================================================================
    def _calculate_ds_monte_carlo(self, num_paths: int = 2000) -> float:
        kappa_rad   = math.radians(self.kappa_deg)
        theta_b_rad = math.radians(self.theta_b_deg)
        theta_c_rad = math.radians(self.theta_c_deg)
        omega_b     = self._get_omega_b()
        G_m, g_m    = self._get_antenna_gains()
        delays      = np.random.exponential(self.tau_hat, num_paths)
        l_signs     = np.random.choice([-1.0, 1.0], size=num_paths)
        theta_n     = theta_c_rad + l_signs * kappa_rad * np.sqrt(delays / self.tau_hat)
        in_beam     = np.abs(theta_b_rad - theta_n) <= omega_b / 2.0
        gains       = np.where(in_beam, G_m, g_m)
        path_powers = np.exp(-delays / self.tau_hat) * gains
        total       = np.sum(path_powers)
        if total < 1e-30:
            return self.tau_hat
        mu    = np.sum(delays * path_powers) / total
        sigma = np.sqrt(np.sum((delays**2) * path_powers) / total - mu**2)
        return float(sigma)

    # =========================================================================
    # UPA 파라미터  [논문 [14] Elliott]
    # =========================================================================
    def _get_omega_b(self) -> float:
        theta_b_rad = math.radians(self.theta_b_deg)
        cos_tb      = max(math.cos(theta_b_rad), 1e-6)
        return 1.772 / (math.sqrt(self.M) * cos_tb)

    def _get_antenna_gains(self):
        G_m = math.pi * self.M
        g_m = G_m * (10 ** (-13.26 / 10.0))
        return G_m, g_m

    # =========================================================================
    # 동적 step_size (v7 유지)
    # =========================================================================
    def _get_step_size(self) -> float:
        if self.sigma_ds < 1e-12:
            return 5e-9
        ratio = self.current_cp / self.sigma_ds
        if ratio < 0.8:
            return self.sigma_ds * 0.5
        elif ratio < 1.5:
            return self.sigma_ds * 0.2
        elif ratio < 5.0:
            return self.sigma_ds * 0.3
        else:
            return self.sigma_ds * 0.5

    # =========================================================================
    # 채널 상태 설정
    # =========================================================================
    def _set_los_channel(self):
        self.snr             = random.uniform(10.0, 25.0)
        self.theta_b_deg     = random.uniform(0.0, 40.0)
        self.theta_c_deg     = self.theta_b_deg
        self.delta_theta_deg = random.uniform(0.0, 3.0)
        self.tau_hat         = self._sample_tau_hat_3gpp(is_nlos=False)
        self.is_nlos         = False

    def _set_nlos_channel(self):
        """
        NLOS 물리적 정의:
          도심 산란체 증가 → 다중 경로 수 증가 → τ̂ 증가 → σ_DS 증가
          빔 방향이 주요 AoA에서 벗어남 → 부엽 경로 포착 비율 증가 → σ_DS 추가 증가
        """
        self.snr             = random.uniform(3.0, 12.0)
        self.theta_b_deg     = random.uniform(30.0, 70.0)
        self.theta_c_deg     = random.uniform(0.0, 20.0)
        self.delta_theta_deg = abs(self.theta_b_deg - self.theta_c_deg)
        self.tau_hat         = self._sample_tau_hat_3gpp(is_nlos=True)
        self.is_nlos         = True

    # =========================================================================
    # reset
    # =========================================================================
    def reset(self, nlos_event: bool = False):
        self.f = random.choice(self.f_options)
        self.M = random.choice(self.M_options)

        if nlos_event or random.random() < 0.3:
            self._set_nlos_channel()
        else:
            self._set_los_channel()

        self.sigma_ds   = self._calculate_ds()
        self.current_cp = self.sigma_ds * self.CP_SAFETY_MARGIN
        self.state      = self._make_state()
        return self.state

    # =========================================================================
    # Spectral Efficiency  [논문 Section IV]
    # SE = (Ts / (Ts + TCP)) · log2(1 + SNR)
    # =========================================================================
    def _calculate_se(self, cp_length: float) -> float:
        snr_linear = 10 ** (self.snr / 10.0)
        se = (self.Ts / (self.Ts + cp_length)) * math.log2(1.0 + snr_linear)
        if cp_length < self.sigma_ds:
            se *= 0.5
        return se

    # =========================================================================
    # [v8 수정 2] State 벡터 — 6차원 (is_nlos 플래그 추가)
    # =========================================================================
    def _make_state(self) -> np.ndarray:
        return np.array([
            self.sigma_ds       * 1e9,
            self.snr,
            self.delta_theta_deg,
            float(self.M),
            self.current_cp     * 1e9,
            float(self.is_nlos),          # [v8 신규] NLOS 인식 플래그
        ], dtype=np.float32)

    # =========================================================================
    # step
    # [v5] NLOS 타이밍: 행동 전 채널 먼저 변경
    # [v7] 보상 기준: fixed_se
    # [v8] CP 하한 σ×1.1, theo_cp=σ×1.1, 보상 클리핑
    # =========================================================================
    def step(self, action: int, nlos_event: bool = False):

        # NLOS 급변 시 행동 전 채널 먼저 업데이트 (v5 유지)
        if nlos_event:
            self._set_nlos_channel()
            self.sigma_ds = max(self._calculate_ds(), 1e-12)

        step_size = self._get_step_size()

        if action == 0:
            self.current_cp = max(1e-9, self.current_cp - step_size)
        elif action == 2:
            self.current_cp += step_size

        # ── [v8 수정 1] CP 하한: sigma_ds × 1.1 (ISI 완전 차단) ─────────────
        # v7: max(current_cp, sigma_ds)
        # v8: max(current_cp, sigma_ds * CP_SAFETY_MARGIN)
        # → ISI 구간(cp < sigma_ds)에 물리적으로 진입 불가
        self.current_cp = max(self.current_cp, self.sigma_ds * self.CP_SAFETY_MARGIN)

        new_se   = self._calculate_se(self.current_cp)
        fixed_se = self._calculate_se(self.fixed_cp)

        # ── [v8 수정 3] theo_cp 재정의: σ×3.0 → σ×1.1 (SE 최적 기준) ────────
        # v7: theo_cp = sigma_ds × rho(=3.0)  — ISI 안전 마진 기준
        # v8: theo_cp = sigma_ds × 1.1         — SE 최적 최소 CP
        # → Theoretical SE = 실질적 SE 상한 (에이전트가 이를 추종하도록 유도)
        theo_cp = self.sigma_ds * self.CP_SAFETY_MARGIN
        theo_se = self._calculate_se(theo_cp)

        # ── 보상 함수 (v7 기반, ISI 페널티 제거 후 Sweet Spot 재조정) ──────────
        # ISI 페널티 불필요: CP 하한 σ×1.1로 ISI 진입 원천 차단됨
        reward = (new_se - fixed_se) * 20.0

        if self.sigma_ds * 1.1 <= self.current_cp <= self.sigma_ds * 2.0:
            # Sweet Spot: 최적 CP 범위 내 보너스
            reward += 3.0
        elif self.current_cp > self.sigma_ds * 3.0:
            # 과도한 CP: 선형 낭비 페널티
            overrun = (self.current_cp / (self.sigma_ds * 3.0)) - 1.0
            reward -= overrun * 5.0

        # ── [v8 수정 4] 보상 클리핑 [-20, 20] ──────────────────────────────
        reward = float(np.clip(reward, -20.0, 20.0))

        # ── 기준값 계산 ──────────────────────────────────────────────────────
        def peff(cp): return (self.Ts / (self.Ts + cp)) * 100.0

        info = {
            "theo_cp"        : theo_cp,
            "theo_se"        : theo_se,
            "theo_power_eff" : peff(theo_cp),
            "fixed_cp"       : self.fixed_cp,
            "fixed_se"       : fixed_se,
            "fixed_power_eff": peff(self.fixed_cp),
            "adaptive_cp"    : self.current_cp,
            "adaptive_se"    : new_se,
            "power_eff"      : peff(self.current_cp),
            "rssi"           : self.snr - (self.current_cp / self.tau_hat),
            "sigma_ds"       : self.sigma_ds,
            "nlos_event"     : nlos_event,
            "is_nlos"        : self.is_nlos,
        }

        # ── 정상 구간 채널 시변 업데이트 ─────────────────────────────────────
        if not nlos_event:
            tau_target   = self._sample_tau_hat_3gpp(is_nlos=False)
            self.tau_hat = float(np.clip(
                self.tau_hat * 0.6 + tau_target * 0.4,
                5e-9, self.TAU_HAT_MAX
            ))
            self.delta_theta_deg = max(0.0,
                self.delta_theta_deg + random.uniform(-0.5, 0.5))
            self.snr = max(5.0, min(30.0, self.snr + random.uniform(-1.0, 1.0)))
            self.is_nlos = False

        self.sigma_ds = max(self._calculate_ds(), 1e-12)
        self.state    = self._make_state()
        return self.state, reward, False, info
