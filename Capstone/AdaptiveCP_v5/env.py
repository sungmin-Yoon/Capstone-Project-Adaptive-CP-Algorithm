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

    Layer 2 — 논문 keyhole closed-form (빔포밍 파라미터에 따른 DS 감소 효과)
      M↑, θb↓, Δθ↓ → σ_DS 감소  (논문 Fig.3 재현)
      최종 σ_DS = _calculate_ds() [논문 수식 (6)(7) 수치 적분]

    ── v5 변경 사항 ──────────────────────────────────────────────────────────
    1. NLOS 타이밍 수정: step() 진입 즉시 채널 변경 후 DS 재계산
       → 에이전트가 해당 스텝에서 변화된 DS를 즉시 관측 (1스텝 지연 제거)
    2. State 정규화 기준값(STATE_NORM) 상수 제공 → DQN 입력 정규화에 사용
    3. is_nlos 플래그 info에 명시적 포함
    """

    # 3GPP TR 38.900 Table 7.5-6 UMa DS 파라미터
    THREEGPP_LOS_MU   = -7.03
    THREEGPP_LOS_SIG  =  0.66
    THREEGPP_NLOS_MU  = -6.44
    THREEGPP_NLOS_SIG =  0.39

    # 논문 기준 τ̂ (3GPP DS를 논문 스케일로 정규화하는 기준점)
    TAU_HAT_REF = 25e-9

    # State 정규화 기준값 (DQN 입력 정규화용)
    STATE_NORM = np.array([
        200.0,   # σ_DS 최대 기준 (ns)
        30.0,    # SNR 최대 (dB)
        45.0,    # Δθ 최대 (°)
        256.0,   # M 최대
        600.0,   # CP 최대 (ns)
    ], dtype=np.float32)

    def __init__(self):
        self.f_options  = [28e9, 60e9]
        self.B          = 100e6
        self.kappa_deg  = 5.0
        self.M_options  = [16, 64, 256]
        self.rho        = 3.0

        self.Ts       = 1.0 / self.B
        self.fixed_cp = self.TAU_HAT_REF * self.rho   # 고정 CP = 75ns

        self.state_dim  = 5
        self.action_dim = 3   # 0: 감소, 1: 유지, 2: 증가

        self.reset()

    # =========================================================================
    # 3GPP TR 38.900 기반 tau_hat 샘플링
    # =========================================================================
    def _sample_tau_hat_3gpp(self, is_nlos: bool) -> float:
        """
        3GPP TR 38.900 Table 7.5-6 UMa DS log-normal 분포에서 tau_hat 샘플링.
        LOS 중앙값(10^-7.03)을 논문 기준 τ̂_ref(25ns)에 매핑하여 정규화.
        """
        mu  = self.THREEGPP_NLOS_MU  if is_nlos else self.THREEGPP_LOS_MU
        sig = self.THREEGPP_NLOS_SIG if is_nlos else self.THREEGPP_LOS_SIG

        ds_3gpp    = 10 ** random.gauss(mu, sig)
        los_median = 10 ** self.THREEGPP_LOS_MU
        tau_hat    = ds_3gpp * (self.TAU_HAT_REF / los_median)
        return float(np.clip(tau_hat, 5e-9, 500e-9))

    # =========================================================================
    # Step 1. θ_minus(τ), θ_plus(τ)  [논문 채널 모델]
    # θn(τn) = l·κ·√(τn/τ̂) + θc  (l = ±1)
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
    # Step 2. G(θn|θb) — Keyhole 안테나 이득  [논문 수식 (4)]
    # G = Gm if |θb-θn| ≤ ωb/2,  gm otherwise
    # =========================================================================
    def _antenna_gain(self, theta_n_rad: float) -> float:
        theta_b_rad = math.radians(self.theta_b_deg)
        omega_b     = self._get_omega_b()
        G_m, g_m    = self._get_antenna_gains()
        return G_m if abs(theta_b_rad - theta_n_rad) <= omega_b / 2.0 else g_m

    # =========================================================================
    # Step 3. Pc(τ) — PDP  [논문 수식 (5)]
    # Pc(τ) = 1/(2τ̂) · exp(-2τ/τ̂) · [G(θ_minus|θb) + G(θ_plus|θb)]
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
    # μ_DS = ∫τ·Pc(τ)dτ / ∫Pc(τ)dτ
    # σ_DS = √(∫τ²·Pc(τ)dτ / ∫Pc(τ)dτ − μ²)
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
    # 동적 step_size
    # CP/σ_DS 비율에 따라 조정 속도 자동 전환
    # =========================================================================
    def _get_step_size(self) -> float:
        if self.sigma_ds < 1e-12:
            return 5e-9
        ratio = self.current_cp / self.sigma_ds
        if ratio < 0.8:
            return self.sigma_ds * 0.5    # ISI 위험: 빠르게 증가
        elif ratio < 1.5:
            return self.sigma_ds * 0.2    # σ_DS 근처: 중간 속도
        else:
            return max(self.sigma_ds * 0.05, 1e-9)   # 정밀 조정

    # =========================================================================
    # 채널 상태 설정 헬퍼
    # =========================================================================
    def _set_los_channel(self):
        self.snr             = random.uniform(10.0, 25.0)
        self.theta_b_deg     = random.uniform(0.0, 40.0)
        self.theta_c_deg     = self.theta_b_deg
        self.delta_theta_deg = random.uniform(0.0, 3.0)
        self.tau_hat         = self._sample_tau_hat_3gpp(is_nlos=False)
        self.is_nlos         = False

    def _set_nlos_channel(self):
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

        # 30% 확률로 NLOS 상태로 시작 → NLOS 대응 정책 학습 강화
        if nlos_event or random.random() < 0.3:
            self._set_nlos_channel()
        else:
            self._set_los_channel()

        self.sigma_ds   = self._calculate_ds()
        self.current_cp = self.sigma_ds * self.rho
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
    # State 벡터
    # =========================================================================
    def _make_state(self) -> np.ndarray:
        return np.array([
            self.sigma_ds       * 1e9,
            self.snr,
            self.delta_theta_deg,
            float(self.M),
            self.current_cp     * 1e9
        ], dtype=np.float32)

    # =========================================================================
    # step
    # [v5] NLOS 타이밍 수정: 행동 결정 전에 채널 상태 먼저 변경
    #      → 에이전트가 해당 스텝에서 변화된 DS를 즉시 관측
    # =========================================================================
    def step(self, action: int, nlos_event: bool = False):

        # [v5] NLOS 급변 시 행동 전에 채널 먼저 업데이트
        if nlos_event:
            self._set_nlos_channel()
            self.sigma_ds = max(self._calculate_ds(), 1e-12)

        prev_se   = self._calculate_se(self.current_cp)
        step_size = self._get_step_size()

        if action == 0:
            self.current_cp = max(1e-9, self.current_cp - step_size)
        elif action == 2:
            self.current_cp += step_size

        # CP 하한 보장: CP >= sigma_ds
        self.current_cp = max(self.current_cp, self.sigma_ds)

        new_se = self._calculate_se(self.current_cp)

        # ── 보상 함수 ────────────────────────────────────────────────────
        reward = new_se * 10.0

        if self.current_cp < self.sigma_ds:
            reward -= 20.0
        elif self.sigma_ds <= self.current_cp <= self.sigma_ds * 1.5:
            reward += 5.0
        elif self.current_cp > self.sigma_ds * self.rho:
            reward -= 2.0

        # ── 기준값 계산 ──────────────────────────────────────────────────
        theo_cp  = self.sigma_ds * self.rho
        theo_se  = self._calculate_se(theo_cp)
        fixed_se = self._calculate_se(self.fixed_cp)

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

        # ── 정상 구간 채널 시변 업데이트 ─────────────────────────────────
        if not nlos_event:
            # 3GPP LOS 분포로 점진 복원
            tau_target   = self._sample_tau_hat_3gpp(is_nlos=False)
            self.tau_hat = float(np.clip(
                self.tau_hat * 0.6 + tau_target * 0.4, 5e-9, 500e-9))
            self.delta_theta_deg = max(0.0,
                self.delta_theta_deg + random.uniform(-0.5, 0.5))
            self.snr = max(5.0, min(30.0, self.snr + random.uniform(-1.0, 1.0)))
            self.is_nlos = False

        self.sigma_ds = max(self._calculate_ds(), 1e-12)
        self.state    = self._make_state()
        return self.state, reward, False, info
