import numpy as np
import random
import math
from scipy import integrate

class AdaptiveCPEnv:
    """
    논문: "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"
    (Akgun, Krunz, Manzi, ICNC 2020)

    DS 계산 전체 흐름:
      Step 1. θ_minus(τ), θ_plus(τ) 계산          ← _theta_minus / _theta_plus
      Step 2. G(θ_minus|θ_b), G(θ_plus|θ_b) 계산  ← _antenna_gain
      Step 3. Pc(τ) 구성                           ← _pdp
      Step 4. μ_DS  = ∫τ·Pc(τ)dτ / ∫Pc(τ)dτ      ← _calculate_ds
      Step 5. σ_DS  = √(∫τ²·Pc(τ)dτ/∫Pc(τ)dτ - μ²)
    """

    def __init__(self):
        # ── 시스템 파라미터 (논문 Section III-B, Fig.3 기준) ──────────────
        self.f_options  = [28e9, 60e9]
        self.B          = 100e6            # 대역폭 100 MHz
        self.kappa_deg  = 5.0              # RMS 각도 확산 κ = 5°
        self.tau_hat    = 25e-9            # 평균 경로 지연 τ̂ = 25 ns
        self.M_options  = [16, 64, 256]
        self.rho        = 3.0              # CP scaling factor ρ = 3

        self.Ts       = 1.0 / self.B
        self.fixed_cp = self.tau_hat * self.rho   # 고정 CP (상용 시스템 기준)

        # RL 파라미터
        # State: [σ_DS(ns), SNR(dB), Δθ(°), M, current_CP(ns)]
        self.state_dim  = 5
        self.action_dim = 3   # 0: 감소, 1: 유지, 2: 증가

        self.reset()

    # =========================================================================
    # Step 1. θ_minus(τ), θ_plus(τ)
    # 논문 수식: θn(τn) = l·κ·√(τn/τ̂) + θc  (l = ±1)
    # =========================================================================
    def _theta_minus(self, tau: float) -> float:
        """l = -1 경로의 AoA [rad]: θc - κ·√(τ/τ̂)"""
        kappa_rad   = math.radians(self.kappa_deg)
        theta_c_rad = math.radians(self.theta_c_deg)
        return theta_c_rad - kappa_rad * math.sqrt(max(tau, 0.0) / self.tau_hat)

    def _theta_plus(self, tau: float) -> float:
        """l = +1 경로의 AoA [rad]: θc + κ·√(τ/τ̂)"""
        kappa_rad   = math.radians(self.kappa_deg)
        theta_c_rad = math.radians(self.theta_c_deg)
        return theta_c_rad + kappa_rad * math.sqrt(max(tau, 0.0) / self.tau_hat)

    # =========================================================================
    # Step 2. G(θn|θb) — Keyhole 안테나 이득 모델
    # 논문 수식 (4):
    #   G(θn|θb) = Gm·cos(θb)   if |θb - θn| ≤ ωb/2
    #            = gm·cos(θb)   otherwise
    # =========================================================================
    def _antenna_gain(self, theta_n_rad: float) -> float:
        """
        주어진 AoA θn에 대한 안테나 이득 반환.
        cos(θb) 항은 분자/분모에서 약분되므로 제거하고 Gm/gm만 반환.
        (논문 수식 전개 시 PL·cos(θb) 항이 약분됨을 확인)
        """
        theta_b_rad = math.radians(self.theta_b_deg)
        omega_b     = self._get_omega_b()
        G_m, g_m    = self._get_antenna_gains()

        if abs(theta_b_rad - theta_n_rad) <= omega_b / 2.0:
            return G_m   # 주엽
        else:
            return g_m   # 부엽

    # =========================================================================
    # Step 3. Pc(τ) — Power Delay Profile
    # 논문 수식 (5):
    #   Pc(τ) = PL/(2τ̂) · exp(-2τ/τ̂) · [G(θ_minus|θb) + G(θ_plus|θb)]
    #
    # l = -1, +1 각각 1/2 확률로 발생하므로 두 항을 더해 2로 나눈 형태.
    # PL은 분자/분모에서 약분되므로 PL=1로 설정.
    # =========================================================================
    def _pdp(self, tau: float) -> float:
        """
        Pc(τ): τ에서의 수신 전력 밀도

        Pc(τ) = (1 / (2·τ̂)) · exp(-2τ/τ̂)
                · [G(θ_minus(τ)|θb) + G(θ_plus(τ)|θb)]
        """
        if tau < 0:
            return 0.0

        # 지수 감쇠 전력 (PL=1 가정, 약분됨)
        power_decay = (1.0 / (2.0 * self.tau_hat)) * math.exp(-2.0 * tau / self.tau_hat)

        # l = -1, +1 각 경로의 AoA와 해당 안테나 이득
        g_minus = self._antenna_gain(self._theta_minus(tau))   # G(θ_minus|θb)
        g_plus  = self._antenna_gain(self._theta_plus(tau))    # G(θ_plus|θb)

        return power_decay * (g_minus + g_plus)

    # =========================================================================
    # Step 4 & 5. μ_DS, σ_DS 계산
    # 논문 수식 (6): μ_DS  = ∫₀^∞ τ·Pc(τ)dτ / ∫₀^∞ Pc(τ)dτ
    # 논문 수식 (7): σ_DS  = √(∫₀^∞ τ²·Pc(τ)dτ / ∫₀^∞ Pc(τ)dτ − μ_DS²)
    # =========================================================================
    def _calculate_ds(self) -> float:
        """
        수치 적분으로 μ_DS, σ_DS 계산.
        적분 상한: τ̂의 20배 (exp 감쇠로 이후 기여 무시 가능)
        """
        tau_max = self.tau_hat * 20.0

        # 분모: ∫₀^∞ Pc(τ) dτ
        denom, _ = integrate.quad(
            lambda tau: self._pdp(tau),
            0, tau_max
        )

        if denom < 1e-30:
            return self.tau_hat

        # 분자1: ∫₀^∞ τ·Pc(τ) dτ  → μ_DS
        num_mu, _ = integrate.quad(
            lambda tau: tau * self._pdp(tau),
            0, tau_max
        )
        mu_ds = num_mu / denom

        # 분자2: ∫₀^∞ τ²·Pc(τ) dτ  → σ_DS²
        num_sq, _ = integrate.quad(
            lambda tau: (tau ** 2) * self._pdp(tau),
            0, tau_max
        )
        sigma_ds_sq = num_sq / denom - mu_ds ** 2

        return math.sqrt(abs(sigma_ds_sq)) if sigma_ds_sq > 0 else self.tau_hat

    # =========================================================================
    # Monte Carlo 교차검증
    # θ_minus/θ_plus를 경로별로 직접 계산해 PDP를 생성하고 DS를 추정
    # =========================================================================
    def _calculate_ds_monte_carlo(self, num_paths: int = 2000) -> float:
        """
        각 경로에 대해 θ_minus 또는 θ_plus를 계산하고
        빔 이득 G(θn|θb)를 적용한 가중 PDP로 DS 추정.
        """
        kappa_rad   = math.radians(self.kappa_deg)
        theta_b_rad = math.radians(self.theta_b_deg)
        theta_c_rad = math.radians(self.theta_c_deg)
        omega_b     = self._get_omega_b()
        G_m, g_m    = self._get_antenna_gains()

        # 지수 분포 지연 샘플링
        delays  = np.random.exponential(self.tau_hat, num_paths)
        l_signs = np.random.choice([-1.0, 1.0], size=num_paths)

        # Step 1: θ_minus / θ_plus 계산
        theta_n = theta_c_rad + l_signs * kappa_rad * np.sqrt(delays / self.tau_hat)

        # Step 2: 안테나 이득 적용
        in_beam = np.abs(theta_b_rad - theta_n) <= omega_b / 2.0
        gains   = np.where(in_beam, G_m, g_m)

        # Step 3: Pc(τ) ∝ exp(-τ/τ̂) · G(θn|θb)
        path_powers = np.exp(-delays / self.tau_hat) * gains

        total = np.sum(path_powers)
        if total < 1e-30:
            return self.tau_hat

        # Step 4 & 5: μ_DS, σ_DS
        mu    = np.sum(delays * path_powers) / total
        sigma = np.sqrt(np.sum((delays ** 2) * path_powers) / total - mu ** 2)
        return float(sigma)

    # =========================================================================
    # UPA 파라미터 헬퍼
    # =========================================================================
    def _get_omega_b(self) -> float:
        """수평 HPBW [rad]: ωb = 1.772 / (√M · cos(θb))"""
        theta_b_rad = math.radians(self.theta_b_deg)
        cos_tb      = max(math.cos(theta_b_rad), 1e-6)
        return 1.772 / (math.sqrt(self.M) * cos_tb)

    def _get_antenna_gains(self):
        """Gm = π·M,  gm = Gm · 10^(-13.26/10)"""
        G_m = math.pi * self.M
        g_m = G_m * (10 ** (-13.26 / 10.0))
        return G_m, g_m

    # =========================================================================
    # reset
    # =========================================================================
    def reset(self, nlos_event: bool = False):
        self.f   = random.choice(self.f_options)
        self.M   = random.choice(self.M_options)
        self.snr = random.uniform(5.0, 25.0)

        self.theta_b_deg     = random.uniform(0.0, 60.0)
        self.theta_c_deg     = self.theta_b_deg
        self.delta_theta_deg = random.uniform(0.0, 5.0)

        if nlos_event:
            self.theta_b_deg     = random.uniform(50.0, 80.0)
            self.theta_c_deg     = random.uniform(0.0, 30.0)
            self.delta_theta_deg = abs(self.theta_b_deg - self.theta_c_deg)
            self.snr             = random.uniform(0.0, 10.0)

        self.sigma_ds   = self._calculate_ds()
        self.current_cp = self.sigma_ds * self.rho
        self.state      = self._make_state()
        return self.state

    # =========================================================================
    # Spectral Efficiency
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
    # =========================================================================
    def step(self, action: int, nlos_event: bool = False):
        prev_se = self._calculate_se(self.current_cp)

        step_size = 5e-9
        if action == 0:
            self.current_cp = max(0.0, self.current_cp - step_size)
        elif action == 2:
            self.current_cp += step_size

        new_se = self._calculate_se(self.current_cp)
        rssi   = self.snr - (self.current_cp / self.tau_hat)

        # 보상 함수
        reward = (new_se - prev_se) * 10.0 + rssi * 0.1

        if self.current_cp < self.sigma_ds:
            reward -= 100.0
        elif self.sigma_ds <= self.current_cp <= self.sigma_ds * 1.5:
            reward += 10.0
        elif self.current_cp > self.sigma_ds * self.rho:
            reward -= 5.0

        # 기준값
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
            "rssi"           : rssi,
            "sigma_ds"       : self.sigma_ds,
            "nlos_event"     : nlos_event,
        }

        # 채널 시변 업데이트
        if nlos_event:
            self.delta_theta_deg = min(45.0,
                self.delta_theta_deg + random.uniform(5.0, 15.0))
            self.snr = max(0.0, self.snr + random.uniform(-8.0, -3.0))
        else:
            self.delta_theta_deg = max(0.0,
                self.delta_theta_deg + random.uniform(-1.0, 1.0))
            self.snr += random.uniform(-1.0, 1.0)

        self.sigma_ds = max(self._calculate_ds(), 1e-12)
        self.state    = self._make_state()
        return self.state, reward, False, info