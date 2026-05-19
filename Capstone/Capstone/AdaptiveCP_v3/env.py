import numpy as np
import random
import math
from scipy import integrate

class AdaptiveCPEnv:
    """
    논문: "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"
    (Akgun, Krunz, Manzi, ICNC 2020)

    NLOS 표현 방식:
      keyhole 모델의 DS는 tau_hat/2 = 12.5ns에서 포화됨.
      실제 NLOS 환경(도심 산란체 급증)은 tau_hat 자체가 커지는 현상으로 모델링.
      - 정상: tau_hat = 25ns (논문 기준값)
      - NLOS: tau_hat = 50~150ns (다중 산란으로 평균 지연 증가)
      이렇게 하면 DS가 논문 Fig.3 수준(수십 ns)으로 변화하여
      고정 CP(75ns)와 Adaptive CP의 차이가 명확히 드러남.
    """

    def __init__(self):
        self.f_options      = [28e9, 60e9]
        self.B              = 100e6
        self.kappa_deg      = 5.0
        self.tau_hat_normal = 25e-9    # 정상 환경 기준 τ̂
        self.M_options      = [16, 64, 256]
        self.rho            = 3.0

        self.Ts       = 1.0 / self.B
        self.fixed_cp = self.tau_hat_normal * self.rho  # 고정 CP = 75ns (정상 기준)

        self.state_dim  = 5
        self.action_dim = 3

        self.reset()

    # =========================================================================
    # Step 1. θ_minus(τ), θ_plus(τ)
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
    # Step 2. G(θn|θb) — Keyhole 안테나 이득
    # =========================================================================
    def _antenna_gain(self, theta_n_rad: float) -> float:
        theta_b_rad = math.radians(self.theta_b_deg)
        omega_b     = self._get_omega_b()
        G_m, g_m    = self._get_antenna_gains()
        if abs(theta_b_rad - theta_n_rad) <= omega_b / 2.0:
            return G_m
        else:
            return g_m

    # =========================================================================
    # Step 3. Pc(τ) — Power Delay Profile
    # Pc(τ) = 1/(2τ̂) · exp(-2τ/τ̂) · [G(θ_minus|θb) + G(θ_plus|θb)]
    # =========================================================================
    def _pdp(self, tau: float) -> float:
        if tau < 0:
            return 0.0
        power_decay = (1.0 / (2.0 * self.tau_hat)) * math.exp(-2.0 * tau / self.tau_hat)
        g_minus = self._antenna_gain(self._theta_minus(tau))
        g_plus  = self._antenna_gain(self._theta_plus(tau))
        return power_decay * (g_minus + g_plus)

    # =========================================================================
    # Step 4 & 5. μ_DS, σ_DS (수치 적분)
    # μ_DS = ∫τ·Pc(τ)dτ / ∫Pc(τ)dτ
    # σ_DS = √(∫τ²·Pc(τ)dτ / ∫Pc(τ)dτ - μ_DS²)
    # =========================================================================
    def _calculate_ds(self) -> float:
        tau_max = self.tau_hat * 20.0

        denom, _ = integrate.quad(lambda t: self._pdp(t), 0, tau_max)
        if denom < 1e-30:
            return self.tau_hat

        num_mu, _ = integrate.quad(lambda t: t * self._pdp(t), 0, tau_max)
        mu_ds = num_mu / denom

        num_sq, _ = integrate.quad(lambda t: (t**2) * self._pdp(t), 0, tau_max)
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

        delays  = np.random.exponential(self.tau_hat, num_paths)
        l_signs = np.random.choice([-1.0, 1.0], size=num_paths)
        theta_n = theta_c_rad + l_signs * kappa_rad * np.sqrt(delays / self.tau_hat)

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
    # UPA 파라미터
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
    # reset
    # =========================================================================
    def reset(self, nlos_event: bool = False):
        self.f   = random.choice(self.f_options)
        self.M   = random.choice(self.M_options)
        self.snr = random.uniform(10.0, 25.0)

        # 정상 환경
        self.tau_hat         = self.tau_hat_normal
        self.theta_b_deg     = random.uniform(0.0, 40.0)
        self.theta_c_deg     = self.theta_b_deg
        self.delta_theta_deg = random.uniform(0.0, 3.0)

        if nlos_event:
            self._apply_nlos()

        self.sigma_ds   = self._calculate_ds()
        self.current_cp = self.sigma_ds * self.rho
        self.state      = self._make_state()
        return self.state

    def _apply_nlos(self):
        """
        NLOS 급변 모델링:
        - tau_hat 증가: 다중 산란으로 평균 경로 지연 급증 → DS 직접 증가
        - SNR 저하: 장애물로 인한 신호 감쇠
        - 빔 오정렬 증가: 이동체가 빔 방향에서 벗어남
        """
        self.tau_hat         = self.tau_hat_normal * random.uniform(3.0, 6.0)
        self.snr             = random.uniform(3.0, 8.0)
        self.theta_b_deg     = random.uniform(40.0, 70.0)
        self.theta_c_deg     = random.uniform(0.0, 20.0)
        self.delta_theta_deg = abs(self.theta_b_deg - self.theta_c_deg)

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
            self.current_cp = max(1e-9, self.current_cp - step_size)
        elif action == 2:
            self.current_cp += step_size

        new_se = self._calculate_se(self.current_cp)

        # ── 보상 함수 ────────────────────────────────────────────────────
        # SE 절대값 기반: 에이전트가 SE를 최대화하도록 유도
        reward = new_se * 10.0

        # ISI 발생: CP < σ_DS → 강한 페널티
        if self.current_cp < self.sigma_ds:
            reward -= 20.0

        # Sweet Spot: σ_DS ≤ CP ≤ 1.5×σ_DS → 보너스
        elif self.sigma_ds <= self.current_cp <= self.sigma_ds * 1.5:
            reward += 5.0

        # 낭비 구간: CP > ρ×σ_DS → 페널티
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
        }

        # ── 채널 시변 업데이트 ───────────────────────────────────────────
        if nlos_event:
            self._apply_nlos()
        else:
            # 정상: tau_hat 복원 + 소폭 변동
            self.tau_hat = max(
                self.tau_hat_normal,
                self.tau_hat * random.uniform(0.7, 0.9)  # NLOS 후 점진적 복원
            )
            self.delta_theta_deg = max(0.0,
                self.delta_theta_deg + random.uniform(-0.5, 0.5))
            self.snr = max(5.0, min(30.0, self.snr + random.uniform(-1.0, 1.0)))

        self.sigma_ds = max(self._calculate_ds(), 1e-12)
        self.state    = self._make_state()
        return self.state, reward, False, info