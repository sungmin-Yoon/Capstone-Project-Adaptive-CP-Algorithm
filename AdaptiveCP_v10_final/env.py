import numpy as np
import random
import math
from scipy import integrate


class AdaptiveCPEnv:
    """
    논문: "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"
    (Akgun, Krunz, Manzi — ICNC 2020)

    ── v10 변경 사항 ──────────────────────────────────────────────────────────
    [v10] fixed_m 파라미터 추가
      - __init__(fixed_m=None): None이면 랜덤, 16/64/256이면 해당 값으로 고정
      - reset() 내 M 선택 로직에 반영
      - train.py / test.py 상단 FIXED_M 상수와 연동
    """

    THREEGPP_LOS_MU   = -7.03
    THREEGPP_LOS_SIG  =  0.66
    THREEGPP_NLOS_MU  = -6.44
    THREEGPP_NLOS_SIG =  0.39

    TAU_HAT_REF      = 25e-9
    TAU_HAT_MAX      = 100e-9
    CP_SAFETY_MARGIN = 1.1

    STATE_NORM = np.array([200.0, 30.0, 45.0, 256.0, 600.0, 1.0], dtype=np.float32)

    def __init__(self, fixed_m=None):
        self.f_options  = [28e9, 60e9]
        self.B          = 100e6
        self.kappa_deg  = 5.0
        self.M_options  = [16, 64, 256]
        # [v10] 안테나 배열 선택 설정
        # None  → reset() 마다 M_options 에서 랜덤 선택
        # 16/64/256 → 항상 해당 값으로 고정
        self.fixed_m    = fixed_m
        self.rho        = 3.0

        self.Ts       = 1.0 / self.B
        self.fixed_cp = self.TAU_HAT_REF * self.rho

        self.state_dim  = 6
        self.action_dim = 3

        self.reset()

    def _sample_tau_hat_3gpp(self, is_nlos: bool) -> float:
        mu  = self.THREEGPP_NLOS_MU  if is_nlos else self.THREEGPP_LOS_MU
        sig = self.THREEGPP_NLOS_SIG if is_nlos else self.THREEGPP_LOS_SIG
        ds_3gpp    = 10 ** random.gauss(mu, sig)
        los_median = 10 ** self.THREEGPP_LOS_MU
        tau_hat    = ds_3gpp * (self.TAU_HAT_REF / los_median)
        return float(np.clip(tau_hat, 5e-9, self.TAU_HAT_MAX))

    def _theta_minus(self, tau: float) -> float:
        return math.radians(self.theta_c_deg) - math.radians(self.kappa_deg) * math.sqrt(max(tau, 0.0) / self.tau_hat)

    def _theta_plus(self, tau: float) -> float:
        return math.radians(self.theta_c_deg) + math.radians(self.kappa_deg) * math.sqrt(max(tau, 0.0) / self.tau_hat)

    def _antenna_gain(self, theta_n_rad: float) -> float:
        theta_b_rad = math.radians(self.theta_b_deg)
        omega_b     = self._get_omega_b()
        G_m, g_m    = self._get_antenna_gains()
        return G_m if abs(theta_b_rad - theta_n_rad) <= omega_b / 2.0 else g_m

    def _pdp(self, tau: float) -> float:
        if tau < 0:
            return 0.0
        power_decay = (1.0 / (2.0 * self.tau_hat)) * math.exp(-2.0 * tau / self.tau_hat)
        return power_decay * (
            self._antenna_gain(self._theta_minus(tau)) +
            self._antenna_gain(self._theta_plus(tau))
        )

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

    def _get_omega_b(self) -> float:
        cos_tb = max(math.cos(math.radians(self.theta_b_deg)), 1e-6)
        return 1.772 / (math.sqrt(self.M) * cos_tb)

    def _get_antenna_gains(self):
        G_m = math.pi * self.M
        g_m = G_m * (10 ** (-13.26 / 10.0))
        return G_m, g_m

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

    def _set_los_channel(self):
        self.snr             = random.uniform(10.0, 25.0)
        self.theta_b_deg     = random.uniform(0.0, 40.0)
        self.theta_c_deg     = self.theta_b_deg
        self.delta_theta_deg = 0.0
        self.tau_hat         = self._sample_tau_hat_3gpp(is_nlos=False)
        self.is_nlos         = False

    def _set_nlos_channel(self):
        self.snr             = random.uniform(3.0, 12.0)
        self.theta_b_deg     = random.uniform(30.0, 70.0)
        self.theta_c_deg     = random.uniform(0.0, 20.0)
        self.delta_theta_deg = abs(self.theta_b_deg - self.theta_c_deg)
        self.tau_hat         = self._sample_tau_hat_3gpp(is_nlos=True)
        self.is_nlos         = True

    def reset(self, nlos_event: bool = False):
        self.f = random.choice(self.f_options)
        # [v10] 안테나 배열 선택
        if self.fixed_m is not None:
            self.M = self.fixed_m
        else:
            self.M = random.choice(self.M_options)

        if nlos_event or random.random() < 0.3:
            self._set_nlos_channel()
        else:
            self._set_los_channel()

        self.sigma_ds   = self._calculate_ds()
        self.current_cp = self.sigma_ds * self.CP_SAFETY_MARGIN
        self.state      = self._make_state()
        return self.state

    def _calculate_se(self, cp_length: float) -> float:
        snr_linear = 10 ** (self.snr / 10.0)
        se = (self.Ts / (self.Ts + cp_length)) * math.log2(1.0 + snr_linear)
        if cp_length < self.sigma_ds:
            se *= 0.5
        return se

    def _make_state(self) -> np.ndarray:
        return np.array([
            self.sigma_ds       * 1e9,
            self.snr,
            self.delta_theta_deg,
            float(self.M),
            self.current_cp     * 1e9,
            float(self.is_nlos),
        ], dtype=np.float32)

    def step(self, action: int, nlos_event: bool = False):
        if nlos_event:
            self._set_nlos_channel()
            self.sigma_ds = max(self._calculate_ds(), 1e-12)

        step_size = self._get_step_size()
        if action == 0:
            self.current_cp = max(1e-9, self.current_cp - step_size)
        elif action == 2:
            self.current_cp += step_size

        self.current_cp = max(self.current_cp, self.sigma_ds * self.CP_SAFETY_MARGIN)

        new_se   = self._calculate_se(self.current_cp)
        fixed_se = self._calculate_se(self.fixed_cp)
        theo_cp  = self.sigma_ds * self.rho
        theo_se  = self._calculate_se(theo_cp)

        reward = (new_se - fixed_se) * 20.0
        if self.sigma_ds * 1.1 <= self.current_cp <= self.sigma_ds * 2.0:
            reward += 3.0
        elif self.current_cp > self.sigma_ds * 3.0:
            overrun = (self.current_cp / (self.sigma_ds * 3.0)) - 1.0
            reward -= overrun * 5.0
        reward = float(np.clip(reward, -20.0, 20.0))

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

        if not nlos_event:
            tau_target   = self._sample_tau_hat_3gpp(is_nlos=False)
            self.tau_hat = float(np.clip(
                self.tau_hat * 0.6 + tau_target * 0.4,
                5e-9, self.TAU_HAT_MAX
            ))
            self.delta_theta_deg = abs(self.theta_b_deg - self.theta_c_deg)
            self.snr = max(5.0, min(30.0, self.snr + random.uniform(-1.0, 1.0)))
            self.is_nlos = False

        self.sigma_ds = max(self._calculate_ds(), 1e-12)
        self.state    = self._make_state()
        return self.state, reward, False, info
