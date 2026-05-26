import numpy as np
import random
import math
from scipy import integrate


class AdaptiveCPEnv:
    """
    논문: "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"
    (Akgun, Krunz, Manzi — ICNC 2020)

    ── v11 변경 사항 ──────────────────────────────────────────────────────────
    [v11] SINR / BER / No-CP SE 지표 추가
      - _compute_sinr_ber(cp_length) → (sinr_linear, sinr_db, ber, log10_ber)
        · ISI 에너지 분율: 지수 PDP 가정 → exp(−CP / τ_hat)
        · SINR = SNR_linear / (1 + SNR_linear × ISI_fraction)
        · BER  = 0.5 × erfc(√SINR)  [BPSK 근사]
      - step() info 에 아래 4가지 CP 케이스 전부 SINR·BER 추가:
          Adaptive CP   → sinr_adapt_db,  ber_adapt,  log_ber_adapt
          Fixed CP      → sinr_fixed_db,  ber_fixed,  log_ber_fixed
          Theoretical   → sinr_theo_db,   ber_theo,   log_ber_theo
          No CP (CP=0)  → sinr_no_cp_db,  ber_no_cp,  log_ber_no_cp
      - no_cp_se: CP=0 일 때 SE (논문 핵심 포인트 시각화용)
      - snr: info 에 명시적 추가

    [v10] fixed_m 파라미터 추가
      - __init__(fixed_m=None): None이면 랜덤, 16/64/256이면 해당 값으로 고정
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
        self.fixed_m    = fixed_m
        self.rho        = 3.0

        self.Ts       = 1.0 / self.B
        self.fixed_cp = self.TAU_HAT_REF * self.rho

        self.state_dim  = 6
        self.action_dim = 3

        self.reset()

    # ── 채널 샘플링 ──────────────────────────────────────────────────────────

    def _sample_tau_hat_3gpp(self, is_nlos: bool) -> float:
        mu  = self.THREEGPP_NLOS_MU  if is_nlos else self.THREEGPP_LOS_MU
        sig = self.THREEGPP_NLOS_SIG if is_nlos else self.THREEGPP_LOS_SIG
        ds_3gpp    = 10 ** random.gauss(mu, sig)
        los_median = 10 ** self.THREEGPP_LOS_MU
        tau_hat    = ds_3gpp * (self.TAU_HAT_REF / los_median)
        return float(np.clip(tau_hat, 5e-9, self.TAU_HAT_MAX))

    # ── 빔 / 안테나 모델 ─────────────────────────────────────────────────────

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

    # ── SE 계산 ──────────────────────────────────────────────────────────────

    def _calculate_se(self, cp_length: float) -> float:
        """
        SE = (Ts / (Ts + CP)) × log2(1 + SNR)
        CP < sigma_ds 이면 ISI 패널티 × 0.5 적용 (단순화 모델)
        CP = 0 이면: SE = log2(1 + SNR) × 0.5
          → 오버헤드 없음 + ISI 패널티 → 논문 핵심 포인트 시각화용
        """
        snr_linear = 10 ** (self.snr / 10.0)
        se = (self.Ts / (self.Ts + cp_length)) * math.log2(1.0 + snr_linear)
        if cp_length < self.sigma_ds:
            se *= 0.5
        return se

    # ── [v11] SINR / BER 계산 ────────────────────────────────────────────────

    def _compute_sinr_ber(self, cp_length: float):
        """
        CP 길이가 cp_length 일 때의 SINR과 BER을 계산한다.

        ISI 에너지 분율 (지수 PDP 가정):
          ∫_CP^∞ (1/τ_hat)·exp(−τ/τ_hat)dτ = exp(−CP/τ_hat)
          CP = 0 → ISI_fraction = 1.0 (전 멀티패스 에너지가 ISI)
          CP ≫ τ_hat → ISI_fraction → 0 (ISI 거의 없음)

        SINR = SNR_linear / (1 + SNR_linear × ISI_fraction)

        BER (BPSK 근사):
          BER = 0.5 × erfc(√SINR)
        """
        snr_linear = 10 ** (self.snr / 10.0)

        if self.tau_hat > 1e-15:
            isi_fraction = math.exp(-max(cp_length, 0.0) / self.tau_hat)
        else:
            isi_fraction = 0.0

        isi_power   = snr_linear * isi_fraction
        sinr_linear = snr_linear / (1.0 + isi_power)
        sinr_db     = 10.0 * math.log10(max(sinr_linear, 1e-20))

        ber     = 0.5 * math.erfc(math.sqrt(max(sinr_linear, 1e-20)))
        ber     = max(ber, 1e-15)
        log_ber = math.log10(ber)

        return sinr_linear, sinr_db, ber, log_ber

    # ── 상태 벡터 ─────────────────────────────────────────────────────────────

    def _make_state(self) -> np.ndarray:
        return np.array([
            self.sigma_ds       * 1e9,
            self.snr,
            self.delta_theta_deg,
            float(self.M),
            self.current_cp     * 1e9,
            float(self.is_nlos),
        ], dtype=np.float32)

    # ── 스텝 ─────────────────────────────────────────────────────────────────

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

        # [v11] No-CP SE (CP = 0)
        # SE = log2(1+SNR) × 0.5  (오버헤드 없음, ISI 패널티 적용)
        # → 논문 핵심 포인트: 고정 CP 대비 No-CP SE가 높은 경우가 있음을 시각화
        no_cp_se = self._calculate_se(0.0)

        reward = (new_se - fixed_se) * 20.0
        if self.sigma_ds * 1.1 <= self.current_cp <= self.sigma_ds * 2.0:
            reward += 3.0
        elif self.current_cp > self.sigma_ds * 3.0:
            overrun = (self.current_cp / (self.sigma_ds * 3.0)) - 1.0
            reward -= overrun * 5.0
        reward = float(np.clip(reward, -20.0, 20.0))

        def peff(cp): return (self.Ts / (self.Ts + cp)) * 100.0

        # [v11] 4가지 CP 케이스 SINR / BER
        sinr_a, sinr_a_db, ber_a, lber_a = self._compute_sinr_ber(self.current_cp)
        sinr_f, sinr_f_db, ber_f, lber_f = self._compute_sinr_ber(self.fixed_cp)
        sinr_t, sinr_t_db, ber_t, lber_t = self._compute_sinr_ber(theo_cp)
        sinr_n, sinr_n_db, ber_n, lber_n = self._compute_sinr_ber(0.0)

        info = {
            # ── 기존 지표 ──────────────────────────────────────────────────
            "snr"            : self.snr,
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
            # ── [v11] No-CP SE ─────────────────────────────────────────────
            "no_cp_se"       : no_cp_se,
            # ── [v11] SINR ─────────────────────────────────────────────────
            "sinr_adapt_db"  : sinr_a_db,
            "sinr_fixed_db"  : sinr_f_db,
            "sinr_theo_db"   : sinr_t_db,
            "sinr_no_cp_db"  : sinr_n_db,
            # ── [v11] BER ──────────────────────────────────────────────────
            "ber_adapt"      : ber_a,   "log_ber_adapt"  : lber_a,
            "ber_fixed"      : ber_f,   "log_ber_fixed"  : lber_f,
            "ber_theo"       : ber_t,   "log_ber_theo"   : lber_t,
            "ber_no_cp"      : ber_n,   "log_ber_no_cp"  : lber_n,
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
