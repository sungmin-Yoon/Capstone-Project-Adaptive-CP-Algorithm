import numpy as np
import random
import math

class AdaptiveCPEnv:
    def __init__(self):
        # 시스템 파라미터 기본값 (논문 기준)
        self.f_options = [28e9, 60e9]
        self.B = 100e6
        self.kappa = 5.0
        self.tau_hat = 25e-9
        self.M_options = [16, 64, 256]
        
        self.Ts = 1 / self.B
        self.current_cp = self.tau_hat * 3 # 초기 CP 설정
        
        # RL Parameter
        self.state_dim = 5
        self.action_dim = 3 # 0: 감소, 1: 유지, 2: 증가
        
        self.reset()

    def reset(self):
        self.f = random.choice(self.f_options)
        self.M = random.choice(self.M_options)
        self.snr = random.uniform(5.0, 25.0)
        self.theta_c = random.uniform(0, 60)
        
        # DS 교차 검증 (Cross-validation)
        ds_cf = self._calculate_ds_closed_form()
        ds_mc = self._calculate_ds_monte_carlo()
        
        self.sigma_ds = ds_cf # 실제 환경 State로는 closed-form 사용
        self.current_cp = self.sigma_ds * 3
        
        # State 정규화: 1e9를 곱해 나노초(ns) 스케일로 변환
        self.state = np.array([
            self.sigma_ds * 1e9, 
            self.snr, 
            self.theta_c, 
            self.M, 
            self.current_cp * 1e9
        ], dtype=np.float32)
        
        return self.state

    def _calculate_ds_closed_form(self):
        # 정사각 배열(Uniform Planar Array) 특성 반영
        sqrt_M = math.sqrt(self.M) 
        
        # 안테나 빔폭 (omega_b) 계산 (라디안)
        omega_b = 1.772 / sqrt_M 
        
        # 안테나 이득 계산
        G_m = math.pi * self.M
        g_m = G_m * (10 ** (-13.26 / 10)) 
        
        # [🛠️버그 수정] kappa(5.0도)를 라디안으로 변환하여 단위 통일
        kappa_rad = math.radians(self.kappa)
        
        # 빔폭(omega_b) 감소에 따른 캡처율 축소 효과 반영 (단위 일치)
        a1 = self.tau_hat * ((omega_b / 2)**2) / (kappa_rad**2) 
        
        num = 0.25 * (G_m - g_m) * (self.tau_hat**2) * (1 - math.exp(-2 * a1 / self.tau_hat)) 
        den = 0.5 * G_m 
        
        mu_ds = num / den if den != 0 else self.tau_hat
        sigma_ds_sq = (num * 2 / den) - (mu_ds**2)
        
        return math.sqrt(abs(sigma_ds_sq)) if sigma_ds_sq > 0 else self.tau_hat

    def _calculate_ds_monte_carlo(self, num_paths=1000):
        # Monte Carlo 시뮬레이션 기반 난수 생성 및 PDP 도출
        delays = np.random.exponential(self.tau_hat, num_paths)
        powers = np.exp(-delays / self.tau_hat)
        
        sum_power = np.sum(powers)
        mu_ds_mc = np.sum(delays * powers) / sum_power
        sigma_ds_mc = np.sqrt(np.sum((delays**2) * powers) / sum_power - mu_ds_mc**2)
        return sigma_ds_mc

    def _calculate_se(self, cp_length):
        # Spectral Efficiency (SE) 물리적 수식
        snr_linear = 10 ** (self.snr / 10)
        se = (self.Ts / (self.Ts + cp_length)) * math.log2(1 + snr_linear)
        
        # 물리적 ISI 통신 성능 저하 반영
        if cp_length < self.sigma_ds:
            se *= 0.5 
        return se

    def step(self, action):
        prev_se = self._calculate_se(self.current_cp)
        
        step_size = 5e-9 
        if action == 0:
            self.current_cp = max(0, self.current_cp - step_size)
        elif action == 2:
            self.current_cp += step_size
            
        new_se = self._calculate_se(self.current_cp)
        rssi = self.snr - (self.current_cp / self.tau_hat) 
        
        # 기본 보상: SE 개선분 + RSSI 가중치
        se_improvement = new_se - prev_se
        reward = (se_improvement * 10) + (rssi * 0.1)
        
        # RL 행동 훈련용 추가 보상/페널티 체계 도입
        if self.current_cp < self.sigma_ds:
            reward -= 100.0 # ISI 페널티
        elif self.sigma_ds <= self.current_cp <= self.sigma_ds * 1.5:
            reward += 10.0  # 스윗스팟 보너스
        elif self.current_cp > self.sigma_ds * 3.0:
            reward -= 5.0   # 낭비 페널티
            
        # 이론적 지표 연산
        theo_cp = self.sigma_ds * 3.0
        theo_se = self._calculate_se(theo_cp)
        
        theo_power_eff = (self.Ts / (self.Ts + theo_cp)) * 100 
        power_eff = (self.Ts / (self.Ts + self.current_cp)) * 100 
        
        info = {
            "theo_cp": theo_cp,
            "theo_se": theo_se,
            "theo_power_eff": theo_power_eff,
            "adaptive_cp": self.current_cp,
            "adaptive_se": new_se,
            "rssi": rssi,
            "power_eff": power_eff
        }
        
        # State 업데이트
        self.sigma_ds *= random.uniform(0.9, 1.1)
        self.snr += random.uniform(-1, 1)
        
        self.state = np.array([
            self.sigma_ds * 1e9, 
            self.snr, 
            self.theta_c, 
            self.M, 
            self.current_cp * 1e9
        ], dtype=np.float32)
        
        done = False 
        return self.state, reward, done, info