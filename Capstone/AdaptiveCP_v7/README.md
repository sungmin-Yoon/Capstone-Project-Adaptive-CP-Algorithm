# AdaptiveCP — DQN 기반 적응형 CP 조절 알고리즘

> **논문 기반:** "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"  
> Akgun, Krunz, Manzi — ICNC 2020

---

## 프로젝트 개요

NLOS 환경에서 고속 이동체의 통신 연속성 유지를 위해, DQN 기반 강화학습을 활용하여  
채널의 DS(Delay Spread) 추정치에 따라 CP(Cyclic Prefix) 길이를 동적으로 조절하는 알고리즘 연구.

### 비교 대상 (3-way)
| 방식 | 설명 |
|---|---|
| Fixed CP | 상용 시스템 기준 고정값 (`τ̂ × ρ = 75ns`) |
| Theoretical CP | 매 스텝 실제 σ_DS 기반 최적값 (`σ_DS × ρ`) |
| Adaptive CP (DQN) | 강화학습 에이전트가 동적으로 결정 |

---

## 파일 구조

```
AdaptiveCP/
├── env.py                     # 강화학습 환경 (채널 모델, PDP, DS 계산, 보상 함수)
├── DQN.py                     # 신경망 구조 (3층 FC, 입력 정규화 포함)
├── train.py                   # 학습 루프 (ε-greedy, Experience Replay, Target Network)
├── test.py                    # 평가 및 대시보드 시각화 (3×3 서브플롯)
├── dqn_adaptive_cp.pth        # 학습된 모델 가중치 (train 후 생성)
├── train_rewards.npy          # 에피소드별 보상 기록 (train 후 생성)
└── adaptive_cp_dashboard.png  # 결과 대시보드 (test 후 생성)
```

---

## 실행 방법

```powershell
py -3.11 train.py
py -3.11 test.py
```

---

## DS 계산 흐름 (env.py)

논문 수식을 단계별로 명시적으로 구현.

```
Step 1.  θ_minus(τ) = θc - κ·√(τ/τ̂)      ← l = -1 경로 AoA  [논문 채널 모델]
         θ_plus(τ)  = θc + κ·√(τ/τ̂)      ← l = +1 경로 AoA

Step 2.  G(θn|θb) = Gm  if |θb - θn| ≤ ωb/2    ← 주엽  [논문 수식 (4)]
                  = gm  otherwise                 ← 부엽

Step 3.  Pc(τ) = 1/(2τ̂) · exp(-2τ/τ̂) · [G(θ_minus|θb) + G(θ_plus|θb)]  [논문 수식 (5)]

Step 4.  μ_DS  = ∫τ·Pc(τ)dτ / ∫Pc(τ)dτ          ← 논문 수식 (6)

Step 5.  σ_DS  = √(∫τ²·Pc(τ)dτ / ∫Pc(τ)dτ − μ²) ← 논문 수식 (7)
```

수치 적분: `scipy.integrate.quad` (적분 상한: `τ̂ × 20`)  
교차검증: Monte Carlo 시뮬레이션 (경로 수 2,000개) 병행

---

## NLOS 모델링 (v4~v7 기준)

논문 수식과 3GPP TR 38.900을 통합한 2-layer 구조.

```
Layer 1 — 3GPP TR 38.900 Table 7.5-6 UMa (채널 환경의 절대적 DS 스케일)
  LOS:  tau_hat ~ LogNormal(μ=-7.03, σ=0.66) → 논문 τ̂=25ns 기준 정규화
  NLOS: tau_hat ~ LogNormal(μ=-6.44, σ=0.39) → LOS 대비 평균 3.89배 증가
  클리핑 상한 100ns (v6 이후 유지)

Layer 2 — 논문 keyhole closed-form (빔포밍 파라미터에 따른 DS 감소 효과)
  M↑, θb↓, Δθ↓ → σ_DS 감소  (논문 Fig.3 재현)
  최종 σ_DS = _calculate_ds() [논문 수식 (6)(7)]
```

---

## 하이퍼파라미터 (train.py, v7 기준)

| 파라미터 | 값 | 설명 |
|---|---|---|
| LR | 0.001 | 초기 학습률 |
| GAMMA | 0.99 | 할인율 |
| EPSILON_START | 1.0 | 초기 탐색률 |
| EPSILON_END | 0.01 | 최소 탐색률 |
| EPSILON_DECAY | 0.995 | 탐색률 감쇠 |
| BATCH_SIZE | 64 | 미니배치 크기 |
| MEMORY_SIZE | 10,000 | 경험 재생 버퍼 크기 |
| EPISODES | 1,000 | 학습 에피소드 수 |
| STEPS_PER_EP | 100 | 에피소드당 스텝 수 |
| TARGET_UPDATE | 10 | Target Network 갱신 주기 |
| NLOS_EPISODE_RATIO | 0.4 | NLOS 구간 포함 에피소드 비율 |
| NLOS_STEPS | 10 | NLOS 구간 지속 스텝 수 |
| LR Scheduler | CosineAnnealingLR(T_max=1000, eta_min=1e-4) | **[v7 변경]** StepLR → CosineAnnealingLR |

---

## 의존성

```
Python 3.11
torch / numpy / scipy / matplotlib
```

---

## 버전 히스토리

---

### v1 — 프로토타입

**Abstract**  
DQN 기반 적응형 CP 조절 알고리즘의 최초 프로토타입.  
env, DQN, train, test 4개 모듈로 구성. 이론적 CP vs Adaptive CP 2×2 대시보드 비교.

**Issues**
- DS closed-form 수식 단순화 오류 (분모를 `0.5*Gm`으로 단순화)
- θ_minus/θ_plus 계산 과정 누락, PDP 구성 없이 적분 결과 하드코딩
- 빔 정렬 오차(Δθ) DS 계산에 미반영
- 고정 CP 비교 없음 (2-way만 존재)
- NLOS 환경 시나리오 없음
- ISI 페널티(-100) 과도하여 보상 발산

---

### v2 — 수식 재구현 + 환경 보완

**Abstract**  
v1 수식 오류 전면 수정. θ_minus/θ_plus → PDP → 수치 적분 전 과정 명시적 구현.  
3-way 비교, NLOS 시나리오, 3×3 대시보드 추가.

**Changes**
- DS 계산 전면 재구현 (수식 5~7 명시적 구현)
- NLOS 1차 시도: theta_b/theta_c 변경
- ISI 페널티 -100 → -20, delta SE 기반 → new_se 절대값 기반 보상

**Issues**
- NLOS에서 σ_DS 전혀 변화 없음 (keyhole 모델 DS 포화: `tau_hat/2 = 12.5ns`)
- Adaptive CP가 초반에 급감 후 고정
- 학습 보상 수렴 불안정 (음수 구간 지속)

---

### v3 — NLOS 모델링 수정 + 보상 안정화

**Abstract**  
NLOS DS 포화 문제를 tau_hat 직접 스케일링으로 해결.  
학습 보상 양수 수렴. Adaptive SE(0.956)가 이론 기준(0.615)과 고정 CP(0.392) 모두 상회.

**v3 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.615 | 0.392 | **0.956** |
| 평균 SE (정상) | 0.762 | 0.427 | **1.139** |
| 평균 SE (NLOS) | 0.271 | 0.312 | **0.529** |

**Changes**
- NLOS: tau_hat을 정상값(25ns)의 3~6배로 직접 스케일링
- reset() SNR 하한 상향, current_cp 초기값 = 이론값으로 안정화

**Issues**
- NLOS 급변 시 Adaptive CP 추종 속도 부족 (step_size 5ns 고정)
- NLOS 구간 보상 최저 -20 발생
- tau_hat 임의 스케일링은 논문/3GPP 비준수

---

### v4 — 3GPP TR 38.900 + 논문 keyhole 통합 모델

**Abstract**  
v3의 임의 tau_hat 스케일링을 3GPP TR 38.900 Table 7.5-6 UMa 표준으로 교체.  
동적 step_size, CP 하한 보장, NLOS 훈련 노출 30% 추가.

**v4 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.560 | 0.375 | **0.900** |
| 평균 SE (정상) | 0.695 | 0.410 | **1.064** |
| 평균 SE (NLOS) | 0.246 | 0.294 | **0.518** |

**Changes**
- 3GPP TR 38.900 기반 tau_hat 샘플링 (`_sample_tau_hat_3gpp`)
- 동적 step_size: CP/σ_DS < 0.8 → σ_DS×0.5, < 1.5 → σ_DS×0.2, 그 외 → σ_DS×0.05
- CP 하한 보장: `current_cp = max(current_cp, sigma_ds)`
- NLOS 훈련 노출 30%

**Issues**
- NLOS 급변 시 Adaptive CP 추종 실패  
  원인 1: train.py가 `nlos_event=False`로만 학습 (NLOS 경험 전무)  
  원인 2: DS 변화를 1스텝 늦게 인지하는 타이밍 문제
- v3 대비 SE 소폭 하락 (3GPP 기반 변동폭 증가로 학습 난이도 상승)
- CP/σ_DS 비율 NLOS 초반 1.0 미만 하락

---

### v5 — 4개 파일 전체 수정

**Abstract**  
env.py뿐 아니라 DQN/train/test 3개 파일도 함께 수정.  
DQN 입력 정규화, train NLOS 구간 명시적 학습, test NLOS 타이밍 1스텝 지연 제거.

**v5 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.411 | 0.344 | **0.438** |
| 평균 SE (정상) | 0.539 | 0.400 | **0.565** |
| 평균 SE (NLOS) | 0.112 | 0.213 | 0.144 ← Fixed CP보다 낮음 |

**Changes**
- **env.py**: NLOS 타이밍 수정 (step() 진입 즉시 채널 변경), STATE_NORM 상수 제공
- **DQN.py**: 입력 정규화 추가 (`x / STATE_NORM`), 은닉층 64→128 확대
- **train.py**: NLOS 구간 명시적 학습 (40% 에피소드), LR 스케줄러, Gradient Clipping
- **test.py**: NLOS 타이밍 수정 (step() 전 채널 갱신)

**Issues**
- NLOS 이후 CP 과도 증가 미해결 (CP/σ_DS 비율 30까지 폭등)
- NLOS 구간 Adaptive SE(0.144)가 Fixed CP(0.213)보다 낮음
- 전체 SE v3(0.956) → v5(0.438)로 크게 하락
- 학습 보상 이동평균 수렴 불안정

---

### v6 — 보상 함수 재설계 + tau_hat 클리핑 조정

**Abstract**  
v5의 낭비 페널티 약점을 해결하기 위해 보상 함수를 `se_ratio × 10 + 동적 페널티` 구조로 전면 재설계.  
tau_hat 클리핑 상한을 500ns → 100ns로 조정하여 극단적 DS 환경 완화.  
결과적으로 Adaptive SE가 Fixed CP를 전 구간에서 하회하여 목표 달성 실패.

**v6 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.408 | 0.330 | 0.263 ← Fixed CP보다 낮음 |
| 평균 SE (정상) | 0.482 | 0.342 | 0.277 ← Fixed CP보다 낮음 |
| 평균 SE (NLOS) | 0.236 | 0.301 | 0.229 ← Fixed CP보다 낮음 |

**Changes**
- **env.py**: tau_hat 클리핑 상한 500ns → 100ns
- **env.py**: 보상 함수 재설계: `(new_se / theo_se) × 10` 기반 + 동적 낭비 페널티

**Issues**

1. **`_get_step_size()` 비대칭 — NLOS 이후 CP 회복 불가 (가장 심각)**  
   ratio > 1.5 시 `step_size = σ_DS × 0.05 ≈ 0.5ns/step`.  
   NLOS 이후 250ns → 30ns까지 440스텝 필요. 30스텝 테스트로는 영구 복귀 불가.

2. **보상 기준점 오류 (`theo_se` 기반 se_ratio)**  
   NLOS에서 theo_cp가 커질수록 theo_se가 낮아져 se_ratio 스케일이 왜곡됨.  
   보상 스케일이 LOS/NLOS 환경마다 달라 에이전트가 일관된 가치 함수 학습 실패.

3. **StepLR 과도한 LR 감쇠**  
   500 에피소드 후 LR 0.001 → 0.0001로 급감. 미수렴 상태에서 학습 사실상 정체.

---

### v7 — step_size 비대칭 해소 + 보상 기준 변경 + LR 스케줄러 교체 (현재 버전)

**Abstract**  
v6에서 진단한 3가지 근본 문제를 각각 env.py 2곳, train.py 1곳을 수정하여 해결.  
Adaptive SE가 Fixed CP를 전 구간에서 상회하는 목표를 최초로 달성.  
DQN.py, test.py는 변경 없음.

**v7 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.680 | 0.435 | **1.146** ← Fixed CP 대비 +163% |
| 평균 SE (정상) | 0.882 | 0.472 | **1.412** |
| 평균 SE (NLOS) | 0.211 | 0.349 | **0.527** ← NLOS도 Fixed CP 상회 |

**Changes**

- **env.py `_get_step_size()`**: ratio 구간 세분화로 NLOS 이후 빠른 CP 복귀 가능

  | ratio 구간 | v6 step_size | v7 step_size |
  |---|---|---|
  | < 0.8 | σ×0.5 | σ×0.5 (유지) |
  | 0.8 ~ 1.5 | σ×0.2 | σ×0.2 (유지) |
  | 1.5 ~ 5.0 | σ×0.05 | **σ×0.3 (신규)** |
  | ≥ 5.0 | σ×0.05 | **σ×0.5 (신규)** |

  실측 복귀 속도 (NLOS 종료 후 sigma_DS=12ns로 복귀 기준):

  | NLOS 이후 CP | v6 복귀 스텝 | v7 복귀 스텝 | 단축 배율 |
  |---|---|---|---|
  | 100ns | 83스텝 | 19스텝 | 4.4배 |
  | 150ns | 133스텝 | 27스텝 | 4.9배 |
  | 200ns | 183스텝 | 35스텝 | 5.2배 |
  | 250ns | 233스텝 | 44스텝 | 5.3배 |

- **env.py `step()` 보상 함수**: 기준선을 `theo_se` → `fixed_se`로 변경

  ```python
  # v6
  reward = (new_se / theo_se) * 10.0        # 환경마다 스케일 변동
  # v7
  reward = (new_se - fixed_se) * 20.0       # Fixed CP 대비 개선량 (일관된 척도)
  ```

- **train.py `scheduler`**: `StepLR` → `CosineAnnealingLR`

  ```python
  # v6: StepLR(step_size=500, gamma=0.1)  → 500 에피소드 후 LR 10배 급감
  # v7: CosineAnnealingLR(T_max=1000, eta_min=1e-4)  → 완만한 코사인 감쇠
  ```

**Issues — v7에서 새롭게 확인된 문제**

#### 문제 1: CP/σ_DS 비율이 1.0 미만으로 진입하는 구간 존재 (차트 ⑦)

에이전트가 Sweet Spot(1~1.5×σ)을 목표로 학습하면서, 정상 구간 초반(Step 1~5)에서  
CP/σ_DS 비율이 간헐적으로 1.0 미만으로 떨어지는 사례가 관찰됨.  
`step()` 내 `current_cp = max(current_cp, sigma_ds)` clamp가 있지만,  
이는 action 실행 *후* 적용되므로 SE 계산 시점에는 이미 ISI 페널티(×0.5)가 적용됨.

실질적 영향: CP=1.25σ에서 Decrease action 시 step_size=σ×0.2=0.25σ 감소 → CP=σ(=clamp 경계).  
Sweet Spot 하한에서 한 번이라도 Decrease를 선택하면 즉시 clamp에 도달하며 ISI 위험 발생.

**v8 수정 방향**: CP 하한을 `sigma_ds`에서 `sigma_ds × 1.1`로 상향하여 안전 마진 확보.
```python
self.current_cp = max(self.current_cp, self.sigma_ds * 1.1)
```

#### 문제 2: Adaptive SE가 Theoretical SE를 크게 상회 — 이론 기준 재정의 필요

v7 결과에서 Adaptive SE(1.146) >> Theoretical SE(0.680).  
현재 `theo_cp = σ_DS × ρ = σ_DS × 3` 은 ISI 안전 마진 기준이지, **SE 최대화 기준이 아님**.  
SE는 CP가 짧을수록 높아지므로, 진짜 SE 최적 CP는 `σ_DS`(ISI 경계) 혹은 `σ_DS × 1.1`(안전 마진 포함).  
이로 인해 Theoretical이 실질적인 상한 역할을 못 하고 있으며, 비교 결과 해석이 왜곡됨.

**v8 수정 방향**: Theoretical CP를 SE 최적 기준으로 재정의.
```python
# 현재 (ISI 안전 마진 기준)
theo_cp = sigma_ds * rho          # = σ × 3.0

# v8 (SE 최적 기준 — ISI 방지 최소 CP)
theo_cp = sigma_ds * 1.1          # ISI 경계 + 10% 마진
```
이렇게 하면 Theoretical SE가 실질적인 SE 상한이 되고, Adaptive가 이를 얼마나 추종하는지  
의미 있는 비교가 가능해짐.

#### 문제 3: NLOS 구간 보상 및 SE 급락 (차트 ②⑤)

NLOS 진입 시(Step 10~18) SE가 0.1~0.2 수준으로 급락.  
이는 NLOS에서 SNR이 3~12dB로 낮아지는 물리적 현상이 주된 원인이지만,  
에이전트가 NLOS 진입 초기에 CP를 과도하게 낮게 유지하다 ISI에 노출되는 문제도 기여.  
NLOS 인식 후 CP를 올리는 데 수 스텝이 소요되는 동안 SE 손실 발생.

**v8 수정 방향**: 상태 벡터에 `is_nlos` 플래그를 명시적으로 포함하여  
에이전트가 NLOS 진입을 즉시 인식하고 선제적으로 CP를 상향할 수 있도록 유도.
```python
# state에 is_nlos 플래그 추가 (state_dim: 5 → 6)
return np.array([
    self.sigma_ds * 1e9,
    self.snr,
    self.delta_theta_deg,
    float(self.M),
    self.current_cp * 1e9,
    float(self.is_nlos),   # [v8 신규]
], dtype=np.float32)
```

#### 문제 4: 학습 보상 이동평균 수렴 불완전 (차트 ①)

이동평균이 1,500~3,500 구간에서 진동하며 안정적인 수렴 구간이 없음.  
보상 분산이 높아 학습 후반부에도 정책이 계속 변동하고 있음을 시사.  
CosineAnnealingLR 적용으로 v6 대비 개선됐으나 아직 불충분.

**v8 수정 방향**: 에피소드 수 1,000 → 2,000으로 확대, 또는 보상 정규화(Reward Clipping/Scaling) 도입.
```python
# 보상 클리핑으로 이상치 억제
reward = np.clip(reward, -20.0, 20.0)
```

---

## 버전별 핵심 지표 비교

| 버전 | Adaptive SE (전체) | NLOS SE | Fixed CP 대비 | 비고 |
|---|---|---|---|---|
| v1 | — | — | — | 수식 오류 |
| v2 | — | — | — | DS 포화 |
| v3 | 0.956 | 0.529 | +144% | tau_hat 임의 스케일링 (3GPP 비준수) |
| v4 | 0.900 | 0.518 | +140% | 3GPP 도입, NLOS 추종 실패 |
| v5 | 0.438 | 0.144 | +27% | NLOS 타이밍 수정 |
| v6 | 0.263 | 0.229 | **-20%** | 보상 재설계 역효과 |
| **v7** | **1.146** | **0.527** | **+163%** | **Fixed CP 전 구간 상회 최초 달성** |

---

## v8 개선 방침

v7 분석을 통해 도출된 개선 방침. 수정 대상 파일과 위치를 명시한다.

| # | 파일 | 수정 위치 | 내용 |
|---|---|---|---|
| 1 | `env.py` | `step()` CP 하한 clamp | `sigma_ds` → `sigma_ds × 1.1` (ISI 안전 마진 10% 확보) |
| 2 | `env.py` | `_make_state()` 및 `step()` | `is_nlos` 플래그를 state 6번째 차원으로 추가 (`state_dim: 5 → 6`) |
| 3 | `env.py` | `step()` theo_cp 정의 | `sigma_ds × 3.0` → `sigma_ds × 1.1` (SE 최적 기준으로 재정의) |
| 4 | `env.py` | `step()` 보상 클리핑 | `reward = np.clip(reward, -20.0, 20.0)` 추가 |
| 5 | `DQN.py` | `STATE_NORM` 및 네트워크 입력 | state_dim 변경에 따라 STATE_NORM 6차원으로 확장 |
| 6 | `train.py` | `EPISODES` | 1,000 → 2,000으로 확대 (학습 수렴 안정화) |

**목표**: CP/σ_DS가 항상 1.1 이상 유지되고, Adaptive SE가 새로운 Theoretical SE(σ×1.1 기준)에 수렴하는 안정적 정책 달성.
