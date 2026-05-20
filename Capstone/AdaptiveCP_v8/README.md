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
| Theoretical CP | 매 스텝 SE 최적 기준값 (`σ_DS × 1.1`, v8 이후) |
| Adaptive CP (DQN) | 강화학습 에이전트가 동적으로 결정 |

> **v8 이전**: Theoretical CP = `σ_DS × 3.0` (ISI 안전 마진 기준)  
> **v8 이후**: Theoretical CP = `σ_DS × 1.1` (SE 최적 최소 CP 기준)

---

## 파일 구조

```
AdaptiveCP/
├── env.py                     # 강화학습 환경 (채널 모델, PDP, DS 계산, 보상 함수)
├── DQN.py                     # 신경망 구조 (3층 FC, 입력 정규화)
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

## NLOS 모델링 (v4~v8 기준)

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

## 하이퍼파라미터 (train.py, v8 기준)

| 파라미터 | 값 | 설명 |
|---|---|---|
| LR | 0.001 | 초기 학습률 |
| GAMMA | 0.99 | 할인율 |
| EPSILON_START | 1.0 | 초기 탐색률 |
| EPSILON_END | 0.01 | 최소 탐색률 |
| EPSILON_DECAY | 0.995 | 탐색률 감쇠 |
| BATCH_SIZE | 64 | 미니배치 크기 |
| MEMORY_SIZE | 10,000 | 경험 재생 버퍼 크기 |
| **EPISODES** | **2,000** | **[v8 변경]** 1,000 → 2,000 |
| STEPS_PER_EP | 100 | 에피소드당 스텝 수 |
| TARGET_UPDATE | 10 | Target Network 갱신 주기 |
| NLOS_EPISODE_RATIO | 0.4 | NLOS 구간 포함 에피소드 비율 |
| NLOS_STEPS | 10 | NLOS 구간 지속 스텝 수 |
| LR Scheduler | CosineAnnealingLR(T_max=2000, eta_min=1e-4) | [v8] T_max 업데이트 |

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
- 3GPP TR 38.900 기반 tau_hat 샘플링
- 동적 step_size (ratio 구간별 3단계)
- CP 하한 보장, NLOS 훈련 노출 30%

**Issues**
- NLOS 급변 시 Adaptive CP 추종 실패 (NLOS 경험 전무, 타이밍 지연)
- CP/σ_DS 비율 NLOS 초반 1.0 미만 하락

---

### v5 — 4개 파일 전체 수정

**Abstract**  
DQN 입력 정규화, train NLOS 구간 명시적 학습, test NLOS 타이밍 1스텝 지연 제거.

**v5 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.411 | 0.344 | **0.438** |
| 평균 SE (정상) | 0.539 | 0.400 | **0.565** |
| 평균 SE (NLOS) | 0.112 | 0.213 | 0.144 ← Fixed CP보다 낮음 |

**Changes**
- env.py: NLOS 타이밍 수정, STATE_NORM 제공
- DQN.py: 입력 정규화, 은닉층 64→128
- train.py: NLOS 명시적 학습(40%), LR 스케줄러, Gradient Clipping
- test.py: NLOS 타이밍 수정

**Issues**
- NLOS 이후 CP 과도 증가 (CP/σ_DS 30까지 폭등)
- NLOS 구간 Adaptive SE < Fixed CP
- 학습 보상 이동평균 수렴 불안정

---

### v6 — 보상 함수 재설계 + tau_hat 클리핑 조정

**Abstract**  
보상 함수 `se_ratio × 10 + 동적 페널티` 구조로 재설계. tau_hat 클리핑 500ns → 100ns.  
Adaptive SE가 Fixed CP 전 구간 하회 — 목표 달성 실패.

**v6 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.408 | 0.330 | 0.263 ← 최저 |
| 평균 SE (정상) | 0.482 | 0.342 | 0.277 |
| 평균 SE (NLOS) | 0.236 | 0.301 | 0.229 |

**Issues**
- `_get_step_size()` 비대칭: ratio > 1.5 시 step_size ≈ 0.5ns → CP 회복 440스텝 필요
- 보상 기준 `theo_se` 스케일 왜곡
- StepLR 과도한 LR 감쇠 (500 에피소드 후 0.1배)

---

### v7 — step_size 비대칭 해소 + 보상 기준 변경 + LR 스케줄러 교체

**Abstract**  
v6의 3가지 근본 문제 수정. Adaptive SE가 Fixed CP를 전 구간 상회하는 목표 최초 달성.

**v7 결과**

| 지표 | Theoretical (σ×3) | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.680 | 0.435 | **1.146** (+163%) |
| 평균 SE (정상) | 0.882 | 0.472 | **1.412** |
| 평균 SE (NLOS) | 0.211 | 0.349 | **0.527** |

**Changes**
- env.py `_get_step_size()`: ratio > 1.5 구간 세분화 (σ×0.3 / σ×0.5)  
  → NLOS 이후 CP 복귀 속도 최대 5.3배 단축
- env.py `step()` 보상: `(new_se / theo_se) × 10` → `(new_se - fixed_se) × 20`
- train.py: StepLR → CosineAnnealingLR(T_max=1000, eta_min=1e-4)

**Issues**
- CP/σ_DS 비율이 간헐적으로 1.0 미만 진입 (ISI 위반 가능성)
- Adaptive SE >> Theoretical SE: `theo_cp = σ×3.0`은 SE 최적이 아님 — 비교 기준 부적절
- NLOS 진입 시 에이전트의 선제 대응 부족 (is_nlos 정보 미활용)
- 학습 보상 이동평균 수렴 불완전 (1,500~3,500 진동)

---

### v8 — ISI 차단 + is_nlos state 추가 + Theoretical CP 재정의 (현재 버전)

**Abstract**  
v7에서 확인된 4가지 문제를 모두 수정.  
CP 하한 σ×1.1으로 ISI 원천 차단, is_nlos 플래그 state 추가,  
Theoretical CP를 SE 최적 기준(σ×1.1)으로 재정의, 보상 클리핑 및 에피소드 2,000으로 확대.

**v8 결과**

| 지표 | Theoretical (σ×1.1) | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | (측정 예정) | (측정 예정) | (측정 예정) |
| 평균 SE (정상) | — | — | — |
| 평균 SE (NLOS) | — | — | — |

**Changes**

- **env.py `step()` CP 하한** (`수정 1`):
  ```python
  # v7: max(current_cp, sigma_ds)
  # v8: max(current_cp, sigma_ds * 1.1)
  self.current_cp = max(self.current_cp, self.sigma_ds * self.CP_SAFETY_MARGIN)
  ```
  ISI 발생 조건(`cp < sigma_ds`) 물리적 진입 불가. 10% 마진으로 채널 추정 오차 흡수.

- **env.py `_make_state()` + state_dim** (`수정 2`):
  ```python
  # v7: state = [σ_DS, SNR, Δθ, M, CP]           state_dim=5
  # v8: state = [σ_DS, SNR, Δθ, M, CP, is_nlos]  state_dim=6
  ```
  에이전트가 NLOS 진입을 즉시 인식하여 선제적 CP 상향 정책 학습 가능.

- **env.py `step()` theo_cp 재정의** (`수정 3`):
  ```python
  # v7: theo_cp = sigma_ds * rho       (= σ × 3.0, ISI 안전 마진 기준)
  # v8: theo_cp = sigma_ds * 1.1       (= SE 최적 최소 CP)
  ```
  Theoretical SE가 실질적 SE 상한이 되어 Adaptive의 이론 추종 정도를 정확히 평가 가능.

- **env.py `step()` 보상 클리핑** (`수정 4`):
  ```python
  reward = float(np.clip(reward, -20.0, 20.0))
  ```
  이상치 보상 억제 → 학습 분산 감소.

- **DQN.py STATE_NORM** (`수정 5`):
  ```python
  # v7: [200.0, 30.0, 45.0, 256.0, 600.0]          5차원
  # v8: [200.0, 30.0, 45.0, 256.0, 600.0, 1.0]     6차원
  ```

- **train.py EPISODES + CosineAnnealingLR T_max** (`수정 6`):
  ```python
  EPISODES = 2_000   # 1,000 → 2,000
  CosineAnnealingLR(optimizer, T_max=EPISODES, eta_min=1e-4)
  ```

---

## 버전별 핵심 지표 비교

| 버전 | Adaptive SE (전체) | NLOS SE | Fixed CP 대비 | Theoretical 기준 |
|---|---|---|---|---|
| v1 | — | — | — | σ×3 |
| v2 | — | — | — | σ×3 |
| v3 | 0.956 | 0.529 | +144% | σ×3 (3GPP 비준수) |
| v4 | 0.900 | 0.518 | +140% | σ×3 |
| v5 | 0.438 | 0.144 | +27% | σ×3 |
| v6 | 0.263 | 0.229 | **-20%** | σ×3 |
| v7 | 1.146 | 0.527 | **+163%** | σ×3 |
| **v8** | **(측정 예정)** | **(측정 예정)** | — | **σ×1.1 (SE 최적)** |

**v8 목표**:
- CP/σ_DS 항상 ≥ 1.1 (ISI 위반 0%)
- Adaptive SE가 Theoretical SE(σ×1.1 기준)에 수렴
- 학습 보상 이동평균 안정 수렴
