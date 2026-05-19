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
├── DQN.py                     # 신경망 구조 (3층 FC, 입력 5차원, 출력 3차원)
├── train.py                   # 학습 루프 (ε-greedy, Experience Replay, Target Network)
├── test.py                    # 평가 및 대시보드 시각화 (3×3 서브플롯)
├── dqn_adaptive_cp.pth        # 학습된 모델 가중치 (train 후 생성)
├── train_rewards.npy          # 에피소드별 보상 기록 (train 후 생성)
└── adaptive_cp_dashboard.png  # 결과 대시보드 (test 후 생성)
```

---

## 실행 방법

```powershell
# 학습
py -3.11 train.py

# 평가
py -3.11 test.py
```

---

## DS 계산 흐름 (env.py)

논문 수식을 단계별로 명시적으로 구현.

```
Step 1.  θ_minus(τ) = θc - κ·√(τ/τ̂)      ← l = -1 경로 AoA
         θ_plus(τ)  = θc + κ·√(τ/τ̂)      ← l = +1 경로 AoA

Step 2.  G(θn|θb) = Gm  if |θb - θn| ≤ ωb/2    ← 주엽  [논문 수식 (4)]
                  = gm  otherwise                 ← 부엽

Step 3.  Pc(τ) = 1/(2τ̂) · exp(-2τ/τ̂) · [G(θ_minus|θb) + G(θ_plus|θb)]  [논문 수식 (5)]

Step 4.  μ_DS  = ∫τ·Pc(τ)dτ / ∫Pc(τ)dτ          ← 논문 수식 (6)

Step 5.  σ_DS  = √(∫τ²·Pc(τ)dτ / ∫Pc(τ)dτ − μ²) ← 논문 수식 (7)
```

수치 적분: `scipy.integrate.quad` 사용 (적분 상한: `τ̂ × 20`)  
교차검증: Monte Carlo 시뮬레이션 (경로 수 2,000개) 병행

---

## NLOS 모델링 (v4 기준)

논문 수식과 3GPP TR 38.900을 통합한 2-layer 구조.

```
Layer 1 — 3GPP TR 38.900 Table 7.5-6 UMa (채널 환경 결정)
  LOS:  tau_hat ~ LogNormal(μ=-7.03, σ=0.66) → 논문 스케일 정규화
  NLOS: tau_hat ~ LogNormal(μ=-6.44, σ=0.39) → LOS 대비 평균 3.89배 증가

Layer 2 — 논문 keyhole closed-form (빔포밍 효과 반영)
  M↑, θb↓, Δθ↓ → σ_DS 감소  (논문 Fig.3 재현)
  최종 σ_DS = _calculate_ds() (수식 6,7 수치 적분)
```

---

## 하이퍼파라미터 (train.py)

| 파라미터 | 값 | 설명 |
|---|---|---|
| LR | 0.001 | 학습률 |
| GAMMA | 0.99 | 할인율 |
| EPSILON_START | 1.0 | 초기 탐색률 |
| EPSILON_END | 0.01 | 최소 탐색률 |
| EPSILON_DECAY | 0.995 | 탐색률 감쇠 |
| BATCH_SIZE | 64 | 미니배치 크기 |
| MEMORY_SIZE | 10,000 | 경험 재생 버퍼 크기 |
| EPISODES | 1,000 | 학습 에피소드 수 |
| STEPS_PER_EP | 100 | 에피소드당 스텝 수 |
| TARGET_UPDATE | 10 | Target Network 갱신 주기 |

---

## 의존성

```
Python 3.11
torch
numpy
scipy
matplotlib
```

---

## 버전 히스토리

---

### v1 — 프로토타입 (최초 구현)

**Abstract**  
DQN 기반 적응형 CP 조절 알고리즘의 최초 프로토타입.  
논문의 수식 구조를 참고하여 env, DQN, train, test 4개 모듈로 구성.  
학습 후 이론적 CP와 Adaptive CP의 SE 및 전력 효율을 2×2 대시보드로 비교.

**Issues**

- **DS closed-form 수식 단순화 오류**  
  논문 수식 (6)(7)의 분모를 `0.5 * G_m`으로 단순화.  
  실제 PDP 적분값을 사용하지 않아 DS 계산값이 부정확함.

- **빔 정렬 오차(Δθ) 미반영**  
  `theta_c`가 State 벡터에는 포함되어 있으나 DS 계산 시 `Δθ = 0`으로 고정되어  
  beam misalignment 효과가 DS에 반영되지 않음.

- **θ_minus / θ_plus 계산 과정 누락**  
  논문의 핵심 채널 모델인 `θn(τn) = l·κ·√(τn/τ̂) + θc` 수식이 구현되지 않음.  
  PDP 구성 과정 없이 적분 결과값을 하드코딩.

- **고정 CP 비교 없음**  
  이론적 CP vs Adaptive CP 2-way 비교만 존재. 상용 고정 CP 케이스 미포함.

- **NLOS 환경 시나리오 없음**  
  테스트 구간 내 채널 급변 시나리오가 없어 동적 적응 능력 검증 불가.

- **보상 함수 스케일 불균형**  
  ISI 페널티 `-100`이 SE 개선분 대비 과도하게 커서 학습 초기 보상이 극단적으로 발산.

---

### v2 — 수식 재구현 + 환경 보완

**Abstract**  
v1의 수식 오류 및 구조적 문제를 전면 수정.  
논문 수식의 전체 유도 과정 (θ_minus/θ_plus → PDP → 수치 적분)을 단계별로 명시적으로 구현.  
고정 CP를 포함한 3-way 비교, NLOS 급변 시나리오, 확장된 3×3 대시보드 추가.

**Changes**

- DS 계산 전면 재구현 (`_theta_minus`, `_theta_plus`, `_antenna_gain`, `_pdp`, `_calculate_ds`)
- NLOS 환경 모델링 1차 시도 (theta_b/theta_c 변경)
- 보상 함수: `delta SE` 기반 → `new_se * 10.0` 절대값 기반, ISI 페널티 `-100 → -20`
- 3-way 비교 및 3×3 대시보드 추가

**Issues**

- **NLOS에서 σ_DS 전혀 변화 없음 (핵심 문제)**  
  keyhole 모델 구조적 한계: DS가 `tau_hat/2 = 12.5ns`에서 포화.  
  `theta_b/theta_c` 변경으로는 DS가 12.5ns를 초과하지 않음.

- **Adaptive CP가 초반 급감 후 고정**  
  에이전트가 action=0(감소)만 반복 선택. ISI 경계 학습 불충분.

- **학습 보상 수렴 불안정**  
  전반에 걸쳐 음수 보상 구간 지속.

---

### v3 — NLOS 모델링 수정 + 보상 함수 안정화

**Abstract**  
v2의 핵심 문제였던 NLOS DS 포화 현상을 `tau_hat` 스케일링으로 해결.  
학습 보상이 처음부터 양수 수렴. Adaptive CP 평균 SE(0.956)가  
이론 기준(0.615) 및 고정 CP(0.392) 대비 모두 우수함을 달성.

**v3 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.615 | 0.392 | **0.956** |
| 평균 SE (정상) | 0.762 | 0.427 | **1.139** |
| 평균 SE (NLOS) | 0.271 | 0.312 | **0.529** |

**Changes**

- NLOS 모델링: `tau_hat` 직접 스케일링 (정상 × 3~6배)  
  → 정상 σ_DS ≈ 6~12ns, NLOS σ_DS ≈ 30~65ns (최대 7배 증가)
- reset() SNR 하한 상향, current_cp 초기값 이론값으로 안정화

**Issues**

- NLOS 급변 시 Adaptive CP 추종 속도 부족 (step_size 5ns 고정)
- NLOS 구간 보상 최저 -20 발생
- CP/σ_DS 비율이 NLOS 초반 1.0 미만으로 하락
- `tau_hat` 임의 스케일링은 논문/3GPP 비준수

---

### v4 — 3GPP TR 38.900 + 논문 keyhole 통합 모델 (현재 버전)

**Abstract**  
v3의 임의적 `tau_hat` 스케일링을 3GPP TR 38.900 Table 7.5-6 UMa 표준 기반으로 교체.  
논문 keyhole closed-form이 빔포밍 효과(M, θb, Δθ)에 따른 DS 상대 변화를 담당하고,  
3GPP log-normal 분포가 LOS/NLOS 채널 환경의 절대적 DS 스케일을 결정하는 2-layer 구조.  
동적 step_size 및 CP 하한 보장 추가.

**v4 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.560 | 0.375 | **0.900** |
| 평균 SE (정상) | 0.695 | 0.410 | **1.064** |
| 평균 SE (NLOS) | 0.246 | 0.294 | **0.518** |

**Changes**

- **3GPP TR 38.900 기반 tau_hat 샘플링 도입**  
  `_sample_tau_hat_3gpp(is_nlos)`:  
  LOS: `tau_hat ~ LogNormal(μ=-7.03, σ=0.66)` 논문 스케일 정규화  
  NLOS: `tau_hat ~ LogNormal(μ=-6.44, σ=0.39)` LOS 대비 평균 3.89배 증가

- **동적 step_size 도입**  
  CP/σ_DS < 0.8 → `σ_DS × 0.5` / < 1.5 → `σ_DS × 0.2` / 그 외 → `σ_DS × 0.05`

- **CP 하한 보장**  
  `current_cp = max(current_cp, sigma_ds)` 클리핑

- **NLOS 훈련 노출 30%**  
  `reset()` 시 30% 확률로 NLOS 상태로 초기화

**Issues**

- **NLOS 급변 시 Adaptive CP 추종 실패 (미해결)**  
  원인 1: test.py가 train.py와 달리 `nlos_event=True`로 step() 호출.  
  에이전트가 학습 중 DS 급변 상황을 경험하지 못해 대응 정책 미학습.  
  원인 2: DS 변화를 1스텝 늦게 인지하는 타이밍 문제.

- **v3 대비 SE 소폭 하락**  
  3GPP 기반 tau_hat 샘플링의 변동폭이 커서 학습 난이도 증가.

- **CP/σ_DS 비율 NLOS 초반 1.0 미만 하락**  
  step() 내 DS 업데이트 후 클리핑이 적용되지 않는 타이밍 문제.

---

### v5 — 예정 (개선 계획)

**개선 목표**

- **train.py: NLOS 에피소드 명시적 학습**  
  에피소드 루프 내에서 `step(action, nlos_event=True)` 구간을 포함시켜  
  에이전트가 DS 급변 상황을 직접 경험하도록 구성.

- **DQN.py: 입력 정규화 레이어 추가**  
  State 값의 스케일 차이(DS: 수십 ns / M: 16~256 / SNR: 0~30dB)가  
  학습 안정성에 영향을 미침. 입력 BatchNorm 또는 수동 정규화 추가.

- **test.py: NLOS 타이밍 문제 수정**  
  현재 `step(action, nlos_event=True)` 호출 시 DS가 변경된 후  
  다음 스텝에서야 에이전트가 인지하는 1스텝 지연 구조 수정.

- **train.py: NLOS 구간 경험 우선 샘플링**  
  Prioritized Experience Replay 또는 NLOS 경험 별도 버퍼 구성.
