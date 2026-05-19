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

## NLOS 모델링 (v4~v5 기준)

논문 수식과 3GPP TR 38.900을 통합한 2-layer 구조.

```
Layer 1 — 3GPP TR 38.900 Table 7.5-6 UMa (채널 환경의 절대적 DS 스케일)
  LOS:  tau_hat ~ LogNormal(μ=-7.03, σ=0.66) → 논문 τ̂=25ns 기준 정규화
  NLOS: tau_hat ~ LogNormal(μ=-6.44, σ=0.39) → LOS 대비 평균 3.89배 증가

Layer 2 — 논문 keyhole closed-form (빔포밍 파라미터에 따른 DS 감소 효과)
  M↑, θb↓, Δθ↓ → σ_DS 감소  (논문 Fig.3 재현)
  최종 σ_DS = _calculate_ds() [논문 수식 (6)(7)]
```

---

## 하이퍼파라미터 (train.py, v5 기준)

| 파라미터 | 값 | 설명 |
|---|---|---|
| LR | 0.001 | 학습률 (500 에피소드 후 0.1배 감소) |
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

### v5 — 4개 파일 전체 수정 (현재 버전)

**Abstract**  
env.py뿐 아니라 DQN/train/test 3개 파일도 함께 수정.  
DQN 입력 정규화, train NLOS 구간 명시적 학습, test NLOS 타이밍 1스텝 지연 제거.

**v5 결과**

| 지표 | Theoretical | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.411 | 0.344 | **0.438** |
| 평균 SE (정상) | 0.539 | 0.400 | **0.565** |
| 평균 SE (NLOS) | 0.112 | 0.213 | **0.144** |

**Changes**
- **env.py**: NLOS 타이밍 수정 (step() 진입 즉시 채널 변경), STATE_NORM 상수 제공
- **DQN.py**: 입력 정규화 추가 (`x / STATE_NORM`), 은닉층 64→128 확대
- **train.py**: NLOS 구간 명시적 학습 (40% 에피소드), LR 스케줄러, Gradient Clipping
- **test.py**: NLOS 타이밍 수정 (step() 전 채널 갱신)

**Issues**

- **NLOS 이후 CP 과도 증가 미해결**  
  NLOS 종료 후에도 CP가 계속 상승해 Step 30에서 150ns 수준까지 치솟음.  
  DS가 정상(10ns)으로 복원됐으나 CP는 내려오지 않음 → CP/σ_DS 비율 30까지 폭등.  
  원인: 낭비 페널티(-2)가 너무 약해 CP를 줄이는 동기 부족.

- **NLOS 구간 Adaptive SE(0.144)가 Fixed CP(0.213)보다 낮음**  
  Adaptive가 고정 CP보다 못한 결과. 가장 심각한 성능 퇴보.

- **전체 SE v3(0.956) → v5(0.438)로 크게 하락**  
  3GPP NLOS DS 변동폭(최대 175ns)이 학습 구조가 감당할 수 있는 수준 초과.  
  에이전트가 극단적 DS 환경에서의 CP 조정 정책을 충분히 학습하지 못함.

- **학습 보상 이동평균 수렴 불안정**  
  이동평균이 전반에 걸쳐 500~1,000 수준에서 진동. 명확한 수렴 경향 없음.

---

## v6 개선 방침 (대기 중)

v5 분석을 통해 도출된 개선 방침. 다음 세 가지 원칙을 준수하여 v6를 설계한다.

### 원칙 1 — 논문 수식 및 3GPP 표준 절대 준수
논문의 수식 (4)~(8) 및 3GPP TR 38.900 파라미터는 임의로 변경하지 않는다.  
성능 개선을 위해 이 항목들을 수정하는 것은 허용하지 않는다.

### 원칙 2 — 학습 이슈는 보상 함수로만 해결
학습 불안정, CP 과도 증가, NLOS 대응 실패 등 학습 관련 문제는  
DQN의 보상 함수(reward function) 재설계로 해결한다.  
검토할 사항:
- 낭비 페널티 강화: `-2 → -10` 이상으로 상향
- SE 절대값 기반 보상 구조의 타당성 재검토
- NLOS 구간과 정상 구간의 보상 스케일 균형

### 원칙 3 — NLOS 환경 정의 재검토
현재 3GPP log-normal 샘플링이 DS를 최대 175ns까지 생성하여  
학습/테스트 환경이 지나치게 가혹해졌다는 문제가 있다.  
다음 항목을 검토하고 명시적으로 정의한 뒤 구현한다.

- **NLOS 물리적 정의**: 도심 NLOS 환경에서 DS가 증가하는 물리적 메커니즘 명시  
  (산란체 증가 → 다중 경로 수 증가 → τ̂ 증가 → σ_DS 증가)
- **DS 증가 방식**: 3GPP log-normal 분포를 유지하되, 학습 가능한 범위로 상한 조정이  
  논리적으로 타당한지 검토
- **논리적 타당성 검토**: 현재 tau_hat 스케일링이 실제 NLOS 채널 물리 현상을  
  올바르게 모델링하는지 재검토. 대안으로 클러스터 수 증가 기반 모델 검토.

---

## 버전별 핵심 지표 비교

| 버전 | Adaptive SE | NLOS SE | NLOS 보상 최저 | DS 변화 |
|---|---|---|---|---|
| v1 | - | - | -∞ | 없음 |
| v2 | - | - | -20 | 없음 (포화) |
| v3 | **0.956** | **0.529** | -20 | 7배 |
| v4 | 0.900 | 0.518 | ~0 | 6배 |
| v5 | 0.438 | 0.144 | ~0 | **14배 (과도)** |

**최고 성능: v3**  
v3의 tau_hat 직접 스케일링이 결과적으로 학습 안정성 측면에서 가장 우수했으나,  
논문/3GPP 비준수라는 근본적 한계가 있음.  
v6의 목표는 논문/3GPP를 준수하면서 v3 수준의 성능을 달성하는 것.
