# AdaptiveCP_v2 — DQN 기반 적응형 CP 조절 알고리즘

> **논문 기반:** "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"  
> Akgun, Krunz, Manzi — ICNC 2020

---

## 프로젝트 개요

NLOS 환경에서 고속 이동체의 통신 연속성 유지를 위해, DQN 기반 강화학습을 활용하여 채널의 DS(Delay Spread) 추정치에 따라 CP(Cyclic Prefix) 길이를 동적으로 조절하는 알고리즘 연구.

### 비교 대상 (3-way)
| 방식 | 설명 |
|---|---|
| Fixed CP | 상용 시스템 기준 고정값 (`τ̂ × ρ = 75ns`) |
| Theoretical CP | 매 스텝 실제 σ_DS 기반 최적값 (`σ_DS × ρ`) |
| Adaptive CP (DQN) | 강화학습 에이전트가 동적으로 결정 |

---

## 파일 구조

```
AdaptiveCP_v2/
├── env.py        # 강화학습 환경 (채널 모델, PDP, DS 계산, 보상 함수)
├── DQN.py        # 신경망 구조 (3층 FC, 입력 5차원, 출력 3차원)
├── train.py      # 학습 루프 (ε-greedy, Experience Replay, Target Network)
├── test.py       # 평가 및 대시보드 시각화 (3×3 서브플롯)
├── dqn_adaptive_cp.pth   # 학습된 모델 가중치 (train 후 생성)
├── train_rewards.npy     # 에피소드별 보상 기록 (train 후 생성)
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

Step 2.  G(θn|θb) = Gm  if |θb - θn| ≤ ωb/2    ← 주엽
                  = gm  otherwise                 ← 부엽

Step 3.  Pc(τ) = 1/(2τ̂) · exp(-2τ/τ̂) · [G(θ_minus|θb) + G(θ_plus|θb)]

Step 4.  μ_DS  = ∫τ·Pc(τ)dτ / ∫Pc(τ)dτ          ← 논문 수식 (6)

Step 5.  σ_DS  = √(∫τ²·Pc(τ)dτ / ∫Pc(τ)dτ − μ²) ← 논문 수식 (7)
```

수치 적분: `scipy.integrate.quad` 사용 (적분 상한: `τ̂ × 20`)  
교차검증: Monte Carlo 시뮬레이션 (경로 수 2,000개) 병행

---

## 버전 히스토리

---

### v1 — 프로토타입 (최초 구현)

**Abstract**  
DQN 기반 적응형 CP 조절 알고리즘의 최초 프로토타입. 논문의 수식 구조를 참고하여 env, DQN, train, test 4개 모듈로 구성. 학습 후 이론적 CP와 Adaptive CP의 SE 및 전력 효율을 2×2 대시보드로 비교.

**Issues**

- **DS closed-form 수식 단순화 오류**  
  논문 수식 (6)(7)의 분모를 `0.5 * G_m`으로 단순화. 실제 PDP 적분값을 사용하지 않아 DS 계산값이 부정확함.

- **빔 정렬 오차(Δθ) 미반영**  
  `theta_c`가 State 벡터에는 포함되어 있으나 DS 계산 시 `Δθ = 0`으로 고정되어 beam misalignment 효과가 DS에 반영되지 않음.

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

### v2 — 현재 버전

**Abstract**  
v1의 수식 오류 및 구조적 문제를 전면 수정. 논문 수식의 전체 유도 과정(θ_minus/θ_plus → PDP → 수치 적분)을 단계별로 명시적으로 구현. 고정 CP를 포함한 3-way 비교, NLOS 급변 시나리오, 확장된 3×3 대시보드 추가.

**Changes**

- **DS 계산 전면 재구현**  
  `_theta_minus()`, `_theta_plus()`, `_antenna_gain()`, `_pdp()`, `_calculate_ds()` 메서드를 순서대로 구현하여 논문 수식 흐름을 코드에서 직접 추적 가능하도록 구성.

- **NLOS 환경 모델링 수정**  
  기존: `theta_b`, `theta_c` 변경 → keyhole 모델 특성상 DS가 `tau_hat/2 = 12.5ns`에서 포화되어 변화 없음.  
  수정: **`tau_hat` 스케일링**으로 NLOS 표현.  
  실제 NLOS 환경에서 산란체 급증으로 평균 경로 지연 자체가 길어지는 현상을 모델링.
  ```
  정상: tau_hat = 25ns  → σ_DS ≈  6~12 ns
  NLOS: tau_hat = 75~150ns → σ_DS ≈ 30~65 ns  (최대 7배 증가)
  ```

- **보상 함수 재설계**  
  기존: `SE 개선분(delta SE)` 기반 → CP를 무조건 줄이는 정책 학습  
  수정: `new_se * 10.0` 절대값 기반으로 변경. ISI 페널티(-20)가 SE 이득보다 커지는 시점에 자연스럽게 CP 증가를 선택하도록 유도.
  
  | 구간 조건 | 보상 |
  |---|---|
  | CP < σ_DS (ISI 발생) | `new_se * 10 - 20` |
  | σ_DS ≤ CP ≤ 1.5×σ_DS (Sweet Spot) | `new_se * 10 + 5` |
  | CP > ρ×σ_DS (낭비 구간) | `new_se * 10 - 2` |
  | 그 외 | `new_se * 10` |

- **3-way 비교 추가**  
  Fixed CP (75ns 고정) / Theoretical CP / Adaptive CP (DQN) 비교.

- **대시보드 확장 (2×2 → 3×3)**  
  ① 학습 보상 추이, ② 스텝별 보상, ③ σ_DS 변화,  
  ④ CP 길이 비교, ⑤ SE 비교, ⑥ 전력 효율 비교,  
  ⑦ CP/σ_DS 비율 (ISI 안전 마진), ⑧ 평균 SE 요약, ⑨ Normal vs NLOS SE 비교

**Known Issues**

- 수치 적분(`scipy.integrate.quad`) 사용으로 인해 학습 속도가 CPU 기준 v1 대비 느림.  
  에피소드당 `_calculate_ds()` 호출 횟수 최소화 또는 lookup table 방식으로 최적화 가능.

- 학습 보상 수렴이 아직 불안정. 하이퍼파라미터 튜닝 또는 Dueling DQN 구조 도입 검토 필요.

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
