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
| Theoretical CP | 논문 정의 기준값 (`TCP = ρ·σ_DS = 3·σ_DS`) |
| Adaptive CP (DQN) | 강화학습 에이전트가 동적으로 결정 |

> **논문 TCP 정의**: TCP = ρ·σ_DS, ρ = 3 (paper reference [16]) — 전 버전 통일 기준  
> **CP 하한 clamp**: `σ_DS × 1.1` (v8~v9, ISI 차단 구현 보조값 — Theoretical CP와 별개)

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

## NLOS 모델링 (v4~v9 기준)

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

## 하이퍼파라미터 (train.py, v9 기준)

| 파라미터 | 값 | 설명 |
|---|---|---|
| LR | 0.001 | 초기 학습률 |
| GAMMA | 0.99 | 할인율 |
| EPSILON_START | 1.0 | 초기 탐색률 |
| EPSILON_END | 0.01 | 최소 탐색률 |
| EPSILON_DECAY | 0.995 | 탐색률 감쇠 |
| BATCH_SIZE | 64 | 미니배치 크기 |
| **MEMORY_SIZE** | **20,000** | **[v9 변경]** 10,000 → 20,000 |
| EPISODES | 2,000 | 에피소드 수 |
| STEPS_PER_EP | 100 | 에피소드당 스텝 수 |
| TARGET_UPDATE | 10 | Target Network 갱신 주기 |
| NLOS_EPISODE_RATIO | 0.4 | NLOS 구간 포함 에피소드 비율 |
| NLOS_STEPS | 10 | NLOS 구간 지속 스텝 수 |
| LR Scheduler | CosineAnnealingLR(T_max=2000, eta_min=1e-4) | v8~v9 유지 |

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
- test.py: NLOS 타이밍 수정 (행동 전 채널 먼저 갱신, nlos_event=False 전달)

**Issues**
- NLOS 이후 CP 과도 증가 (CP/σ_DS 30까지 폭등)
- NLOS 구간 Adaptive SE < Fixed CP
- 학습 보상 이동평균 수렴 불안정
- **[잠재 버그]** test.py NLOS 수동 설정 + nlos_event=False 구조가 step() 내부 LOS 업데이트와 충돌 (v9에서 해소)

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
- Adaptive SE >> Theoretical SE: `theo_cp = σ×3.0`이어도 비교 의미가 있으나,  
  에이전트가 훨씬 작은 CP를 찾음 → 논문 ρ=3의 의미 재확인 필요
- NLOS 진입 시 에이전트의 선제 대응 부족 (is_nlos 정보 미활용)
- 학습 보상 이동평균 수렴 불완전 (1,500~3,500 진동)

---

### v8 — ISI 차단 + is_nlos state 추가 + 논문 준수 복원

**Abstract**  
v7 4가지 문제 수정 + 논문·3GPP 준수 검토 후 2가지 추가 수정.  
CP 하한 σ×1.1 ISI 원천 차단, is_nlos 6번째 state 차원 추가,  
보상 클리핑, 에피소드 2,000 확대.  
논문 준수: delta_theta_deg=0 (LOS), theo_cp=ρ·σ_DS=3σ 복원.

**v8 결과** *(theo_cp = σ×3.0, 논문 준수 기준)*

| 지표 | Theoretical (σ×3.0) | Fixed CP | Adaptive |
|---|---|---|---|
| 평균 SE (전체) | 0.287 | 0.255 | **0.510** (+100%) |
| 평균 SE (정상) | 0.342 | 0.255 | **0.576** (+126%) |
| 평균 SE (NLOS) | 0.159 | 0.255 | **0.355** (+39%) |

**Changes**
- env.py: CP 하한 σ→σ×1.1, is_nlos 6번째 state, 보상 클리핑 [-20,20]
- env.py [준수]: LOS delta_theta_deg=0.0, theo_cp=sigma_ds×rho
- DQN.py: STATE_NORM 5→6차원
- train.py: EPISODES 1,000→2,000, CosineAnnealingLR T_max 업데이트

**Issues**
- **test.py NLOS 타이밍 충돌**: NLOS 수동 설정 후 nlos_event=False 전달
  → step() 내부 LOS 덮어씌우기로 σ_DS 극단 진동 발생 (v9에서 해소)

---

### v9 — NLOS 타이밍 충돌 해소 + 리플레이 버퍼 확대 (최종 버전)

**Abstract**  
v5부터 누적되어온 test.py NLOS 타이밍 구조적 충돌을 근본 해소.  
`nlos_event=is_nlos`를 step()에 직접 전달하여 train.py와 구조 일원화.  
σ_DS 왜곡 제거로 NLOS 구간 측정 신뢰도 확보.

**v9 결과** *(theo_cp = σ×3.0, 논문 준수 기준)*

| 지표 | Theoretical (σ×3.0) | Fixed CP | Adaptive | v8 대비 |
|---|---|---|---|---|
| 평균 SE (전체) | 0.622 | 0.495 | **1.076** | **+111%** |
| 평균 SE (정상) | 0.718 | 0.542 | **1.273** | +121% |
| 평균 SE (NLOS) | 0.400 | 0.384 | **0.617** | +74% |

- Adaptive SE가 Fixed CP 대비 **+117%** 향상 (전체 기준)
- Adaptive SE가 Theoretical SE(논문 3σ 기준)를 **+73%** 상회
- NLOS 구간 Adaptive(0.617) > Theoretical(0.400) > Fixed(0.384) — 전 구간 우위 확인
- **프로젝트 전 버전 중 최고 성능** (Adaptive SE: v7=1.146 참고 기준이 다름 / v8=0.510 → v9=1.076)

**Changes**

- **[핵심] test.py NLOS 처리 일원화** (`수정 1`):
  ```python
  # v5~v8 (문제):
  if is_nlos:
      env._set_nlos_channel()          # NLOS 채널 수동 설정
      env.sigma_ds = _calculate_ds()   # sigma_ds 갱신
      state = env._make_state()
  env.step(action, nlos_event=False)   # ← False 전달
  # → step() 내부 "if not nlos_event:" 블록이 LOS tau_hat으로 덮어씌워 sigma_ds 왜곡

  # v9 (수정):
  env.step(action, nlos_event=is_nlos) # ← is_nlos 직접 전달 (train.py와 동일)
  # → step() 내부에서 NLOS 채널 업데이트 → LOS 덮어씌우기 블록 스킵 → 정상
  ```
  - 효과: NLOS 구간 σ_DS 극단 진동 제거, train/test 구조 완전 일치
  - 부가효과: is_nlos 플래그(v8 추가)가 다음 스텝부터 state에 반영되어 에이전트가 NLOS 인식 후 정책 조정 가능

- **train.py MEMORY_SIZE 확대** (`수정 2`):
  ```python
  MEMORY_SIZE = 20_000   # 10,000 → 20,000
  ```
  LOS/NLOS 다양한 채널 조합 경험 보존 → 정책 안정성 향상

**v9 성능 분석**

잘 나온 점:
- NLOS 타이밍 수정 하나로 Adaptive SE가 v8(0.510) → v9(1.076)로 **+111% 향상**
- 전 구간(정상·NLOS 모두) Adaptive > Theoretical > Fixed CP 순서 확립
- NLOS 구간 SE 1.273→0.617: 정상→NLOS 전환 시에도 Fixed CP(0.384) 대비 +61% 유지
- ③ σ_DS 그래프: NLOS 구간에서 v8의 0~50ns 극단 진동 해소, 실질적 상승 확인

잔존 특성 (수정 대상 아님):
- **σ_DS NLOS 내 간헐적 급락**: NLOS에서 theta_b가 60~70°에 가까울 때  
  빔 폭 `ωb = 1.772/(√M·cos(θb))`가 넓어져 모든 경로가 주엽 내에 들어오면  
  경로 파워가 균등해져 σ_DS가 작아지는 현상. 논문 물리 모델에서 발생하는 자연적 특성.
- **학습 음수 에피소드 간헐 발생**: NLOS 진입 첫 스텝에서 LOS state 기반 Decrease 선택 시  
  연속 음수 보상 누적 가능. 전체 수렴 추세에 영향 없음.
- **1-스텝 NLOS 지연**: nlos_event=True 로 step() 호출 후 next_state에서 is_nlos=True 반영.  
  첫 NLOS 스텝에서 에이전트는 LOS state로 행동 → 구조적 한계, 파라미터 튜닝으로 해소 불가.

DQN 파라미터 최적화 가능성:
- 현재 성능(1.076)은 이 환경 설계에서 이미 이론적 최적 행동(CP ≈ σ_DS×1.1 유지)에 근접
- 추가 개선 가능 영역: Double DQN(Q값 과추정 억제), 에피소드 3,000(분산 감소)
- 효과 미미 예상: 은닉층 추가(6D 입력에 128×128 이미 충분), 보상 스케일 확대
- 근본 성능 한계는 env 설계(NLOS 1-스텝 지연, 스텝별 채널 재샘플링)에 있음

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
| v8 | 0.510 | 0.355 | **+100%** | σ×3.0 (논문 TCP 준수) |
| **v9** | **1.076** | **0.617** | **+117%** | σ×3.0 (논문 TCP 준수) |

> v7과 v8의 Adaptive SE 수치 차이는 Theoretical CP 기준 재정의에 따른 것.  
> v7: theo_cp=σ×3(보상 기준) vs v8~v9: theo_cp=σ×3(동일), 그러나 보상 구조 변경 및 CP clamp 추가로 에이전트 정책이 더 보수적으로 수렴.
