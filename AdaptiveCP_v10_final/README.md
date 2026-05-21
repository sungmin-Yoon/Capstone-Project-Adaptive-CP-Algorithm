# AdaptiveCP v10 — DQN 기반 적응형 CP 조절 알고리즘

> **논문:** "Impact of Beamforming on Delay Spread in Wideband Millimeter-wave Systems"  
> Akgun, Krunz, Manzi — ICNC 2020  
> **표준:** 3GPP TR 38.900 (채널 모델 파라미터)

---

## 1. 프로젝트 개요

고층 건물, 터널, 산간 지형 등 장애물이 많은 NLOS(Non-Line-of-Sight) 환경에서는 전파의 반사·굴절·회절로 인해 다중 경로 오류(Multipath Error)가 발생한다. 특히 UAV·자율주행차 등 고속 이동체에서는 채널의 DS(Delay Spread)가 동적으로 변화하는 시간 선택적 페이딩에 노출되어, 통신 단절 시 제어 상실로 이어지는 2차 사고 위험이 존재한다.

기존 OFDM 시스템은 고정 길이의 CP(Cyclic Prefix)를 사용한다. CP가 DS보다 짧으면 ISI(Inter-Symbol Interference)가 발생해 BER이 급증하고, 반대로 CP를 과도하게 길게 설정하면 유효 데이터 전송 구간이 줄어 주파수 및 전력 효율이 저하된다.

본 연구는 DQN(Deep Q-Network) 기반 강화학습을 활용하여 채널의 RMS Delay Spread 추정치에 따라 CP 길이를 실시간으로 적응 조절하는 알고리즘을 구현한다. 레퍼런스 논문의 빔포밍 채널 모델(키홀 안테나 모델)과 3GPP TR 38.900 UMa DS 통계를 시뮬레이션 환경으로 삼아, ISI를 방지하면서 Spectral Efficiency(SE)를 극대화하는 것이 목표이다.

---

## 2. 파일 구조

```
AdaptiveCP_v10/
├── env.py       — 강화학습 환경 (채널 모델, PDP, DS 계산, 보상 함수)
├── DQN.py       — 신경망 구조 (3층 FC, 입력 정규화)
├── train.py     — 학습 루프 (ε-greedy, Experience Replay, Target Network)
├── test.py      — 평가 및 차트 저장
├── dqn_adaptive_cp.pth    (train 후 생성)
├── train_rewards.npy      (train 후 생성)
└── charts/                (test 후 생성)
```

### 실행

```powershell
py -3.11 train.py
py -3.11 test.py
```

`train.py`와 `test.py` 상단의 `FIXED_M` 값을 동일하게 설정해야 한다.

```python
FIXED_M = None   # None = 매 에피소드마다 랜덤 (M ∈ {16, 64, 256})
FIXED_M = 16     # M = 16 고정
FIXED_M = 64     # M = 64 고정
FIXED_M = 256    # M = 256 고정
```

---

## 3. 채널 모델 및 수식 (논문 기반)

본 프로젝트는 레퍼런스 논문의 수식과 파라미터를 변형 없이 그대로 사용한다.

### 3.1 Power Delay Profile (논문 Eq. 5)

```
Pc(τ) = (1 / 2τ̂) · exp(−2τ / τ̂) · [G(θ⁻(τ)) + G(θ⁺(τ))]

θ⁻(τ) = θ_c − κ · √(τ / τ̂)
θ⁺(τ) = θ_c + κ · √(τ / τ̂)
```

- `τ̂` : 평균 초과 지연 (mean excess delay)
- `θ_c` : 클러스터 중심 도래각
- `κ = 5.0°` : 각도 확산 계수 (논문 고정값)

### 3.2 RMS Delay Spread 계산 (논문 Eq. 6–7)

```
μ_DS  = ∫ τ · Pc(τ) dτ  /  ∫ Pc(τ) dτ
σ_DS  = √[ ∫ τ² · Pc(τ) dτ / ∫ Pc(τ) dτ  −  μ_DS² ]
```

수치 적분(`scipy.integrate.quad`)으로 매 스텝 계산하며, 채널 상태 변화가 즉시 반영된다.

### 3.3 UPA 빔포밍 모델 (논문 Eq. 3–4)

```
ωb = 1.772 / (√M · cos θb)    — 빔폭
G_m = π · M                   — 메인로브 이득
g_m = G_m × 10^(−13.26/10)   — 사이드로브 이득
```

안테나 배열 M ∈ {16, 64, 256}이 빔폭과 이득에 직접 영향을 준다.  
M이 클수록 빔이 좁아져 LOS 이득이 높아지지만, 빔 미정렬(NLOS) 시 SE 손실이 커진다.

### 3.4 Spectral Efficiency (논문 Eq. 8)

```
SE = (T_s / (T_s + T_CP)) · log₂(1 + SNR)
```

CP가 σ_DS보다 짧으면(CP < σ_DS) ISI 발생으로 SE를 50% 패널티 처리한다.

### 3.5 논문 기준 Theoretical CP

```
T_CP = ρ · σ_DS    (ρ = 3.0, 논문 reference [16])
Fixed CP = τ̂_ref × ρ = 25 ns × 3.0 = 75 ns
```

---

## 4. 3GPP TR 38.900 채널 파라미터

채널 초기화 시 DS 통계는 3GPP TR 38.900 Table 7.5-6 UMa 모델을 따른다.

| 시나리오 | μ (log₁₀[DS/s]) | σ (log₁₀[DS/s]) | SNR 범위 | 빔 정렬 |
|---|---|---|---|---|
| LOS | −7.03 | 0.66 | 10 ~ 25 dB | θ_b = θ_c, Δθ = 0° |
| NLOS | −6.44 | 0.39 | 3 ~ 12 dB | θ_b ∈ [30°,70°], θ_c ∈ [0°,20°] |

`τ̂`는 3GPP DS 샘플로부터 다음과 같이 환산한다.

```
τ̂ = DS_3gpp × (τ̂_ref / DS_LOS_median)
τ̂_ref = 25 ns,  DS_LOS_median = 10^(−7.03) s
```

---

## 5. 강화학습 구조

### 5.1 State 벡터 (6차원)

| 인덱스 | 변수 | 물리적 의미 | 정규화 기준 |
|---|---|---|---|
| 0 | σ_DS (ns) | RMS Delay Spread | 200 ns |
| 1 | SNR (dB) | 수신 SNR | 30 dB |
| 2 | Δθ (°) | 빔 정렬 오차 | 45° |
| 3 | M | 안테나 배열 수 | 256 |
| 4 | T_CP (ns) | 현재 CP 길이 | 600 ns |
| 5 | is_nlos | NLOS 플래그 (0 / 1) | 1 |

모든 입력은 DQN 내부에서 정규화 기준값으로 나눠 [0, 1] 범위로 스케일링된다.

### 5.2 Action 공간 (3가지)

| Action | 동작 |
|---|---|
| 0 | CP 감소 (Decrease CP) |
| 1 | CP 유지 (Maintain CP) |
| 2 | CP 증가 (Increase CP) |

스텝 크기는 현재 CP / σ_DS 비율에 따라 동적으로 결정된다.

```
CP/σ_DS < 0.8  →  step = σ_DS × 0.5   (빠른 증가 필요)
CP/σ_DS < 1.5  →  step = σ_DS × 0.2   (미세 조정)
CP/σ_DS < 5.0  →  step = σ_DS × 0.3
CP/σ_DS ≥ 5.0  →  step = σ_DS × 0.5   (과도 CP 빠른 감소)
```

CP 하한은 항상 `σ_DS × 1.1` 이상으로 클램핑하여 ISI를 원천 차단한다.

### 5.3 Reward 함수

```python
reward = (SE_adaptive − SE_fixed) × 20.0     # Fixed CP 대비 SE 개선이 기본 보상

if 1.1·σ_DS ≤ CP ≤ 2.0·σ_DS:
    reward += 3.0                             # 효율적 CP 범위 보너스

if CP > 3.0·σ_DS:
    overrun = CP / (3.0·σ_DS) − 1.0
    reward −= overrun × 5.0                  # 논문 기준값 초과 패널티

reward = clip(reward, −20.0, 20.0)
```

설계 의도: Fixed CP 대비 SE를 높이는 방향을 학습시키되, CP가 논문 기준값(3σ)을 초과해 불필요하게 길어지는 것을 억제한다. 1.1σ ~ 2.0σ 구간에 안착하면 보너스를 부여하여 ISI 안전성과 효율성을 동시에 달성하도록 유도한다.

### 5.4 DQN 신경망 구조

```
입력 (6차원, 정규화)
    │
FC(6 → 128) + ReLU
    │
FC(128 → 128) + ReLU
    │
FC(128 → 3)  — 각 Action의 Q-value 출력
```

### 5.5 학습 하이퍼파라미터

| 파라미터 | 값 |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 (CosineAnnealingLR, η_min = 1e-4) |
| Discount Factor γ | 0.99 |
| ε-greedy | 1.0 → 0.01 (decay = 0.995) |
| Batch Size | 64 |
| Replay Memory | 20,000 |
| Episodes | 2,000 |
| Steps / Episode | 100 |
| Target Network 갱신 주기 | 10 에피소드 |
| Gradient Clipping | max_norm = 1.0 |
| NLOS 에피소드 비율 | 40% |
| NLOS 지속 스텝 수 | 10 스텝 |

---

## 6. 최종 평가 결과

> 테스트 조건: FIXED_M = None (랜덤), 30 스텝, NLOS 구간 Step 10–18 (9스텝)

### 평균 Spectral Efficiency (bps/Hz)

| 방식 | 평균 SE | Fixed CP 대비 |
|---|---|---|
| Theoretical CP (= 3·σ_DS) | 0.622 | 기준값 |
| Fixed CP (75 ns, 고정) | 0.495 | — |
| **Adaptive CP (DQN)** | **1.076** | **+117%** |

### 구간별 SE 분석

| 구간 | Adaptive SE |
|---|---|
| 정상 구간 (LOS 위주) | 1.273 bps/Hz |
| NLOS 급변 구간 | 0.617 bps/Hz |

### 핵심 관찰

**① ISI 위반 0건**  
CP 하한 클램프(σ_DS × 1.1)가 NLOS 급변 구간에서도 ISI를 완전히 차단하였다.

**② Adaptive SE > Theoretical SE**  
Theoretical CP = 3σ_DS는 항상 3배 여유를 두는 보수적 기준이다. DQN은 1.1σ ~ 2.0σ 범위의 최적 운용 포인트를 학습하여, 불필요한 CP 오버헤드를 제거하고 SE를 72% 추가 개선하였다.

**③ Fixed CP와의 비교 (+117%)**  
Fixed CP(75 ns)는 LOS 환경에서 σ_DS 대비 지나치게 긴 값으로 설정되어 SE 손실이 발생한다. DQN은 채널 상태에 맞게 CP를 능동적으로 줄여 전송 효율을 극대화하였다.

**④ NLOS 동적 대응 확인**  
NLOS 이벤트 발생 시 CP를 빠르게 증가시켜 ISI를 억제하고, NLOS 해소 직후 CP를 즉시 감소시켜 SE를 회복하는 동적 거동이 테스트에서 관찰되었다.