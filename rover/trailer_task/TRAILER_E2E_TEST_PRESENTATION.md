---
marp: true
theme: default
paginate: true
---

# Trailer Task E2E Driving Test

BEV Lane + Trailer Angle 기반 Temporal End-to-End 주행 정책

2026-06-11

---

# 발표 목표

- 기존 rule-based trailer lane following의 한계를 정리한다.
- 실선/점선 lane segmentation과 trailer angle을 이용한 E2E 주행 정책을 설명한다.
- 데이터 수집, 모델 구조, 학습 결과, 실차 추론 결과를 공유한다.
- 현재 한계와 다음 개선 방향을 명확히 한다.

---

# 시스템 개요

입력 센서:

- 전방 좌/우 CSI 카메라 2대
- Trailer AI panel detection 기반 trailer angle 추정

중간 표현:

- SegFormer lane segmentation
- Dual-camera BEV 변환
- BEV lane mask: background / solid line / dashed line

출력:

- Skid-steer rover left/right motor command

---

# 기존 Rule-Based 주행 구조

기존 주행 pipeline:

```text
Camera
 -> lane segmentation
 -> BEV transform
 -> lane geometry estimation
 -> route/corner/pivot controller
 -> wheel mixer
 -> left/right motor
```

장점:

- 해석 가능하고 디버깅이 쉽다.
- straight, corner, pivot 같은 상태를 사람이 직접 조정할 수 있다.

한계:

- 직선 구간을 코너로 오인하는 경우가 있었다.
- 점선 복구, pivot, post-corner bias 같은 상태 전환이 복잡했다.
- 좌/우 회전 dynamics와 트레일러 각도까지 rule로 모두 커버하기 어렵다.

---

# 핵심 문제

트레일러 주행은 현재 프레임 하나만으로 결정하기 어렵다.

예시:

- 코너 진입 전 lane shape
- pivot 중 dashed line loss/re-acquisition
- post-corner bias 복구
- trailer angle이 남아 있는 상태에서 점선으로 복귀

따라서 단일 이미지 기반 정책보다 temporal policy가 더 적합하다.

---

# E2E 학습 방향

입력:

- 최근 N프레임 BEV lane mask
- trailer angle sequence
- angle confidence / angle rate / age / valid flag

출력:

- left wheel command
- right wheel command

보조 출력:

- normal / straight / corner / pivot / bias mode classification

---

# 모델 Boundary

이번 E2E는 모든 perception을 한 번에 학습하지 않는다.

별도 모델:

- Lane segmentation model
- Trailer angle detection/calibration model

E2E policy 입력:

- lane segmentation 결과를 BEV mask로 변환한 것
- trailer angle estimator가 낸 scalar metadata

이렇게 나눈 이유:

- perception 문제와 control 문제를 분리할 수 있다.
- 적은 데이터로도 policy 학습이 가능하다.
- lane/trailer detector를 따로 개선할 수 있다.

---

# E2E Dataset 구성

저장 위치:

```text
logs/dotted_lane_following_run/*/e2e_dataset/
```

저장 항목:

- `bev_masks/*.png`
  - class id mask: 0 background, 1 solid, 2 dashed
- `bev_input/*.png`
  - high-contrast input: 0 background, 127 solid, 255 dashed
- `metadata.csv`
  - trailer angle
  - angle confidence/rate/source
  - final wheel_left / wheel_right
  - route/mode/debug metadata

---

# 수집 데이터 현황

총 E2E 학습 샘플:

```text
1,878 frames
```

Run별 샘플:

```text
20260611_175622: 229
20260611_175742: 1001
20260611_180337: 156
20260611_180438: 51
20260611_180529: 441
```

주의:

- 각 run은 좌회전 또는 우회전 위주로 따로 수집되었다.
- rover 좌우 무게중심이 달라 좌우 flip augmentation은 사용하지 않았다.

---

# Train / Validation Split

방향별 run 분포:

```text
left: 2 runs
right: 1 run
mixed: 2 runs
```

최종 split:

```text
train:
  20260611_175622
  20260611_175742
  20260611_180529

validation:
  20260611_180337
  20260611_180438
```

핵심 주의:

- 우회전 run은 1개뿐이라 train에 반드시 남겼다.
- 따라서 validation에는 우회전 일반화 검증이 아직 없다.

---

# 데이터 분포

Train samples:

```text
normal:   769
straight: 215
corner:   246
pivot:    142
bias:     292
total:   1664
```

Validation samples:

```text
normal:   111
straight: 15
corner:   33
pivot:    9
bias:     11
total:   179
```

Pivot/bias validation sample이 적기 때문에 mode별 성능은 참고용으로 해석해야 한다.

---

# Temporal Window 결정

로그 주기:

```text
median dt ~= 0.198 s
fps ~= 5 Hz
```

구간별 지속시간:

- pivot: 약 0.7-1.0 s
- corner: 약 2.2-2.5 s
- post-corner bias: 약 1.8-2.0 s

결론:

- 6 frames는 약 1.2 s라 pivot 순간만 보기에는 가능
- corner -> pivot -> bias 흐름까지 보려면 부족
- 최종 history는 10 frames, 약 2.0 s로 설정

---

# 모델 구조

사용 모델:

```text
BEV mask sequence
 -> CNN frame encoder
 -> trailer scalar embedding
 -> GRU temporal encoder
 -> motor head: wheel_left, wheel_right
 -> auxiliary mode head
```

입력 shape:

```text
history = 10
BEV input = 2 channels, 192 x 160
scalar features = 6
```

Scalar features:

- angle_deg
- angle_rate_deg_s
- angle_confidence
- angle_age_s
- angle_ok
- dt_s

---

# 왜 작은 CNN + GRU인가

데이터가 아직 작다.

```text
total: 1,878 frames
pivot train: 142 frames
validation pivot: 9 frames
```

큰 transformer/large CNN은 과적합 위험이 크다.

선택 기준:

- Jetson Orin에서 실시간 inference 가능
- 5 Hz 카메라 loop에서 충분히 가벼움
- pivot/bias의 짧은 temporal pattern을 볼 수 있음
- dataset이 늘어나면 모델 크기를 키울 수 있음

---

# Augmentation 전략

사용:

- trailer angle Gaussian noise
- angle rate noise
- BEV mask pixel dropout
- 작은 rectangle dropout
- 약한 morphology erosion/dilation

사용하지 않음:

- left/right flip
- 큰 rotation/translation
- motor output label noise

좌우 flip을 쓰지 않은 이유:

- rover 좌우 무게중심이 다르다.
- 좌회전/우회전 motor response가 대칭이 아니다.
- flip은 물리적으로 잘못된 데이터를 만들 수 있다.

---

# 학습 설정

Checkpoint:

```text
e2e_temporal_policy/runs/temporal_gru_20260611_182905/best.pt
```

학습:

```text
epochs: 80
batch size: 16
optimizer: AdamW
loss:
  Huber wheel loss
  + auxiliary mode cross entropy
```

Best checkpoint:

```text
epoch 36
```

---

# Offline Validation 결과

Best loss checkpoint, epoch 36:

```text
val_loss:        0.06598
wheel_left_mae: 0.03827
wheel_right_mae:0.04345
avg wheel MAE:  0.04086
mode_acc:       0.955
```

해석:

- wheel output scale 기준 평균 약 4% 오차
- offline imitation 기준으로는 좋은 성능
- 단, validation set이 작고 우회전 validation이 없다

---

# Mode별 Validation MAE

Epoch 36 기준:

```text
normal:   0.0148
straight: 0.0756
corner:   0.0829
pivot:    0.0668
bias:     0.1090
```

해석:

- normal은 매우 잘 맞는다.
- straight/corner/pivot은 사용 가능한 수준이다.
- bias가 가장 약하다.
- pivot/bias validation sample 수가 적어 과신하면 안 된다.

---

# Offline Replay Inference

Validation replay:

```text
samples: 179
avg wheel MAE: 0.0409
mode_acc: 0.955
```

전체 로그 replay:

```text
samples: 1843
avg wheel MAE: 0.0263
mode_acc: 0.992
```

주의:

- 전체 로그 replay는 train 포함이므로 높게 나오는 것이 정상이다.
- 실질적인 평가는 validation replay를 기준으로 봐야 한다.

---

# Live E2E Runner

새 실차용 파일:

```text
live_e2e_temporal_policy.py
```

모드:

- shadow
  - 기존 controller로 실제 주행
  - E2E output은 로그만 저장
- assist
  - 기존 controller wheel과 E2E wheel을 blending
- drive
  - E2E model wheel을 직접 송신

안전 처리:

- 10-frame warmup 전 wheel 송신 금지
- output scale
- max wheel clamp
- frame-to-frame delta clamp
- inactive 시 zero command

---

# Live Drive 설정

현재 설정:

```yaml
e2e_live_policy:
  checkpoint: e2e_temporal_policy/runs/temporal_gru_20260611_182905/best.pt
  mode: shadow
  require_full_history: true
  output_scale: 1.05
  max_abs_wheel: 0.87
  max_delta_per_frame: 0.30
  deadband: 0.02
```

출력 scale은 실차에서 조정한 tuning parameter다.

실차 테스트 중 사용한 scale:

```text
0.85 -> 1.20 -> 1.10 -> 1.05
```

---

# Live Test 결과

E2E drive run summary:

```text
e2e_live_20260611_193848
  scale 0.85, frames 1204, ready 1195

e2e_live_20260611_194546
  scale 1.20, frames 247, ready 238

e2e_live_20260611_194705
  scale 1.10, frames 333, ready 324

e2e_live_20260611_194846
  scale 1.05, frames 958, ready 949
```

최종에 가까운 run:

```text
scale 1.05
mean abs selected wheel: 0.295
max abs selected wheel: 0.87
```

---

# Live Log 구성

저장 위치:

```text
logs/dotted_lane_following_run/e2e_live_YYYYMMDD_HHMMSS/
```

주요 파일:

```text
dotted_lane_following_log.csv
e2e_policy_live_log.csv
run_config.json
```

`e2e_policy_live_log.csv` 주요 컬럼:

- e2e_raw_left / e2e_raw_right
- e2e_safe_left / e2e_safe_right
- rule_left / rule_right
- selected_left / selected_right
- e2e_pred_mode
- e2e_mode_confidence
- trailer_angle_deg

---

# 실차에서 관찰한 점

긍정적:

- E2E 직접 wheel 출력으로 실제 주행 가능했다.
- temporal model이 normal, corner, pivot, bias mode를 연속적으로 추론했다.
- 출력 scale을 조정하면서 주행 감도를 맞출 수 있었다.
- max wheel clamp 덕분에 pivot/강한 조향 구간에서 출력이 제한되었다.

주의:

- 출력이 너무 약하면 corner/pivot 반응이 늦다.
- 출력 scale을 너무 키우면 max clamp에 자주 걸리고 움직임이 거칠 수 있다.
- 우회전 validation 데이터가 부족하다.

---

# 현재 한계

데이터:

- 전체 데이터가 아직 작다.
- 우회전 run이 1개뿐이라 우회전 validation 불가.
- pivot/bias validation sample이 적다.

모델:

- rule-based controller imitation이므로 closed-loop 누적 오차 가능.
- 모델이 잘못된 상태로 들어갔을 때 회복 능력은 아직 제한적이다.

평가:

- offline MAE만으로 실제 주행 성공을 보장할 수 없다.
- live closed-loop 평가 metric이 더 필요하다.

---

# 다음 개선 방향

데이터 추가:

- 우회전 run 추가 수집
- corner -> pivot -> bias 구간을 더 많이 수집
- 실패/복구 케이스도 intentional하게 수집

모델 개선:

- bias 구간 sample weighting 조정
- motor MAE 기준 checkpoint 별도 저장
- TCN/GRU 비교
- 이전 wheel command를 scalar input으로 추가 검토

실차 평가:

- shadow mode에서 rule vs E2E 차이 분석
- closed-loop 성공률, lane loss 횟수, pivot 성공률 측정
- output scale / clamp / delta limit grid test

---

# 발표용 핵심 메시지

이번 테스트의 핵심은 perception 전체를 E2E로 학습한 것이 아니라,

```text
segmentation + trailer angle estimation
```

을 신뢰 가능한 중간 표현으로 두고,

```text
temporal control policy
```

를 E2E로 학습했다는 점이다.

결과적으로:

- 기존 복잡한 corner/pivot rule을 policy가 일부 흡수했다.
- 실차에서 E2E direct drive까지 가능했다.
- 다만 데이터 불균형과 우회전 검증 부족은 다음 단계의 핵심 과제다.

---

# 데모 / 영상 삽입 추천

이 슬라이드에는 다음 중 하나를 넣으면 좋다.

- BEV lane mask 화면
- trailer angle 추정 overlay
- E2E live run 영상
- `rule wheel` vs `E2E wheel` plot
- mode timeline: normal -> corner -> pivot -> bias

추천 로그:

```text
logs/dotted_lane_following_run/e2e_live_20260611_194846/
```

---

# Q&A

감사합니다.

