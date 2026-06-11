# Trailer Reversing and Right-Angle Parking Strategy

이 문서는 지금 만든 각도 추정기를 실제 후진 직각 주차와 저속 주행에 붙일 때의 구조입니다.

## 1. 각도 정의부터 고정

제어에 들어가기 전에 부호를 하나로 고정해야 합니다.

- `theta = 0 deg`: 차량과 트레일러가 일직선
- `theta > 0`: 위에서 봤을 때 트레일러가 차량 기준 왼쪽으로 꺾임
- `theta < 0`: 트레일러가 차량 기준 오른쪽으로 꺾임

현재 `config.yaml`의 `mirror_sign`은 이 부호를 맞추기 위한 임시값입니다. 실제 각도계를 놓고 캘리브레이션 CSV를 만들면 부호와 스케일이 같이 보정됩니다.

## 2. 인식 파이프라인

권장 루프:

```text
camera frame
  -> fixed mirror ROI
  -> SIFT marker homography
  -> geometric features
  -> calibrated angle model
  -> confidence gate
  -> low-pass filter
  -> controller
```

제어에 넘길 값은 항상 다음 3개를 묶어서 보냅니다.

```text
angle_deg, confidence, age_ms
```

`confidence < 0.18` 또는 `age_ms > 500`이면 후진 자동 제어를 멈추는 쪽이 안전합니다.

## 3. 후진 제어의 핵심

트레일러 후진은 일반 차 후진과 다릅니다. 조향 입력이 처음에는 트레일러 각도를 키우고, 각도가 커진 뒤에는 jackknife로 빠르게 넘어갑니다.

따라서 제어는 한 번에 "주차칸 중심으로 가기"가 아니라 두 층으로 나누는 것이 좋습니다.

```text
outer loop: 원하는 trailer angle theta_ref를 만든다
inner loop: 현재 theta가 theta_ref를 따라가도록 steering을 제한한다
```

초기에는 단순 P 제어로 시작하세요.

```text
steer_cmd = clamp(k_theta * (theta_ref - theta), -steer_limit, +steer_limit)
```

후진 중에는 steering과 trailer angle 변화의 부호가 직관과 반대로 느껴질 수 있으므로, 처음 0도 근처에서 아주 천천히 움직이며 부호를 확인해야 합니다.

## 4. 직각 주차 단계

실차/로버에서 가장 안정적인 방식은 상태 머신입니다.

```text
APPROACH
  주차칸 옆으로 천천히 전진하며 시작 위치 확보

BREAK_ANGLE
  후진 시작, theta_ref를 +/-25~35도로 만들어 트레일러를 주차칸 쪽으로 꺾음

HOLD_ANGLE
  theta를 제한하면서 트레일러 바퀴/후미가 칸 안으로 들어가게 유지

CHASE_TRAILER
  차량이 트레일러를 따라가며 theta_ref를 0도로 천천히 줄임

STRAIGHTEN
  theta가 0도 근처가 되면 차체까지 정렬

STOP
  confidence 저하, 장애물, theta 한계, 목표 도달 시 정지
```

처음 목표값은 보수적으로 잡는 편이 좋습니다.

- `theta_ref`: 25도부터 시작
- `theta_soft_limit`: 45도
- `theta_hard_limit`: 55~60도
- 후진 속도: 가능한 최저속

## 5. 데이터 수집 계획

최소 캘리브레이션:

```text
cam0: -45, -30, -15, 0, 15, 30, 45 deg
cam1: -45, -30, -15, 0, 15, 30, 45 deg
```

가능하면 각 각도에서 3장씩 저장하세요. 조명/거리/흔들림이 조금씩 달라져야 모델이 덜 예민해집니다.

주행 데이터:

- 직선 후진 30초
- 왼쪽으로 약하게 꺾는 후진 30초
- 오른쪽으로 약하게 꺾는 후진 30초
- 마커가 일부 가려지는 실패 케이스 20장

실패 케이스도 중요합니다. 검출기가 "모른다"고 말해야 할 장면을 알아야 제어가 안전해집니다.

## 6. 마커/거울 세팅

현재 `TRAILER` 마커도 동작하지만, 실제 주행 안정성을 최대로 올리려면 다음 조건을 맞추세요.

- 마커는 가능한 한 평평하게 붙임
- ROI 안에서 마커 높이가 프레임 높이의 8~25% 정도가 되게 함
- 거울에는 마커 전체가 자주 들어오게 함
- 밝은 반사/검은 타이어 패턴과 마커가 겹치지 않게 함
- 비상용으로 좌우 두 카메라 중 하나만 살아도 각도가 나오게 함

장기적으로는 `TRAILER` 글자 마커 옆에 ArUco/AprilTag 2~4개를 같이 붙이는 방식이 더 좋습니다. 태그는 homography가 더 안정적이고, 부분 가림에도 어느 태그가 살아있는지 판단하기 쉽습니다.

## 7. 다음 구현 순서

1. `live_dual_trailer_angle.py`로 실제 거울 ROI를 맞춥니다.
2. 각도계를 써서 `calibration_samples.csv`를 만듭니다.
3. `model=linear_calibration` 상태에서 좌우 왕복 테스트를 합니다.
4. confidence 끊김/마커 가림 시 정지 로직을 먼저 붙입니다.
5. 그 다음 저속 후진의 `theta_ref` 추종만 테스트합니다.
6. 마지막에 직각 주차 상태 머신을 붙입니다.
