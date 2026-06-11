# Trailer Angle Estimator

이 폴더는 고정된 양쪽 사이드미러 ROI에서 트레일러 옆면의 `TRAILER` 마커를 잡고, 트레일러 각도를 추정하기 위한 프로토타입입니다.

핵심 판단은 이렇습니다.

- 지금은 라벨 데이터가 없으므로 딥러닝보다 마커 기반 기하 알고리즘이 맞습니다.
- 사이드미러는 고정되어 있으니 카메라마다 ROI를 한 번만 잡습니다.
- 거울 반사 때문에 마커 원본과 좌우 반전 템플릿을 둘 다 SIFT로 매칭합니다.
- 실제 거울 영상처럼 homography가 무너지는 경우에는 마커 위 매칭점이 넓게 뭉치는지 보는 `feature_cluster` fallback을 사용합니다.
- 타이어/조명/선반을 마커로 착각하지 않도록 직사각형 contour fallback은 기본으로 꺼져 있습니다.
- 캘리브레이션 CSV가 없으면 `visual_proxy` 각도입니다. 실제 articulation angle로 쓰려면 반드시 각도 샘플을 모아 선형 보정을 켜야 합니다.

## 빠른 테스트

저장된 한 카메라 프레임:

```bash
cd /home/ircv02/HYU-ECL3003/rover
python3 trailer_task/estimate_image.py snapshot_cam0_000.jpg --camera cam0 --output trailer_task/out_cam0.jpg --json
python3 trailer_task/estimate_image.py snapshot_cam1_000.jpg --camera cam1 --output trailer_task/out_cam1.jpg --json
```

듀얼 카메라 창을 캡처한 넓은 스크린샷은 자동으로 좌/우 반으로 나눕니다.

```bash
python3 trailer_task/estimate_image.py "trailer_task/example/Screenshot from 2026-06-05 20-10-17.png" --json
```

실시간 듀얼 카메라:

```bash
cd /home/ircv02/HYU-ECL3003/rover
python3 trailer_task/live_dual_trailer_angle.py
```

Tag36h11 실시간 pose/XYZ축 모니터링:

```bash
cd /home/ircv02/HYU-ECL3003/rover
python3 trailer_task/live_tag36h11_pose.py --tag-size-mm 30
```

로버에서 효율 우선으로 볼 때는 VPI CUDA 전처리와 작은 검출 입력을 씁니다. 태그가 작아 인식이 끊기면 `--process-scale 0.75` 또는 `--process-scale 1.0`으로 올리고, 태그가 나오는 영역을 알면 ROI를 잘라 CPU 부하를 줄입니다.

```bash
python3 trailer_task/live_tag36h11_pose.py --tag-size-mm 30 --process-scale 0.5 --preprocess vpi-cuda
python3 trailer_task/live_tag36h11_pose.py --tag-size-mm 30 --roi 0.25,0.15,0.50,0.70
python3 trailer_task/live_tag36h11_pose.py --dual --tag-size-mm 30 --use-config-roi
```

SSH에서 실행하면 웹 모니터가 자동으로 켜집니다. 기본 포트는 `8765`이고, 이미 사용 중이면 다음 포트를 잡습니다. 웹 스트림은 최신 프레임만 유지하고, 접속자가 없으면 JPEG 인코딩을 멈추며, 송출 이미지는 낮은 해상도로 줄입니다.

```bash
python3 trailer_task/live_tag36h11_pose.py --dual --tag-size-mm 30 \
  --preprocess vpi-cuda --process-scale 0.5 --use-config-roi
```

직접 포트/품질을 정하고 싶으면:

```bash
python3 trailer_task/live_tag36h11_pose.py --dual --tag-size-mm 30 \
  --preprocess vpi-cuda --process-scale 0.5 --use-config-roi \
  --web-port 8765 --web-width 720 --web-fps 10 --web-jpeg-quality 45
```

같은 네트워크에서는 터미널에 출력된 `same-network` 주소를 열고, SSH 터널을 쓸 때는 로컬 PC에서 아래처럼 접속한 뒤 출력된 `localhost` 주소를 엽니다.

```bash
ssh -L 8765:localhost:8765 ircv02@ROVER_IP
```

창 조작은 `q` 종료, `s` 현재 화면 저장입니다. 화면에는 tag id, X/Y/Z 위치, roll/pitch/yaw 각도, 그리고 X(red)/Y(green)/Z(blue) 축이 같이 표시됩니다.

기존 트레일러 각도 스크립트 `live_dual_trailer_angle.py`의 기본값은 속도를 위해 `960x540` 캡처, 검출 `8`프레임 주기입니다. 더 빠르게 보려면:

```bash
python3 trailer_task/live_dual_trailer_angle.py --capture-width 800 --capture-height 450 --estimate-every 12 --inactive-estimate-every 36
```

실시간 스크립트는 `camera_live_dual.py`와 같은 좌우 가장자리 red/magenta 톤 보정을 기본으로 적용합니다. 보정 때문에 느린지 비교하려면:

```bash
python3 trailer_task/live_dual_trailer_angle.py --no-edge-color-fix
```

창 조작:

- `q`: 종료
- `s`: 현재 표시 화면 저장

## ROI 조정

`config.yaml`의 `cameras.cam1.roi`, `cameras.cam0.roi`를 수정합니다.

현재 기본값:

- `cam1`: 왼쪽 사이드미러 영역, `x=0.00, w=0.40`
- `cam0`: 오른쪽 사이드미러 영역, `x=0.60, w=0.40`

ROI는 정규화 좌표입니다. 예를 들어 1280x720 프레임에서 `x=0.60, w=0.40`이면 x=768부터 오른쪽 끝까지 봅니다.

## 실제 각도로 캘리브레이션

후진 제어에는 proxy 각도가 아니라 실제 힌지/articulation 각도가 필요합니다. 다음 순서로 샘플을 모으세요.

1. 차량과 트레일러를 일직선으로 놓고 각도계를 0도로 맞춥니다.
2. 좌/우로 `-45, -30, -15, 0, 15, 30, 45`도 정도를 만들어 각 카메라 프레임을 저장합니다.
3. 각 저장 이미지마다 아래처럼 CSV에 추가합니다.

```bash
python3 trailer_task/estimate_image.py sample_cam0_m30.jpg --camera cam0 --angle-deg -30 --append-calibration
python3 trailer_task/estimate_image.py sample_cam1_p30.jpg --camera cam1 --angle-deg 30 --append-calibration
```

카메라마다 5개 이상 샘플이 쌓이면 `linear_calibration` 모델이 자동으로 사용됩니다. 이후 `--json` 출력의 `model`이 `linear_calibration`인지 확인하세요.

## 후진 직각 주차/주행까지 가는 추천 구조

1. 인식 계층: 이 폴더의 추정기가 `angle_deg`, `confidence`, `source`를 10Hz 이상 출력합니다.
2. 상태 계층: confidence가 낮으면 마지막 값을 짧게 유지하고, 0.5초 이상 끊기면 후진 제어를 멈춥니다.
3. 제어 계층: 저속 후진에서 트레일러 각도 목표값을 먼저 제어하고, 그 다음 차체 경로를 제어합니다.
4. 안전 제한: articulation이 `50~60도`를 넘으면 jackknife 위험으로 보고 즉시 조향을 풀거나 정지합니다.
5. 주차 전략: 직각 주차는 처음부터 한 번에 넣으려 하지 말고 `진입 각도 만들기 -> 트레일러 꺾기 -> 각도 유지 후 밀어넣기 -> 차체 정렬` 네 단계로 나눕니다.

현장에서 제일 중요한 것은 마커가 ROI 안에서 충분히 크게 보이도록 거울 각도와 마커 위치를 정하는 것입니다. 마커가 프레임 높이의 8% 미만으로 작거나 절반 이상 잘리면 각도 품질이 급격히 떨어집니다.

후진 직각 주차 제어 구조는 `PARKING_STRATEGY.md`에 따로 정리했습니다.
