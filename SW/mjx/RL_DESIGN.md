# MJX Classical Whole-Body + Residual RL

전체 실행법은 [docs/RESIDUAL_RL.md](../../docs/RESIDUAL_RL.md)를 따른다. 기준 제어
순서는 원본 `SW/Controller/Controller_Architecture.md`의 2차 완성본과 같다.

```text
continuous body command
  -> x/y Position PI + Heading PI
  -> Final Body Twist (rate/workspace limited)
  -> controller-owned Tripod: stance PULL + quintic/cubic-Bezier swing
  -> airborne-aware Early/Late contact adaptation
  -> bounded swing XYZ / stance Z foot residual
  -> single posture PI + inverse 6-DOF body pose overlay
  -> whole-body workspace candidate accept/hold
  -> final workspace projection + analytical IK
  -> joint jump hold + 315.8 deg/s limiter + fixed ±8 Nm force cap
```

- action: `classical_wbc_cartesian_body6d_residual_v1`, 24-D
  - `[0:18]`: 여섯 발의 swing XYZ / stance Z-only residual
  - `[18:24]`: body forward, lateral, height, roll, pitch, yaw residual
- observation: `gt_attitude_collision_contact6_coarse9_touchdown6_v3`, 113-D
- gait phase, stride, frequency, swing height `0.20 m`, radial offset `0.07 m`는 제어기 소유
- 0.5 s가 지나도 swing 3발 착지가 끝나지 않으면 late-landing 탐색 중 phase를 유지
- stance-anchor 위치 추정은 직전 추정값+FK만 사용하고 simulator world 발 위치는 사용하지 않음
- contact는 압력/높이 proxy 없이 MuJoCo foot–world terrain collision만 사용
- roll/pitch/yaw 측정은 IMU 없이 MuJoCo root ground truth 사용
- body 자세 후보는 6개 다리가 모두 IK/관절 범위 안일 때만 세 회전축을 함께 채택
- residual은 nominal touchdown이나 Tripod phase를 바꾸지 않는다.
- terrain: 짧은 flat baseline → rough 우선 → stairs, 전체 계단 상승은 최대 `0.20 m`

필수 invariant:

```text
Safety > Contact > bounded Residual > Nominal
Tripod_Enable=0 -> six-foot stance hold (zero target가 아님)
policy action cannot change gait phase/frequency/stride
posture overlay is applied immediately before leg-frame IK
stair_riser * stair_count <= 0.20 m
```
