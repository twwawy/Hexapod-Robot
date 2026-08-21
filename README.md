# Hexapod-Robot

6족 로봇의 하드웨어 자산과 MuJoCo MJX 기반 보행 실험을 함께 관리하는 저장소다. 현재 강화학습의 기준 경로는 **classical tripod gait + Cartesian residual RL**이다.

## 현재 기준 경로

```text
command → nominal tripod gait → nominal foot targets
        → RL Δz (swing legs only) → contact/safety → posture layer
        → linearized IK → joint limits → PD torque
```

- RL action: 다리 순서 `LF, LM, LB, RF, RM, RB`의 6차원 swing-foot `Δz`
- RL은 stance 발, 보폭, 착지 XY, gait timing, body pose를 직접 바꾸지 않는다.
- early landing은 contact/safety 계층이 현재 발 위치를 유지해 RL residual을 무시한다.
- 기존 7-D residual checkpoint는 action/observation 계약이 달라 재사용할 수 없다. `fresh`로 새 학습을 시작해야 한다.

설계·관측·보상·실행 방법은 [docs/RESIDUAL_RL.md](docs/RESIDUAL_RL.md)에만 최신 기준으로 정리한다.

## 주요 위치

- `SW/mjx/hexapod_mjx/residual_controller.py`: nominal gait, residual, contact safety, IK
- `SW/mjx/hexapod_mjx/residual_env.py`: MJX observation, reward, termination
- `SW/mjx/train_residual_ppo.py`: PPO 학습 및 checkpoint 계약
- `SW/mjx/visualize_residual_policy.py`: policy replay/render
- `Hexapod-MJX-가이드/residual_rl_run.sh`: 학습·재개·영상 생성 wrapper
- `HW/`: 실제 로봇의 URDF, CAD, PCB, 부품 자료

## 빠른 시작

```bash
cd ~/Hexapod-Robot
./Hexapod-MJX-가이드/빠른학습.sh fresh
```

작은 검증 실행 후 본 학습으로 확장한다. 자세한 명령과 평가 기준은 [Residual RL 가이드](docs/RESIDUAL_RL.md)를 따른다.

## 참고 자료

`완전 튜토리얼.md`는 초기 MuJoCo/Isaac 학습을 위한 배경 자료다. 현재 MJX residual 구현의 명세나 실행 기준은 [docs/RESIDUAL_RL.md](docs/RESIDUAL_RL.md)다.
