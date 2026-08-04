# mujoco_tuto / Hexapod-Robot

이 저장소에는 두 흐름이 함께 들어 있다.

1. 원격 `mujoco_tuto`에 있던 MuJoCo/MJCF 공부용 자료
2. 현재 `Hexapod-Robot`의 URDF/MJX 기반 hexapod 실험 코드와 가이드

## Hexapod MJX 작업물

- 메인 코드: `SW/mjx/`
- 가이드/실행 스크립트: `Hexapod-MJX-가이드/`
- 하드웨어/URDF 자산: `HW/urdf/`

빠른 실행 예시:

```bash
~/Desktop/Hexapod-MJX-가이드/빠른학습.sh fresh
~/Desktop/Hexapod-MJX-가이드/큰병렬.sh fresh --num-envs 512 --rollout-steps 128 --num-updates 500 --minibatch-size 2048
~/Desktop/Hexapod-MJX-가이드/자세튜닝.sh viewer
```

## 기존 mujoco_tuto 공부 자료

- `00_mjx_minimal.py`
- `mjx_tutorial.ipynb`
- `SPIDER_MUJOCO_STUDY_GUIDE.md`
- `reference/spider_rl/`
- `reference/mujoco_playground/tutorials/mjx_pendulum.py`

## 원격에 있던 기존 설명

MuJoCo/MJCF 공부용 메모 저장소다. 이번에 실제로 돌아가게 만든 HEXAPEDAL 변환 스택의 소스 오브 트루스는 이 폴더가 아니라 아래 repo다.

- 실제 코드 repo: `/home/huro/spider_ws/spider_rl`
- MuJoCo 패키지: `/home/huro/spider_ws/spider_rl/source/spider_rl/spider_mujoco`
- 학습/검증 스크립트: `/home/huro/spider_ws/spider_rl/scripts/mujoco`
- 원본 URDF 기본 경로: `/home/huro/spider_ws/HEXAPEDAL_URDF_description/urdf/HEXAPEDAL_URDF_fixed.urdf`
- 이 repo 안에도 업로드용 슬림 코드 사본을 `reference/spider_rl/` 아래 넣어뒀다. 공부/열람용이고, 수정 소스 오브 트루스는 여전히 위 `spider_rl` repo다.
