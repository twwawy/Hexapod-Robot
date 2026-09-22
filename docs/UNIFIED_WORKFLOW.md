# RL 통합 브랜치 사용법

현재 브랜치: **`RL/unified-latest`**. 기준일: 2026-09-22.
최신 main의 하드웨어·캘리브레이션·STM32를 유지하고, adaptive 브랜치의 최신 MJX 복구 로직을 합쳤다.
모든 기존 브랜치 tip은 이 브랜치의 조상 커밋이다. 원본 브랜치는 삭제하거나 수정하지 않는다.

## 처음 받기

```bash
git clone --branch RL/unified-latest --single-branch https://github.com/twwawy/Hexapod-Robot.git
cd Hexapod-Robot
./hexapod
./hexapod doctor
./hexapod check
```

`--single-branch`여도 전체 조상 이력을 받으므로 과거 버전 복원이 가능하다.
`--depth` shallow clone은 과거 복원·migration·이력 묶기에 부적합하며
`git fetch --unshallow origin`으로 보완한다. 브랜치 이름의 `RL`은 대문자다.

## 하나의 실행 명령

| 목적 | 명령 |
|---|---|
| 도움말 / 설치 상태 | `./hexapod` / `./hexapod doctor` |
| 소스·이력·CLI·STM32 native 검사 | `./hexapod check` |
| CPU adaptive 계약 검사까지 | `./hexapod check --rl` |
| 현재 hybrid 지형 curriculum | `./hexapod train [옵션]` |
| 평지 RC 전후·yaw·정지 curriculum | `./hexapod train-rc [옵션]` |
| 한 단계 PPO | `./hexapod train-stage [옵션]` |
| adaptive viewer | `./hexapod view --terrain flat --perception oracle` |
| 기존 18-D policy replay | `./hexapod replay [옵션]` |
| 검토된 v5 → v6 warm start | `./hexapod resume-v5 --checkpoint /절대/경로 [옵션]` |
| 검토된 v6 clearance warm start | `./hexapod resume-v6 --checkpoint /절대/경로 [옵션]` |
| ARM 빌드 (flash 안 함) | `./hexapod build [--compiler /경로/arm-none-eabi-gcc]` |
| 발전 단계 목록 / 원본 복원 | `./hexapod history` / `./hexapod history cartesian ../hexapod-cartesian` |
| 이력 보존 검증 | `./hexapod verify-history` |
| 최신 소스 + 전체 Git 이력 한 파일로 모으기 | `./hexapod bundle` |

기존 스크립트도 유지한다. 인자는 원래 프로그램으로 전달되고, 명시한 옵션이 기본값보다 우선한다.
`./hexapod --dry-run train --run-name my-run`으로 실제 명령을 먼저 볼 수 있다.
각 실행 명령 뒤 `--help`로 세부 옵션을 확인한다. 기본 도움말·dry-run은 학습이나 장치를 실행하지 않는다.

`train` 기본값은 `--profile hybrid --command-mode terrain --perception teacher`다.
처음부터 hybrid gait로 지형을 학습하며, RC curriculum과 성공 기준을 구분한다.
기존 curriculum의 커밋 고정 worktree·70% 승급·retry·W&B/영상 저장 정책을 유지한다.
실제 학습은 GPU 자원과 시간이 필요하며 이 통합 작업에서 PPO를 실행하지 않았다.

## 환경

Python 3.11–3.13 환경에서 기존 요구사항을 설치한다. 통합 검사는 Python 3.13 CPU에서 수행했다.
학습용 Linux NVIDIA GPU의 JAX/CUDA 설치는 장비에 맞춰 별도로 구성한다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r mjx/requirements-train.txt
./hexapod check --rl
```

MuJoCo와 mujoco-mjx는 학습 요구사항에서 둘 다 `3.12.0`으로 맞췄다.
모든 전이 의존성을 고정한 GPU 재현 환경은 아니며, 실제 학습 때 환경 정보를 따로 보관한다.
Python 선택 순서: `HEXAPOD_PYTHON` → 저장소 `.venv` → 활성 `VIRTUAL_ENV` →
`~/.venvs/hexapod-mjx` → `python3`. 다른 환경은
`HEXAPOD_PYTHON=/경로/python ./hexapod ...`로 지정한다.
macOS GUI viewer는 MuJoCo의 `mjpython` 실행 환경이 필요할 수 있다.
`check`에는 C 컴파일러와 Git이 필요하며 보드·GPU는 필요 없다.
현재 원본 main에서도 재현되는 STM32 native 3개 실패 때문에 `check`는 exit 1이다.
나머지 검사도 끝까지 실행하며 [검증 기록](INTEGRATION_VALIDATION.md)에 결과를 구분했다.

## 현재 계약과 실기 경계

- 현재 MJX: action 24-D v4, observation `adaptive_hybrid_elevation_grid24x24x6_path_v6`,
  actor 8790 / critic 9105, elevation CNN. 벽 접촉 분리·한 발 retract retry·free-Wave·stance recovery 포함.
- 원래 18-D policy는 `replay`에 남아 있다. 같은 차원이라고 임의의 checkpoint를 resume하지 않는다.
- 최신 복구 로직은 source hash와 scheduler 계약이 달라졌다. `--restore` 검사를 우회하지 않는다.
  `resume-v5`/`resume-v6`는 코드에서 검토된 소스만 받는 **fresh optimizer warm start**다.
  원본과 완전히 같은 v5 resume는 `history stair-v5 ...`로 해당 소스를 복원한다.
- STM32의 기본 통신은 **64-byte sensor/GPS v3**다. 별도의 **128-byte adaptive v3** codec/API가
  있어도 자동으로 활성화되지 않는다. `JetsonSpi_EnableV3`와 master 설정을 일치시켜야 하며
  자동 frame 길이 협상은 없다. [정확한 wire 규격](ADAPTIVE_SPI_V3.md).
- 새 MJX foot retry/free-Wave는 STM32에 자동 이식된 기능이 아니다. 물리 보행·DMA·PPO·계단 완주,
  실시간 센서/정책 배포는 이 통합 검사로 검증되지 않는다.

## 발전 과정과 단일 파일 백업

정확한 SHA와 통합 판단은 [발전 이력](DEVELOPMENT_HISTORY.md),
[기계 판독 manifest](integration-sources.json)에 있다.
`history`는 현재 checkout을 바꾸지 않고 지정 경로에 detached worktree를 만든다.
기존 경로를 덮어쓰지 않는다. 정리할 때는 `git worktree remove /해당/경로`를 사용한다.

```bash
./hexapod history cartesian ../hexapod-cartesian
./hexapod history stair-v5 ../hexapod-stair-v5
./hexapod bundle /원하는/경로/hexapod-source.bundle
git clone /원하는/경로/hexapod-source.bundle hexapod-restored
```

Bundle은 현재 HEAD에서 도달하는 **모든 소스 이력**을 포함한다. 원격 branch가 없어져도 원본 SHA를 복원할 수 있다.
로컬 학습 결과·외부 checkpoint·가상환경·커밋하지 않은 파일은 포함하지 않는다.
미커밋 파일이 있거나 출력 파일이 이미 존재하면 생성하지 않는다.
과거 빌드 바이너리도 Git 이력에 남아 있어 전체 bundle/clone 용량 자체는 줄지 않는다.

`.build`, STM32 `Debug` 및 Eclipse `.metadata` 1,018개 생성 파일은 현재 추적에서 제거했다.
`.ioc`, `.project`, `.cproject`, 드라이버, linker script 및 소스는 유지한다.
CubeIDE에서는 기존 프로젝트를 import해 빌드 결과를 재생성하거나 `./hexapod build`를 사용한다.
기존 개별 문서는 당시 단계의 기록이다. 현재 명령·버전·기본값은 이 문서를 기준으로 한다.
