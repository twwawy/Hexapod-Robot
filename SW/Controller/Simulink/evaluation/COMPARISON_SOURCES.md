# 사진 8 비교 수치와 출처

이 문서는 이전 속도·발끝·관절 참고 그림의 출처이다. 최신 요청으로 만든 가상 Roll·Pitch 그림의 출처와 수치는 [가상 몸체 자세와 문헌 실측 참고값](BODY_STABILITY_COMPARISON.md)을 참고한다.

작성일: 2026-08-31. 현재 Classical의 Simulink 실행 결과와 RHex, PhantomX AX, HAntR의 공개 문헌 수치를 참고 비교했다. 서로 같은 조건으로 실행한 벤치마크가 아니며 Residual-RL 결과는 포함하지 않는다.

| 로봇 | 실제 주행 최대 속도 (m/s) | 발끝 RMS 오차 (mm) | 관절 RMS 오차 (°) | 근거 |
|---|---:|---:|---:|---|
| 현재 Classical | 미평가 | 3.385 | 0.400 | 저장된 `metrics.json`, 몸체 고정 시뮬레이션 |
| RHex | 0.55 | 미확보 | 미확보 | [1], Table 1 |
| PhantomX AX | 0.29 | 미확보 | 미확보 | [2], Table 1 |
| HAntR | 0.43 | 미확보 | 미확보 | [2], Table 1 |

현재 Classical 오차는 2026-08-31에 실행한 `plant.slx`의 7 ≤ t < 10 s에서 계산했다. 발끝은 6개 다리의 3차원 목표–응답 거리, 관절은 18개 관절의 속도 제한 후 목표–응답 각도에 대해 전체 RMS를 계산했다. 실제 로봇 측정값은 아니다.

## 문헌

1. U. Saranli, M. Buehler, D. E. Koditschek, **RHex: A Simple and Highly Mobile Hexapod Robot**, *The International Journal of Robotics Research*, 20(7), 616–631, 2001. [연구기관 원문 PDF](https://www.ri.cmu.edu/pub_files/pub4/saranli_uluc_2001_1/saranli_uluc_2001_1.pdf). Table 1, 논문 p.617의 `V (m/s)` 열에서 RHex 0.55를 확인했다. 표 주석은 V를 최대 속도로 정의한다.
2. P. Čížek et al., **Design, Construction, and Rough-Terrain Locomotion Control of Novel Hexapod Walking Robot With Four Degrees of Freedom Per Leg**, *IEEE Access*, 2021. DOI: [10.1109/ACCESS.2021.3053492](https://doi.org/10.1109/ACCESS.2021.3053492). [저자 연구실 원문 PDF](https://comrob.fel.cvut.cz/papers/access21hantr.pdf). Table 1의 `Maximum speed [m s−1]` 열에서 PhantomX AX 0.29와 HAntR 0.43을 확인했다. PhantomX 값은 해당 표가 기존 문헌에서 인용한 값이고, HAntR은 이 연구에서 제안한 플랫폼이다.

## 해석 제한

- 속도는 각 논문의 해당 플랫폼에 대해 보고된 값이다. 크기·질량·관절 구조·제어기·지면 조건이 다르며 현재의 최신 최고 성능을 의미하지 않는다.
- 현재 Classical 모델은 몸체가 월드에 고정되어 실제 주행 최대 속도를 평가할 수 없다. 내부 위치 추정기의 전진 평균 속도 **0.2349 m/s**를 실제 속도 막대에 넣거나 최대 속도로 바꾸지 않았다.
- 문헌의 발끝·관절 RMS가 ‘미확보’인 것은 동일 정의의 수치를 선택 출처에서 확보하지 못했다는 뜻이다. 해당 로봇이 부정확하거나 해당 연구 전반에 평가가 없다는 뜻이 아니다.
- 전진 평균 속도와 최대 속도, 가상 IMU와 실제 몸체 흔들림, 발끝 높이 변화폭과 장애물 극복 높이를 서로 대체하지 않는다.
- 따라서 ‘현재 로봇이 기존 로봇보다 몇 % 우수하다’거나 ‘Residual-RL로 개선되었다’는 결론은 이 자료로 내릴 수 없다.

## 재생성

이 폴더에서 MATLAB 함수 `make_figure_08`을 실행하면 `metrics.json`과 코드에 명시된 문헌 수치를 이용해 PNG, FIG, CSV를 다시 만든다. CSV의 `NaN`은 미평가 또는 미확보 값이다. 원래 Simulink 모델은 변경하거나 재실행하지 않는다.
