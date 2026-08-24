# Hexapod Robot

인하대학교 로봇연구회가 2024년부터 개발하고 있는 6족 보행 로봇 프로젝트이다. Jetson Orin Nano Super가 인지와 자율주행을 담당하고, STM32 NUCLEO-F446RE가 센서 수집과 200 Hz 실시간 보행 제어를 담당하는 구조를 목표로 한다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| [HW](HW/README.md) | 기구, PCB, 부품, URDF와 제작 파일 |
| [SW/Controller](SW/Controller/Controller_Architecture.md) | MATLAB/Simulink 보행 제어기 |
| [SW/STM32](SW/STM32/STM32F446RE%20설정%20정리본.md) | STM32 설정과 펌웨어 |
| [SW/Jetson](SW/Jetson/README.md) | Jetson 상위 제어 소프트웨어 |

## 주요 문서

- [제어기 Architecture](SW/Controller/Controller_Architecture.md)
- [제어기 상세 설계](SW/Controller/Controller_detail.md)
- [좌표계와 관절 정의](SW/Controller/좌표축/README.md)
- [드론 조종기 입력](SW/Controller/드론%20조종기%20입력/README.md)
- [STM32F446RE 설정](SW/STM32/STM32F446RE%20설정%20정리본.md)
- [하드웨어 부품 목록](HW/parts.md)
