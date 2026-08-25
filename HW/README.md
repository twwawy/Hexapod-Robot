# 하드웨어 구성

Jetson Orin Nano Super와 STM32 NUCLEO-F446RE를 기반으로 제작하는 6족 보행 로봇 프로젝트입니다.

Jetson Orin Nano Super는 LiDAR와 Depth Camera를 이용한 인지·자율주행을 담당하고, STM32 NUCLEO-F446RE는 센서 수집, 200 Hz 보행 제어, 서보·릴레이 출력과 통신을 담당한다.

## 주요 구성

| Category | Main Parts |
|---|---|
| Main Computer | Jetson Orin Nano Super |
| MCU | STM32 NUCLEO-F446RE |
| Controller | RadioMaster Pocket ELRS |
| Actuator | DS51150-270 Servo Motor |
| LiDAR | Livox Mid-360 |
| Depth Camera | Intel RealSense D435 |
| IMU | WT931 9DOF IMU |
| GPS | NEO-M8N GPS Module |
| Communication | RYLR998 LoRa Module |
| Battery | 3S / 2S 7200mAh Battery |
| Frame | 3D Printed PLA Parts |

자세한 수량과 구매 정보는 [부품 목록](parts.md), STM32 핀과 주변장치 설정은 [STM32F446RE 설정 정리본](../SW/STM32/STM32F446RE%20설정%20정리본.md)을 참고한다.

기구와 제작 자료는 다음 폴더에 있다.

현재 루트 `mjx/` 학습 scene의 로봇 총질량 `10.0 kg`은 시뮬레이션 목표값이다.
완성된 실물의 배터리·배선·센서를 포함한 질량은 별도로 계측해 갱신해야 한다.

| 경로 | 내용 |
|---|---|
| [stl_file](stl_file) | 3D 프린팅용 STL |
| [gcode_file](gcode_file) | 출력용 G-code와 3MF |
| [PCB](PCB) | PCB Gerber와 Drill 파일 |
| [urdf](urdf) | 현재 URDF와 시뮬레이션 설정 |
| [urdf_earlyVersion](urdf_earlyVersion) | 초기 URDF 보관본 |
