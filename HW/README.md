# Hexapod-Robot

Jetson Orin Nano Super와 STM32 NUCLEO-F446RE를 기반으로 제작하는 6족 보행 로봇 프로젝트입니다.

Jetson Orin Nano Super는 LiDAR, Depth Camera 등 각종 센서 데이터를 종합하여 이동 경로와 보행 방법을 계산하고, STM32 NUCLEO-F446RE는 GPS, IMU, LoRa 통신 등 하위 센서 및 통신 제어를 담당합니다.

## Hardware Overview

| Category | Main Parts |
|---|---|
| Main Computer | Jetson Orin Nano Super |
| MCU | STM32 NUCLEO-F446RE |
| Actuator | DS51150-270 Servo Motor |
| LiDAR | Livox Mid-360 |
| Depth Camera | Intel RealSense D435 |
| IMU | WT931 9DOF IMU |
| GPS | NEO-M8N GPS Module |
| Communication | RYLR998 LoRa Module |
| Battery | 3S / 2S 7200mAh Battery |
| Frame | 3D Printed PLA Parts |

자세한 부품 목록은 [parts.md]를 참고하세요.
