# STM32F446RE Hexapod Final Configuration

이 문서는 Hexapod Robot의 STM32F446RE 설정을 정리한 문서입니다.  
STM32는 18개의 서보모터 PWM 출력, MCP3008 기반 아날로그 입력 24채널, 릴레이 제어, Jetson 통신, GPS, IMU, LoRa, ELRS 수신기 입력을 담당합니다.

---

## 1. Board / MCU

| 항목 | 설정 |
|---|---|
| Board | NUCLEO-F446RE |
| MCU | STM32F446RE |
| Development Tool | STM32CubeMX / STM32CubeIDE |
| Debug | Serial Wire / SWD |
| System Clock | 84 MHz |
| APB1 Timer Clock | 84 MHz |
| APB2 Timer Clock | 84 MHz |

PA15, PB3, PB4 핀을 PWM으로 사용하기 때문에 `SYS → Debug` 설정은 반드시 `Serial Wire`로 설정해야 합니다.  
JTAG를 그대로 사용하면 PA15, PB3, PB4를 타이머 PWM 핀으로 사용할 수 없습니다.

---

## 2. Servo Motor PWM Setting

| 항목 | 설정 |
|---|---|
| Servo Motor | DS51150-270 |
| Control Method | PWM |
| PWM Period | 5 ms |
| PWM Frequency | 200 Hz |
| Neutral Pulse | 1500 us |
| Pulse Range | 500 ~ 2500 us |
| Timer Prescaler | 83 |
| Counter Period | 4999 |
| Initial Pulse | 1500 |

타이머 클럭이 84 MHz이므로 Prescaler를 83으로 설정하면 다음과 같습니다.

```text
84 MHz / (83 + 1) = 1 MHz
