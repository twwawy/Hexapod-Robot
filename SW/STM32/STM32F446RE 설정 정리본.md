# STM32F446RE Configuration

이 문서는 Hexapod Robot의 STM32F446RE 최종 설정을 정리한 문서입니다.  
STM32F446RE는 서보모터 PWM 출력, MCP3008 아날로그 입력, 릴레이 제어, Jetson SPI 통신, GPS, IMU, LoRa, ELRS 수신기 입력을 담당합니다.

---

## 1. Board / MCU

| 항목 | 설정 |
|---|---|
| Board | NUCLEO-F446RE |
| MCU | STM32F446RE |
| Development Environment | STM32CubeMX / STM32CubeIDE |
| Debug | Serial Wire / SWD |
| System Clock | 84 MHz |
| APB1 Timer Clock | 84 MHz |
| APB2 Timer Clock | 84 MHz |

PA15, PB3, PB4 핀을 PWM 출력으로 사용하기 때문에 CubeMX에서 다음 설정이 필요합니다.

- `SYS → Debug = Serial Wire`

JTAG를 그대로 사용하면 PA15, PB3, PB4 핀을 PWM 핀으로 사용할 수 없습니다.

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

타이머 클럭이 84 MHz이므로 Prescaler를 83으로 설정하면 타이머 카운터 주파수는 1 MHz가 됩니다.

`84 MHz / (83 + 1) = 1 MHz`

따라서 카운터 1칸은 1 us입니다.

`4999 + 1 = 5000 us = 5 ms`

즉, PWM Compare 값은 us 단위와 동일하게 사용할 수 있습니다.

| Pulse Width | 의미 |
|---:|---|
| 500 | 최소 위치 |
| 1500 | 중립 위치 |
| 2500 | 최대 위치 |

---

## 3. PWM Pin Map

총 18개의 PWM 채널을 사용합니다.  
각 다리는 3개의 서보모터로 구성됩니다.

| Leg | Joint | Timer Channel | Pin |
|---|---|---|---|
| Leg 1 | 1_1 | TIM1_CH1 | PA8 |
| Leg 1 | 1_2 | TIM1_CH2 | PA9 |
| Leg 1 | 1_3 | TIM1_CH3 | PA10 |
| Leg 2 | 2_1 | TIM2_CH1 | PA15 |
| Leg 2 | 2_2 | TIM2_CH2 | PB3 |
| Leg 2 | 2_3 | TIM2_CH3 | PB10 |
| Leg 3 | 3_1 | TIM3_CH1 | PB4 |
| Leg 3 | 3_2 | TIM3_CH2 | PB5 |
| Leg 3 | 3_3 | TIM3_CH3 | PB0 |
| Leg 4 | 4_1 | TIM4_CH1 | PB6 |
| Leg 4 | 4_2 | TIM4_CH2 | PB7 |
| Leg 4 | 4_3 | TIM4_CH3 | PB8 |
| Leg 5 | 5_1 | TIM5_CH1 | PA0 |
| Leg 5 | 5_2 | TIM5_CH2 | PA1 |
| Leg 5 | 5_3 | TIM8_CH3 | PC8 |
| Leg 6 | 6_1 | TIM1_CH4 | PA11 |
| Leg 6 | 6_2 | TIM3_CH4 | PB1 |
| Leg 6 | 6_3 | TIM4_CH4 | PB9 |

사용하는 타이머는 다음과 같습니다.

| Timer | Used Channels |
|---|---|
| TIM1 | CH1, CH2, CH3, CH4 |
| TIM2 | CH1, CH2, CH3 |
| TIM3 | CH1, CH2, CH3, CH4 |
| TIM4 | CH1, CH2, CH3, CH4 |
| TIM5 | CH1, CH2 |
| TIM8 | CH3 |

---

## 4. MCP3008 Analog Input

MCP3008 3개를 사용하여 총 24개의 아날로그 입력을 읽습니다.

- MCP3008 1개: 8채널
- MCP3008 3개: 24채널

### SPI1 Pin Map

| 기능 | STM32 Pin | MCP3008 Pin |
|---|---|---|
| SPI1_SCK | PA5 | CLK |
| SPI1_MISO | PA6 | DOUT |
| SPI1_MOSI | PA7 | DIN |
| CS1 | PA4 | MCP3008 #1 CS |
| CS2 | PC0 | MCP3008 #2 CS |
| CS3 | PC1 | MCP3008 #3 CS |

### SPI1 Setting

| 항목 | 설정 |
|---|---|
| Mode | Full-Duplex Master |
| Data Size | 8 Bits |
| First Bit | MSB First |
| CPOL | Low |
| CPHA | 1 Edge |
| NSS | Software |
| Prescaler | 32 |
| SPI Speed | 약 2.625 Mbit/s |
| CRC | Disable |

CS 핀은 GPIO Output으로 직접 제어합니다.

- 기본 상태: CS HIGH
- 읽을 MCP3008만 CS LOW
- 통신 종료 후 CS HIGH

---

## 5. MCP3008 Channel Assignment

### MCP3008 #1

| CS Pin | Channel | Signal |
|---|---|---|
| CS1 / PA4 | CH0 | 1_1 |
| CS1 / PA4 | CH1 | 1_2 |
| CS1 / PA4 | CH2 | 1_3 |
| CS1 / PA4 | CH3 | Leg 1 Pressure Sensor |
| CS1 / PA4 | CH4 | 2_1 |
| CS1 / PA4 | CH5 | 2_2 |
| CS1 / PA4 | CH6 | 2_3 |
| CS1 / PA4 | CH7 | Leg 2 Pressure Sensor |

### MCP3008 #2

| CS Pin | Channel | Signal |
|---|---|---|
| CS2 / PC0 | CH0 | 3_1 |
| CS2 / PC0 | CH1 | 3_2 |
| CS2 / PC0 | CH2 | 3_3 |
| CS2 / PC0 | CH3 | Leg 3 Pressure Sensor |
| CS2 / PC0 | CH4 | 4_1 |
| CS2 / PC0 | CH5 | 4_2 |
| CS2 / PC0 | CH6 | 4_3 |
| CS2 / PC0 | CH7 | Leg 4 Pressure Sensor |

### MCP3008 #3

| CS Pin | Channel | Signal |
|---|---|---|
| CS3 / PC1 | CH0 | 5_1 |
| CS3 / PC1 | CH1 | 5_2 |
| CS3 / PC1 | CH2 | 5_3 |
| CS3 / PC1 | CH3 | Leg 5 Pressure Sensor |
| CS3 / PC1 | CH4 | 6_1 |
| CS3 / PC1 | CH5 | 6_2 |
| CS3 / PC1 | CH6 | 6_3 |
| CS3 / PC1 | CH7 | Leg 6 Pressure Sensor |

---

## 6. Relay Control

INA, INB, INC는 릴레이 제어용 GPIO 출력으로 사용합니다.

| 기능 | STM32 Pin | 설정 |
|---|---|---|
| INA | PC2 | GPIO Output |
| INB | PC3 | GPIO Output |
| INC | PC4 | GPIO Output |

### GPIO Setting

| 항목 | 설정 |
|---|---|
| Output Level | Low |
| Mode | Output Push Pull |
| Pull-up / Pull-down | No Pull |
| Speed | Low |

### Relay Logic

| GPIO State | Relay State |
|---|---|
| LOW | OFF |
| HIGH | ON |

초기 출력은 LOW로 설정하여 전원 인가 시 릴레이가 자동으로 켜지지 않도록 합니다.

---

## 7. Jetson Orin Nano Super ↔ STM32 SPI Communication

Jetson Orin Nano Super는 SPI Master, STM32F446RE는 SPI Slave로 동작합니다.

| 기능 | Jetson 40-Pin Header | STM32F446RE |
|---|---|---|
| SPI MOSI | Pin 19 | PB15 / SPI2_MOSI |
| SPI MISO | Pin 21 | PB14 / SPI2_MISO |
| SPI SCLK | Pin 23 | PB13 / SPI2_SCK |
| SPI CS0 | Pin 24 | PB12 / SPI2_NSS |
| DRDY | Pin 22 / GPIO Input | PC9 / GPIO Output |
| GND | Pin 6 or Pin 9 | GND |

### SPI2 Setting

| 항목 | 설정 |
|---|---|
| Mode | Full-Duplex Slave |
| Data Size | 8 Bits |
| First Bit | MSB First |
| CPOL | Low |
| CPHA | 1 Edge |
| NSS | Hardware Input |
| CRC | Disable |

### DRDY Signal

DRDY는 STM32가 Jetson에게 데이터 준비 완료 상태를 알려주는 신호입니다.

동작 순서는 다음과 같습니다.

1. STM32가 상태 데이터 준비
2. STM32가 DRDY HIGH 출력
3. Jetson이 DRDY HIGH 확인
4. Jetson이 CS LOW 출력
5. Jetson과 STM32가 SPI 통신
6. 통신 완료 후 STM32가 DRDY LOW 출력

Jetson이 SPI Master이므로 STM32가 데이터를 보내고 싶어도 Jetson이 Clock을 발생시키지 않으면 통신이 진행되지 않습니다.  
따라서 DRDY 핀을 사용해 Jetson에게 데이터 준비 상태를 알려줍니다.

---

## 8. USART2 / GPS

GPS 모듈은 USART2에 연결합니다.

| 항목 | 설정 |
|---|---|
| Module | NEO-M8N GPS Module / TYE-GP001 |
| USART | USART2 |
| TX | PA2 |
| RX | PA3 |

### USART2 Setting

| 항목 | 설정 |
|---|---|
| Mode | Asynchronous |
| Baud Rate | 9600 |
| Word Length | 8 Bits |
| Parity | None |
| Stop Bits | 1 |
| Data Direction | TX/RX |
| Hardware Flow Control | Disable |
| Over Sampling | 16 Samples |

### Wiring

| GPS | STM32 |
|---|---|
| GPS TX | PA3 / USART2_RX |
| GPS RX | PA2 / USART2_TX |
| GND | GND |

---

## 9. USART3 / IMU

IMU 모듈은 USART3에 연결합니다.

| 항목 | 설정 |
|---|---|
| Module | BMI160 + RM3100 9DOF IMU Module |
| USART | USART3 |
| TX | PC10 |
| RX | PC11 |

### USART3 Setting

| 항목 | 설정 |
|---|---|
| Mode | Asynchronous |
| Baud Rate | 9600 |
| Word Length | 8 Bits |
| Parity | None |
| Stop Bits | 1 |
| Data Direction | TX/RX |
| Hardware Flow Control | Disable |
| Over Sampling | 16 Samples |

### Wiring

| IMU | STM32 |
|---|---|
| IMU TX | PC11 / USART3_RX |
| IMU RX | PC10 / USART3_TX |
| GND | GND |

---

## 10. UART5 / LoRa Module

LoRa 통신 모듈은 UART5에 연결합니다.

| 항목 | 설정 |
|---|---|
| Module | RYLR998 UART LoRa Module |
| UART | UART5 |
| TX | PC12 |
| RX | PD2 |

### UART5 Setting

| 항목 | 설정 |
|---|---|
| Mode | Asynchronous |
| Baud Rate | 115200 |
| Word Length | 8 Bits |
| Parity | None |
| Stop Bits | 1 |
| Data Direction | TX/RX |
| Hardware Flow Control | Disable |
| Over Sampling | 16 Samples |

### Wiring

| LoRa | STM32 |
|---|---|
| LoRa TX | PD2 / UART5_RX |
| LoRa RX | PC12 / UART5_TX |
| GND | GND |

---

## 11. USART6 / ELRS Receiver

RadioMaster Pocket ELRS 조종기와 ELRS Nano 수신기를 사용합니다.  
STM32는 ELRS 수신기의 CRSF UART 출력을 USART6으로 수신합니다.

| 항목 | 설정 |
|---|---|
| Transmitter | RadioMaster Pocket ELRS |
| Receiver | ELRS-2.4G-NANO ESP8285 / DarwinFPV 2.4G ELRS Nano |
| Wireless Link | ELRS |
| Receiver Output Protocol | CRSF UART |
| USART | USART6 |
| TX | PC6 |
| RX | PC7 |

실제로 필요한 핀은 PC7 / USART6_RX입니다.  
PC6 / USART6_TX는 CubeMX 설정상 잡혀 있어도 실제 배선하지 않아도 됩니다.

### USART6 Setting

| 항목 | 설정 |
|---|---|
| Mode | Asynchronous |
| Baud Rate | 420000 |
| Word Length | 8 Bits |
| Parity | None |
| Stop Bits | 1 |
| Data Direction | Receive Only |
| Hardware Flow Control | Disable |
| Over Sampling | 16 Samples |

### Wiring

| ELRS Receiver | STM32 |
|---|---|
| Receiver TX / CRSF TX | PC7 / USART6_RX |
| GND | GND |

---

## 12. TIM6 / 5 ms Control Loop Timer

TIM6은 5 ms 제어 루프용 기본 타이머로 사용합니다.

| 항목 | 설정 |
|---|---|
| Timer | TIM6 |
| Prescaler | 83 |
| Period | 4999 |
| Period Time | 5 ms |
| Interrupt | TIM6_DAC_IRQn Enable |

타이머 클럭이 84 MHz이므로 다음과 같습니다.

`84 MHz / (83 + 1) = 1 MHz`

`4999 + 1 = 5000 us = 5 ms`

TIM6 인터럽트 시작 코드는 다음과 같습니다.

```c
HAL_TIM_Base_Start_IT(&htim6);
```

5 ms마다 실행할 제어 루프는 다음 콜백 함수 안에서 처리합니다.

```c
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6)
    {
        // 5 ms control loop
    }
}
```

---

## 13. Software Implementation Policy

본 프로젝트의 STM32 코드는 HAL 기반으로 작성하되, 빠른 처리가 필요한 일부 구간에서는 레지스터 직접 접근 방식을 사용할 수 있습니다.

| 항목 | 구현 방식 |
|---|---|
| 초기화 코드 | STM32CubeMX 생성 코드 사용 |
| GPIO 초기화 | HAL 사용 |
| UART 초기화 | HAL 사용 |
| SPI 초기화 | HAL 사용 |
| PWM 초기화 | HAL 사용 |
| PWM 값 갱신 | HAL Macro 또는 CCR 직접 접근 |
| MCP3008 CS 제어 | HAL_GPIO_WritePin 또는 BSRR 직접 접근 |
| 제어 루프 | TIM6 Interrupt |
| 센서 데이터 관리 | 구조체 기반 |
| Jetson 통신 데이터 | Packet 구조체 기반 |

---

## 14. Recommended Code Structure

```text
Core/
├─ Inc/
│  ├─ main.h
│  ├─ servo.h
│  ├─ mcp3008.h
│  ├─ relay.h
│  ├─ jetson_spi.h
│  ├─ gps.h
│  ├─ imu.h
│  ├─ lora.h
│  ├─ crsf.h
│  └─ control_loop.h
│
└─ Src/
   ├─ main.c
   ├─ servo.c
   ├─ mcp3008.c
   ├─ relay.c
   ├─ jetson_spi.c
   ├─ gps.c
   ├─ imu.c
   ├─ lora.c
   ├─ crsf.c
   └─ control_loop.c
```

---

## 15. Servo Control Policy

서보 PWM 출력은 500 ~ 2500 us 범위로 제한합니다.

```c
#define SERVO_MIN_US      500
#define SERVO_CENTER_US   1500
#define SERVO_MAX_US      2500
```

각 서보모터는 다음과 같은 구조체로 관리합니다.

```c
typedef struct
{
    TIM_HandleTypeDef *htim;
    uint32_t channel;
    uint16_t pulse_us;
} Servo_t;
```

PWM 갱신은 기본적으로 HAL Macro를 사용합니다.

```c
__HAL_TIM_SET_COMPARE(servo->htim, servo->channel, pulse_us);
```

속도가 필요한 경우 CCR 레지스터에 직접 접근할 수 있습니다.

```c
TIM1->CCR1 = 1500;
TIM1->CCR2 = 1500;
TIM1->CCR3 = 1500;
TIM1->CCR4 = 1500;
```

처음 구현할 때는 유지보수성을 위해 `__HAL_TIM_SET_COMPARE()` 사용을 권장합니다.

---

## 16. Main Loop Concept

전체 동작 구조는 다음과 같습니다.

1. `HAL_Init()`
2. `SystemClock_Config()`
3. GPIO / SPI / UART / TIM 초기화
4. PWM 채널 Start
5. UART Receive Interrupt 또는 DMA Start
6. SPI Slave Receive 준비
7. TIM6 5 ms Interrupt Start
8. `while(1)`에서 저속 작업 처리

### 5 ms Control Loop

TIM6 인터럽트에서는 짧고 주기적인 작업만 처리합니다.

- MCP3008 센서값 읽기
- 서보 목표값 계산
- PWM 출력 갱신
- 릴레이 상태 갱신
- Jetson 송신용 상태 데이터 갱신
- DRDY 제어

### Background Loop

`while(1)`에서는 상대적으로 느리거나 문자열 처리가 필요한 작업을 처리합니다.

- GPS NMEA Parsing
- IMU Packet Parsing
- LoRa AT Command 처리
- CRSF Packet Parsing
- Jetson 명령 처리
- Debug 출력

---

## 17. Peripheral Summary

| 기능 | Peripheral | Pin |
|---|---|---|
| MCP3008 SPI | SPI1 | PA5, PA6, PA7 |
| MCP3008 CS1 | GPIO Output | PA4 |
| MCP3008 CS2 | GPIO Output | PC0 |
| MCP3008 CS3 | GPIO Output | PC1 |
| Jetson SPI | SPI2 Slave | PB12, PB13, PB14, PB15 |
| Jetson DRDY | GPIO Output | PC9 |
| GPS | USART2 | PA2, PA3 |
| IMU | USART3 | PC10, PC11 |
| LoRa | UART5 | PC12, PD2 |
| ELRS CRSF | USART6 RX | PC7 |
| Relay INA | GPIO Output | PC2 |
| Relay INB | GPIO Output | PC3 |
| Relay INC | GPIO Output | PC4 |
| Control Loop | TIM6 Interrupt | - |

---

## 18. Important Notes

- PA15, PB3, PB4를 사용하기 위해 `SYS → Debug = Serial Wire`로 설정해야 합니다.
- 모든 PWM 타이머는 Prescaler 83, Period 4999로 설정합니다.
- PWM Compare 값은 us 단위와 동일하게 사용합니다.
- MCP3008 CS 핀은 GPIO Output으로 직접 제어합니다.
- Jetson과 STM32의 GND는 반드시 공통으로 연결해야 합니다.
- SPI2는 STM32가 Slave이므로 Jetson이 Clock을 넣어야 통신이 진행됩니다.
- DRDY는 STM32가 Jetson에게 데이터 준비 상태를 알려주는 용도로 사용합니다.
- GPS, IMU, LoRa, ELRS 수신기 배선 시 TX와 RX는 교차 연결합니다.
- ELRS CRSF 수신은 USART6 RX만 사용하면 됩니다.
- TIM6 인터럽트 안에는 시간이 오래 걸리는 문자열 파싱이나 블로킹 코드를 넣지 않는 것이 좋습니다.
