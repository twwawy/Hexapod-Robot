# STM32F446RE 설정 정리본

---

## 1. 보드 / MCU

| 항목 | 설정 |
|---|---|
| 보드 | NUCLEO-F446RE |
| 개발 환경 | STM32CubeMX / STM32CubeIDE |
| 디버그 | Serial Wire / SWD |
| 시스템 클럭 | 84 MHz |
| APB1 Timer Clock | 84 MHz |
| APB2 Timer Clock | 84 MHz |

PA15, PB3, PB4 같은 핀을 PWM으로 쓰기 때문에 `SYS → Debug = Serial Wire`로 설정해야 함.

---

## 2. 서보모터

| 항목 | 설정 |
|---|---|
| 서보모터 | DS51150-270 |
| 제어 방식 | PWM |
| PWM 주기 | 5 ms |
| PWM 주파수 | 200 Hz |
| 중립 Pulse | 1500 us |
| 사용 범위 | 500 ~ 2500 us |
| 타이머 Prescaler | 83 |
| Counter Period | 4999 |
| 초기 Pulse | 1500 |

타이머 클럭이 84 MHz이므로:

```text
84 MHz / (83 + 1) = 1 MHz
```

즉 카운터 1칸이 1 us이고, `4999 + 1 = 5000 us = 5 ms`가 됨.

---

## 3. MCP3008 3개 / 아날로그 입력 24채널

MCP3008 3개가 박힌 PCB에서 아날로그 신호 24개 읽기  
MCP3008 1개당 8채널 총 24채널

| 기능 | STM32 Pin | MCP3008 |
|---|---|---|
| SPI1_SCK | PA5 | MCP3008 CLK |
| SPI1_MISO | PA6 | MCP3008 DOUT |
| SPI1_MOSI | PA7 | MCP3008 DIN |
| CS1 | PA4 | - |
| CS2 | PC0 | - |
| CS3 | PC1 | - |

### SPI1 설정

| 항목 | 설정 |
|---|---|
| Mode | Full-Duplex Master |
| Data Size | 8 Bits |
| First Bit | MSB First |
| CPOL | Low |
| CPHA | 1 Edge |
| NSS | Software |
| Prescaler | 32 |
| SPI 속도 | 약 2.625 Mbit/s |
| CRC | Disable |

CS 핀은 GPIO Output으로 직접 제어함.

```text
평소 CS = HIGH
읽을 MCP3008만 CS = LOW
통신 끝나면 다시 CS = HIGH
```

---

## 4. PWM 18개 + MCP3008 다리 배정

### 다리 1번

| 항목 | 타이머 / 채널 | 핀 |
|---|---|---|
| 1_1 | TIM1_CH1 | PA8 |
| 1_2 | TIM1_CH2 | PA9 |
| 1_3 | TIM1_CH3 | PA10 |

| MCP3008 | 채널 | 신호 |
|---|---|---|
| CS1 / PA4 | CH0 | 1_1 |
| CS1 / PA4 | CH1 | 1_2 |
| CS1 / PA4 | CH2 | 1_3 |
| CS1 / PA4 | CH3 | 1번 다리 압력센서 |

### 다리 2번

| 항목 | 타이머 / 채널 | 핀 |
|---|---|---|
| 2_1 | TIM2_CH1 | PA15 |
| 2_2 | TIM2_CH2 | PB3 |
| 2_3 | TIM2_CH3 | PB10 |

| MCP3008 | 채널 | 신호 |
|---|---|---|
| CS1 / PA4 | CH4 | 2_1 |
| CS1 / PA4 | CH5 | 2_2 |
| CS1 / PA4 | CH6 | 2_3 |
| CS1 / PA4 | CH7 | 2번 다리 압력센서 |

### 다리 3번

| 항목 | 타이머 / 채널 | 핀 |
|---|---|---|
| 3_1 | TIM3_CH1 | PB4 |
| 3_2 | TIM3_CH2 | PB5 |
| 3_3 | TIM3_CH3 | PB0 |

| MCP3008 | 채널 | 신호 |
|---|---|---|
| CS2 / PC0 | CH0 | 3_1 |
| CS2 / PC0 | CH1 | 3_2 |
| CS2 / PC0 | CH2 | 3_3 |
| CS2 / PC0 | CH3 | 3번 다리 압력센서 |

### 다리 4번

| 항목 | 타이머 / 채널 | 핀 |
|---|---|---|
| 4_1 | TIM4_CH1 | PB6 |
| 4_2 | TIM4_CH2 | PB7 |
| 4_3 | TIM4_CH3 | PB8 |

| MCP3008 | 채널 | 신호 |
|---|---|---|
| CS2 / PC0 | CH4 | 4_1 |
| CS2 / PC0 | CH5 | 4_2 |
| CS2 / PC0 | CH6 | 4_3 |
| CS2 / PC0 | CH7 | 4번 다리 압력센서 |

### 다리 5번

| 항목 | 타이머 / 채널 | 핀 |
|---|---|---|
| 5_1 | TIM5_CH1 | PA0 |
| 5_2 | TIM5_CH2 | PA1 |
| 5_3 | TIM8_CH3 | PC8 |

| MCP3008 | 채널 | 신호 |
|---|---|---|
| CS3 / PC1 | CH0 | 5_1 |
| CS3 / PC1 | CH1 | 5_2 |
| CS3 / PC1 | CH2 | 5_3 |
| CS3 / PC1 | CH3 | 5번 다리 압력센서 |

### 다리 6번

| 항목 | 타이머 / 채널 | 핀 |
|---|---|---|
| 6_1 | TIM1_CH4 | PA11 |
| 6_2 | TIM3_CH4 | PB1 |
| 6_3 | TIM4_CH4 | PB9 |

| MCP3008 | 채널 | 신호 |
|---|---|---|
| CS3 / PC1 | CH4 | 6_1 |
| CS3 / PC1 | CH5 | 6_2 |
| CS3 / PC1 | CH6 | 6_3 |
| CS3 / PC1 | CH7 | 6번 다리 압력센서 |

---

## 5. INA / INB / INC

1: 오른쪽 릴레이
2: 왼쪽 릴레이

| 기능 | STM32 | 설정 |
|---|---|---|
| INA1 | PC2  | GPIO_Output |
| INB1 | PC3  | GPIO_Output |
| INC1 | PC4  | GPIO_Output |
| INA2 | PC5  | GPIO_Output |
| INB2 | PA12 | GPIO_Output |
| INC2 | PB2  | GPIO_Output |

| 상태 | 릴레이 |
|---|---|
| LOW | 릴레이 OFF |
| HIGH | 릴레이 ON |

### GPIO 설정

| 항목 | 설정 |
|---|---|
| Output Level | Low |
| Mode | Output Push Pull |
| Pull-up/Pull-down | No Pull |
| Speed | Low |

---

## 6. Jetson Orin Nano Super ↔ STM32 통신

Jetson Orin Nano Super = SPI Master  
STM32F446RE = SPI Slave

| 기능 | Jetson 40핀 헤더 | STM32F446RE |
|---|---|---|
| SPI MOSI | Pin 19 | PB15 / SPI2_MOSI |
| SPI MISO | Pin 21 | PB14 / SPI2_MISO |
| SPI SCLK | Pin 23 | PB13 / SPI2_SCK |
| SPI CS0 | Pin 24 | PB12 / SPI2_NSS |
| DRDY | Pin 22 / GPIO Input | PC9 / GPIO_Output |
| GND | Pin 6 또는 Pin 9 | GND |

### SPI2 설정

| 항목 | 설정 |
|---|---|
| Mode | Full-Duplex Slave |
| Data Size | 8 Bits |
| First Bit | MSB First |
| CPOL | Low |
| CPHA | 1 Edge |
| NSS | Hardware Input |
| CRC | Disable |

### DRDY 역할

원래는 Jetson이 읽고 싶을 때만 STM32의 값을 읽을 수 있음  
그러면 긴급한 신호에 반응을 못 할 수 있음  
따라서 STM32가 보낼 데이터 준비 완료를 알려주는 신호가 필요함

### 동작 개념

```text
→ STM32가 상태 데이터 준비
→ DRDY HIGH
→ Jetson이 CS LOW
→ SPI 통신
→ 통신 완료 후 DRDY LOW
```

---

## 7. USART2 / GPS

GPS: NEO M8N GPS Module, TYE-GP001

| 기능 | STM32 Pin |
|---|---|
| USART2_TX | PA2 |
| USART2_RX | PA3 |

### 설정

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

---

## 8. USART3 / IMU

WT931 9축 IMU 모듈

| 기능 | STM32 Pin |
|---|---|
| USART3_TX | PC10 |
| USART3_RX | PC11 |

### 설정

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

---

## 9. UART5 / LoRa 통신 모듈

RYLR998 계열 UART LoRa 통신 모듈

| 기능 | STM32 Pin |
|---|---|
| UART5_TX | PC12 |
| UART5_RX | PD2 |

### 설정

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

---

## 10. USART6 / RadioMaster Pocket ELRS + ELRS Nano 수신기

조종기: [RadioMaster Pocket 조종기 ELRS](https://susungrc.com/product/%EB%9D%BC%EB%94%94%EC%98%A4%EB%A7%88%EC%8A%A4%ED%84%B0-pocket-%EC%A1%B0%EC%A2%85%EA%B8%B0-elrs/2385/)

수신기: [ELRS-2.4G-NANO ESP8285 수신기 / DarwinFPV 2.4G ELRS Nano](https://www.devicemart.co.kr/goods/view?no=15138136)
통신 프로토콜: ELRS 무선 링크 + 수신기 출력 CRSF UART

RadioMaster Pocket은 최대 16채널을 지원한다. 실제 조종 입력은 Mode 2 기준으로 왼쪽 짐벌에 Throttle/Yaw, 오른쪽 짐벌에 Pitch/Roll이 배치된다. 물리 입력은 SA·SD 2단 유지형 스위치, SB·SC 3단 스위치, SE 순간 버튼과 S1 가변 입력으로 구성된다. 자세한 외형과 입력 종류는 [RadioMaster Pocket 공식 매뉴얼](https://radiomasterrc.com/pages/user-manuals)을 따른다.

| 기능 | STM32 Pin |
|---|---|
| USART6_TX | PC6 |
| USART6_RX | PC7 |

실제로 필요한 건 RX인 PC7임.  
PC6 TX는 설정상 잡혀 있어도 배선하지 않아도 됨.

### 설정

| 항목 | 설정 |
|---|---|
| Mode | Asynchronous |
| Baud Rate | 420000 |
| Word Length | 8 Bits |
| Parity | None |
| Stop Bits | 1 |
| Data Direction | TX/RX |
| Hardware Flow Control | Disable |
| Over Sampling | 16 Samples |

### 프로젝트 CRSF 채널 배치

ELRS 수신기가 채널 용도를 고정하는 것은 아니다. EdgeTX의 현재 모델에 설정한 `Mixes`가 CRSF CH1~CH16의 용도를 결정하므로 이 프로젝트에서는 다음 배치를 표준으로 사용한다.

| CRSF 채널 | EdgeTX Source | Pocket 물리 입력 | STM32 사용값 |
|---:|---|---|---|
| CH1 | Ail | 오른쪽 짐벌 좌우 | Roll |
| CH2 | Ele | 오른쪽 짐벌 상하 | Pitch |
| CH3 | Thr | 왼쪽 짐벌 상하 | Throttle, 중립 0의 `-1000~1000` |
| CH4 | Rud | 왼쪽 짐벌 좌우 | Yaw |
| CH5 | SA | 2단 유지형 스위치 | SA, OFF/ON을 0/1로 변환 |
| CH6 | SB | 3단 스위치 | SB, 세 위치를 0/1/2로 변환 |
| CH7 | SC | 3단 스위치 | SC, 세 위치를 0/1/2로 변환 |
| CH8 | SD | 2단 유지형 스위치 | SD, 해제/눌림을 0/1로 변환 |
| CH9 | SE | 순간 버튼 | SE, 해제/누름을 0/1로 변환 |
| CH10 | S1 | 가변 입력 | 현재 미사용, 예비 채널 |
| CH11~CH16 | - | - | 미사용 |

CH1~CH4는 `AETR` 순서이다. Radio Setup의 `Default Channel Order`는 새 모델을 만들 때만 반영되며 이미 만든 모델의 채널은 바뀌지 않는다. 따라서 Pocket에서 사용할 전용 모델의 `Mixes`를 위 표대로 설정하고 채널 모니터에서 CH1~CH10을 직접 확인한다.

STM32의 `user_command`는 CH1~CH4를 위 이름으로 다시 배치하고, 네 축을 중립 0의 `-1000~1000`으로 변환한다. CH5~CH9는 제어 모델이 사용하는 스위치 상태로 변환한다. 조종 모드에서 SA OFF일 때는 Throttle로 x이동, Yaw로 회전을 명령하고, SA ON일 때는 Throttle로 x이동, Yaw로 y이동을 명령하며 전환 시점의 Heading을 유지한다. 조이스틱 방향 반전, 중립값과 최종 정규화 계수는 실제 조종기 채널 모니터 및 CRSF 수신 시험에서 확인한 뒤 설정 테이블에 채운다.

이 모드 변경은 기존 CH3·CH4·CH5 입력을 재해석하므로 수신기와 STM32 사이의 추가 배선은 필요하지 않다.

---

## 11. TIM6 / 5ms 제어 루프용 타이머

### 설정

| 항목 | 설정 |
|---|---|
| TIM6 Prescaler | 83 |
| TIM6 Period | 4999 |
| 주기 | 5 ms |
| 인터럽트 | TIM6_DAC_IRQn Enable |

### 용도

5ms마다 제어 루프를 돌릴 때 쓰면 됨.

```c
HAL_TIM_Base_Start_IT(&htim6);
```

---

위의 설정을 기반으로 HAL 기반 + 일부 레지스터 직접 접근 방식으로 코드 작성
