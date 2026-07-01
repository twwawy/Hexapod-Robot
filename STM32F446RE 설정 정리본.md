STM32F446RE Hexapod 최종 설정 정리본


1. 보드 / MCU
보드: NUCLEO-F446RE
개발 환경: STM32CubeMX / STM32CubeIDE
디버그: Serial Wire / SWD
시스템 클럭: 84 MHz
APB1 Timer Clock: 84 MHz
APB2 Timer Clock: 84 MHz

PA15, PB3, PB4 같은 핀을 PWM으로 쓰기 때문에 SYS → Debug = Serial Wire로 설정해야 함.


2. 서보모터
서보모터: DS51150-270
제어 방식: PWM
PWM 주기: 5 ms
PWM 주파수: 200 Hz
중립 Pulse: 1500 us
사용 범위: 500 ~ 2500 us
타이머 Prescaler: 83
Counter Period: 4999
초기 Pulse: 1500

타이머 클럭이 84 MHz이므로: 84 MHz / (83 + 1) = 1 MHz
즉 카운터 1칸이 1 us이고, 4999 + 1 = 5000 us = 5 ms가 됨.


3. MCP3008 3개 / 아날로그 입력 24채널
MCP3008 3개가 박힌 PCB에서 아날로그 신호 24개 읽기
MCP3008 1개당 8채널 총 24채널

SPI1_SCK   PA5   MCP3008 CLK
SPI1_MISO   PA6   MCP3008 DOUT
SPI1_MOSI   PA7   MCP3008 DIN
CS1   PA4
CS2   PC0
CS3   PC1

(SPI1 설정)
Mode: Full-Duplex Master
Data Size: 8 Bits
First Bit: MSB First
CPOL: Low
CPHA   1: Edge
NSS: Software
Prescaler: 32
SPI 속도: 약 2.625 Mbit/s
CRC: Disable

CS 핀은 GPIO Output으로 직접 제어함.
평소 CS = HIGH
읽을 MCP3008만 CS = LOW
통신 끝나면 다시 CS = HIGH


4. PWM 18개 + MCP3008 다리 배정
다리 1번
1_1   TIM1_CH1   PA8
1_2   TIM1_CH2   PA9
1_3   TIM1_CH3   PA10
CS1 / PA4   CH0   1_1
CS1 / PA4   CH1   1_2
CS1 / PA4   CH2   1_3
CS1 / PA4   CH3   1번 다리 압력센서

다리 2번
2_1   TIM2_CH1   PA15
2_2   TIM2_CH2   PB3
2_3   TIM2_CH3   PB10
CS1 / PA4   CH4   2_1
CS1 / PA4   CH5   2_2
CS1 / PA4   CH6   2_3
CS1 / PA4   CH7   2번 다리 압력센서

다리 3번
3_1   TIM3_CH1   PB4
3_2   TIM3_CH2   PB5
3_3   TIM3_CH3   PB0
CS2 / PC0   CH0   3_1
CS2 / PC0   CH1   3_2
CS2 / PC0   CH2   3_3
CS2 / PC0   CH3   3번 다리 압력센서

다리 4번
4_1   TIM4_CH1   PB6
4_2   TIM4_CH2   PB7
4_3   TIM4_CH3   PB8
CS2 / PC0   CH4   4_1
CS2 / PC0   CH5   4_2
CS2 / PC0   CH6   4_3
CS2 / PC0   CH7   4번 다리 압력센서

다리 5번
5_1   TIM5_CH1   PA0
5_2   TIM5_CH2   PA1
5_3   TIM8_CH3   PC8
CS3 / PC1   CH0   5_1
CS3 / PC1   CH1   5_2
CS3 / PC1   CH2   5_3
CS3 / PC1   CH3   5번 다리 압력센서

다리 6번
6_1   TIM1_CH4   PA11
6_2   TIM3_CH4   PB1
6_3   TIM4_CH4   PB9
CS3 / PC1   CH4   6_1
CS3 / PC1   CH5   6_2
CS3 / PC1   CH6   6_3
CS3 / PC1   CH7   6번 다리 압력센서


5. INA / INB / INC
기능   STM32  설정
INA   PC2      GPIO_Output
INB   PC3      GPIO_Output
INC   PC4      GPIO_Output

LOW   릴레이 OFF
HIGH   릴레이 ON

(GPIO 설정)
Output Level: Low
Mode   Output: Push Pull
Pull-up/Pull-down: No Pull
Speed: Low


6. Jetson Orin Nano Super ↔ STM32 통신
Jetson Orin Nano Super = SPI Master
STM32F446RE = SPI Slave

기능           Jetson 40핀 헤더           STM32F446RE
SPI MOSI   Pin 19                   PB15 / SPI2_MOSI
SPI MISO   Pin 21                   PB14 / SPI2_MISO
SPI SCLK   Pin 23                   PB13 / SPI2_SCK
SPI CS0   Pin 24                   PB12 / SPI2_NSS
DRDY           Pin 22 / GPIO Input   PC9 / GPIO_Output
GND           Pin 6 또는 Pin 9           GND

(SPI2 설정)
Mode   : Full-Duplex Slave
Data Size: 8 Bits
First Bit: MSB First
CPOL: Low
CPHA   1: Edge
NSS: Hardware Input
CRC: Disable

DRDY 역할
원래는 Jetson이 읽고 싶을 때만 stm32의 값을 읽을 수 있음
그러면 긴급한 신호에 반응을 못 할 수 있음
따라서 STM32가 보낼 데이터 준비 완료를 알려주는 신호가 필요함

동작 개념
→ STM32가 상태 데이터 준비
→ DRDY HIGH
→ Jetson이 CS LOW
→ SPI 통신
→ 통신 완료 후 DRDY LOW


7. USART2 / GPS
GPS(NEO M8N GPS Module, TYE-GP001)

USART2_TX   PA2
USART2_RX      PA3

(설정)
Mode   : Asynchronous
Baud Rate: 9600
Word Length: 8 Bits
Parity: None
Stop Bits: 1
Data Direction: TX/RX
Hardware Flow Control: Disable
Over Sampling: 16 Samples


8. USART3 / IMU
BMI160 + RM3100 9축 IMU 모듈

USART3_TX     PC10
USART3_RX   PC11

(설정)
Mode: Asynchronous
Baud Rate: 9600
Word Length: 8 Bits
Parity: None
Stop Bits: 1
Data Direction: TX/RX
Hardware Flow Control: Disable
Over Sampling: 16 Samples


9. UART5 / LoRa 통신 모듈
RYLR998 계열 UART LoRa 통신 모듈

UART5_TX   PC12
UART5_RX   PD2

(설정)
Mode: Asynchronous
Baud Rate: 115200
Word Length: 8 Bits
Parity: None
Stop Bits: 1
Data Direction: TX/RX
Hardware Flow Control: Disable
Over Sampling: 16 Samples


10. USART6 / RadioMaster Pocket ELRS + ELRS Nano 수신기
조종기: RadioMaster Pocket 조종기 ELRS
수신기: ELRS-2.4G-NANO ESP8285 수신기 / DarwinFPV 2.4G ELRS Nano
통신 프로토콜: ELRS 무선 링크 + 수신기 출력 CRSF UART

USART6_TX   PC6
USART6_RX   PC7

실제로 필요한 건 RX인 PC7임.
PC6 TX는 설정상 잡혀 있어도 배선하지 않아도 됨.

(설정)
Mode: Asynchronous
Baud Rate: 420000
Word Length: 8 Bits
Parity: None
Stop Bits: 1
Data Direction: Receive Only
Hardware Flow Control: Disable
Over Sampling: 16 Samples


11. TIM6 / 5ms 제어 루프용 타이머
(설정)
TIM6 Prescaler: 83
TIM6 Period: 4999
주기: 5 ms
인터럽트: TIM6_DAC_IRQn Enable

(용도)
5ms마다 제어 루프를 돌릴 때 쓰면 됨.
HAL_TIM_Base_Start_IT(&htim6);


위의 내용을 기반으로 HAL 기반 + 일부 레지스터 직접 접근 방식으로 코드 짤거임