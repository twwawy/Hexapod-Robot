> 통합 v4: 현재 구현·남은 연결 작업은 [통합 문서](../../docs/ADAPTIVE_INTEGRATION_V4.md), wire는 [SPI v3](../../docs/ADAPTIVE_SPI_V3.md)를 참고하세요. 아래 기존 설계 설명과 구분합니다.


# Jetson 소프트웨어

Jetson Orin Nano Super에서 담당할 상위 인지·자율주행 소프트웨어의 범위를 정리한다. 현재 이 폴더에는 실행 코드가 없지만 STM32 센서 패킷 규격과 Slave DMA 전송 경로는 구현되어 있다.

## 예정 역할

- Livox Mid-360과 RealSense D435 데이터 수집
- 지형·장애물 인식
- 경로 계획과 자율주행 명령 생성
- STM32 상태 수신과 상위 상태 관리

Jetson은 5 ms 관절 제어와 서보 PWM을 직접 수행하지 않는다. 실시간 보행 궤적, IK, Safety와 액추에이터 출력은 STM32가 담당한다.

## STM32 연결

Jetson은 SPI Master, STM32는 SPI2 Slave로 사용한다. MOSI, MISO, SCLK, CS와 DRDY 물리 연결은 [STM32F446RE 설정 정리본](../STM32/STM32F446RE%20설정%20정리본.md#6-jetson-orin-nano-super--stm32-통신)을 따른다.

현재 STM32에 확정·구현된 항목은 다음과 같다.

- SPI Mode 0 기반 32바이트 전이중 프레임 v2
- STM32 센서 패킷의 18개 관절각, 6개 발 접촉과 IMU 자세
- CRC-16/CCITT-FALSE와 16비트 Sequence
- STM32 SPI2 Slave RX/TX DMA
- STM32가 DMA를 Arm한 뒤 `DRDY` High, 완료 또는 오류에서 Low
- Jetson의 32바이트 `0x00` Dummy 읽기 허용

Jetson이 보낼 `COMMAND` 패킷 종류와 Raw 24바이트 보관 API도 STM32에 있다. 다만 Payload 내부 명령 배치, 명령 Timeout, 자율주행·수동 조종 전환과 Safety 우선순위 연결은 아직 정하지 않았다.

정확한 필드 배치, 인코딩, CRC와 거래 순서는 [STM32–Jetson SPI 32바이트 패킷 프로토콜](../STM32/STM32-Jetson%20SPI%2032바이트%20패킷%20프로토콜.md)을 따른다. Jetson 구현은 먼저 1 MHz에서 `DRDY`를 기다린 뒤 정확히 32바이트를 교환하고 CRC와 Sequence를 검증하는 수신기부터 작성한다.
