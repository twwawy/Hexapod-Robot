
# Jetson 소프트웨어

Jetson Orin Nano Super에서 담당할 상위 인지·자율주행 소프트웨어의 범위를 정리한다. 현재 이 폴더에는 실행 코드가 없으며 STM32와 교환할 명령과 패킷 규격을 확정한 뒤 구현한다.

## 예정 역할

- Livox Mid-360과 RealSense D435 데이터 수집
- 지형·장애물 인식
- 경로 계획과 자율주행 명령 생성
- STM32 상태 수신과 상위 상태 관리

Jetson은 5 ms 관절 제어와 서보 PWM을 직접 수행하지 않는다. 실시간 보행 궤적, IK, Safety와 액추에이터 출력은 STM32가 담당한다.

## STM32 연결

Jetson은 SPI Master, STM32는 SPI2 Slave로 사용한다. MOSI, MISO, SCLK, CS와 DRDY 물리 연결은 [STM32F446RE 설정 정리본](../STM32/STM32F446RE%20설정%20정리본.md#6-jetson-orin-nano-super--stm32-통신)을 따른다.

다음 항목은 아직 정하지 않는다.

- Jetson이 보낼 명령 종류
- STM32가 보낼 상태 종류
- SPI 프레임 길이와 필드 순서
- CRC 방식
- 통신 주기와 Timeout
- DRDY의 최종 동작
- 자율주행과 수동 조종의 전환 조건

STM32에는 상위 코드가 참조할 최소 인터페이스 자리만 유지한다. 실제 프로토콜은 위 항목이 확정된 뒤 Jetson과 STM32 문서를 동시에 갱신하고 구현한다.
