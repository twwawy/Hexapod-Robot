#ifndef COMMUNICATION_MANIPULATOR_LINK_H
#define COMMUNICATION_MANIPULATOR_LINK_H

#ifdef __cplusplus
extern "C" {
#endif

#include "common/robot_types.h"
#include "stm32f4xx_hal.h"

#include <stdbool.h>
#include <stdint.h>

#define MANIPULATOR_PACKET_SIZE             16U         // 전체 패킷 크기를 정의한다.
#define MANIPULATOR_PACKET_CRC_SIZE         14U         // CRC 입력 구간 크기를 정의한다.
#define MANIPULATOR_TX_PERIOD_MS             5U         // 200 Hz 송신 주기를 정의한다.
#define MANIPULATOR_PACKET_SYNC_0           0xA5U       // 첫 번째 동기 바이트를 정의한다.
#define MANIPULATOR_PACKET_SYNC_1           0x5AU       // 두 번째 동기 바이트를 정의한다.
#define MANIPULATOR_PACKET_VERSION          0x01U       // 현재 프로토콜 버전을 정의한다.
#define MANIPULATOR_FLAG_CONNECTED          (1U << 0U)  // 조종기 연결 비트를 정의한다.
#define MANIPULATOR_FLAG_MOTION_ARMED       (1U << 1U)  // 조종 입력 허가 비트를 정의한다.
#define MANIPULATOR_FLAG_ARM_MODE           (1U << 2U)  // 매니퓰레이터 허가 비트를 정의한다.

typedef struct
{
    UART_HandleTypeDef *uart;                            // 매니퓰레이터 UART를 연결한다.
    uint8_t tx_packet[MANIPULATOR_PACKET_SIZE];          // 인터럽트 송신 중인 패킷을 보존한다.
    uint8_t sequence;                                    // 다음 송신 순번을 저장한다.
    uint32_t last_tx_ms;                                 // 마지막 송신 시작 시각을 저장한다.
    volatile uint32_t tx_count;                          // 정상 송신 완료 횟수를 저장한다.
    volatile uint32_t tx_error_count;                    // UART 송신 오류 횟수를 저장한다.
    volatile uint32_t tx_busy_skip_count;                // 이전 송신 대기 횟수를 저장한다.
    volatile bool tx_busy;                               // 인터럽트 송신 진행 여부를 저장한다.
} ManipulatorLink_Handle_t;

void ManipulatorLink_Init(ManipulatorLink_Handle_t *handle,
                          UART_HandleTypeDef *uart);  // 유선 매니퓰레이터 송신을 준비한다.

uint16_t ManipulatorLink_Crc16CcittFalse(const uint8_t *data,
                                         uint32_t length);  // 패킷 CRC16을 계산한다.

bool ManipulatorLink_BuildPacket(uint8_t packet[MANIPULATOR_PACKET_SIZE],
                                 uint8_t sequence,
                                 const RobotUserCommand_t *user,
                                 RobotControlMode_t active_mode);  // 최신 조종값으로 16바이트 패킷을 만든다.

bool ManipulatorLink_Process(ManipulatorLink_Handle_t *handle,
                             const RobotUserCommand_t *user,
                             RobotControlMode_t active_mode,
                             uint32_t now_ms);  // 5 ms 주기로 최신 패킷을 비동기 송신한다.

void ManipulatorLink_TxCpltCallback(ManipulatorLink_Handle_t *handle,
                                    UART_HandleTypeDef *uart);  // UART 송신 완료를 기록한다.

void ManipulatorLink_ErrorCallback(ManipulatorLink_Handle_t *handle,
                                   UART_HandleTypeDef *uart);  // UART 송신 오류를 복구한다.

#ifdef __cplusplus
}
#endif

#endif
