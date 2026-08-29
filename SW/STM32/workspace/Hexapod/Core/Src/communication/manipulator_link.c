#include "communication/manipulator_link.h"

#include <stddef.h>
#include <string.h>

#define MANIPULATOR_OFFSET_SYNC_0       0U   // 첫 번째 동기 바이트 위치를 정의한다.
#define MANIPULATOR_OFFSET_SYNC_1       1U   // 두 번째 동기 바이트 위치를 정의한다.
#define MANIPULATOR_OFFSET_VERSION      2U   // 프로토콜 버전 위치를 정의한다.
#define MANIPULATOR_OFFSET_SEQUENCE     3U   // 패킷 순번 위치를 정의한다.
#define MANIPULATOR_OFFSET_FLAGS        4U   // 안전 상태 비트 위치를 정의한다.
#define MANIPULATOR_OFFSET_SWITCHES     5U   // 조종기 스위치 위치를 정의한다.
#define MANIPULATOR_OFFSET_ROLL         6U   // Roll 시작 위치를 정의한다.
#define MANIPULATOR_OFFSET_PITCH        8U   // Pitch 시작 위치를 정의한다.
#define MANIPULATOR_OFFSET_THROTTLE    10U   // Throttle 시작 위치를 정의한다.
#define MANIPULATOR_OFFSET_YAW         12U   // Yaw 시작 위치를 정의한다.
#define MANIPULATOR_OFFSET_CRC         14U   // CRC 시작 위치를 정의한다.

/* 16비트 정수를 Little-endian으로 저장한다. */
static void ManipulatorLink_WriteI16(uint8_t *destination, int16_t value)
{
    const uint16_t raw = (uint16_t)value;  // 부호 있는 값을 동일한 비트열로 변환한다.

    destination[0] = (uint8_t)(raw & 0xFFU);  // 하위 바이트를 먼저 저장한다.
    destination[1] = (uint8_t)(raw >> 8U);    // 상위 바이트를 뒤에 저장한다.
}

/* 조종기 스위치 상태를 한 바이트로 묶는다. */
static uint8_t ManipulatorLink_PackSwitches(const RobotUserCommand_t *user)
{
    uint8_t switches = 0U;  // 모든 스위치를 해제 상태로 준비한다.

    switches |= (uint8_t)((user->sa != 0U) ? (1U << 0U) : 0U);  // SA 상태를 bit 0에 넣는다.
    switches |= (uint8_t)((user->sb & 0x03U) << 1U);             // SB 3단 상태를 bit 1~2에 넣는다.
    switches |= (uint8_t)((user->sc & 0x03U) << 3U);             // SC 3단 상태를 bit 3~4에 넣는다.
    switches |= (uint8_t)((user->sd != 0U) ? (1U << 5U) : 0U);  // SD 상태를 bit 5에 넣는다.
    switches |= (uint8_t)((user->se != 0U) ? (1U << 6U) : 0U);  // SE 상태를 bit 6에 넣는다.
    switches |= (uint8_t)((user->s1 != 0U) ? (1U << 7U) : 0U);  // S1 상태를 bit 7에 넣는다.

    return switches;
}

/* 유선 매니퓰레이터 송신 상태를 초기화한다. */
void ManipulatorLink_Init(ManipulatorLink_Handle_t *handle,
                          UART_HandleTypeDef *uart)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 송신 상태를 제거한다.
    handle->uart = uart;                 // 사용할 UART Handle을 저장한다.
}

/* CRC-16/CCITT-FALSE 값을 계산한다. */
uint16_t ManipulatorLink_Crc16CcittFalse(const uint8_t *data,
                                         uint32_t length)
{
    uint16_t crc = 0xFFFFU;  // CCITT-FALSE 초기값을 적용한다.
    uint32_t index;          // 입력 바이트 위치를 저장한다.

    if (data == NULL)
    {
        return 0U;
    }

    for (index = 0U; index < length; ++index)
    {
        uint8_t bit;  // 현재 계산할 비트 위치를 저장한다.

        crc ^= (uint16_t)data[index] << 8U;  // 다음 바이트를 CRC 상위에 반영한다.

        for (bit = 0U; bit < 8U; ++bit)
        {
            crc = ((crc & 0x8000U) != 0U)
                ? (uint16_t)((crc << 1U) ^ 0x1021U)
                : (uint16_t)(crc << 1U);  // CCITT 다항식을 한 비트씩 적용한다.
        }
    }

    return crc;
}

/* 최신 조종값을 고정 16바이트 패킷으로 만든다. */
bool ManipulatorLink_BuildPacket(uint8_t packet[MANIPULATOR_PACKET_SIZE],
                                 uint8_t sequence,
                                 const RobotUserCommand_t *user,
                                 RobotControlMode_t active_mode)
{
    uint8_t flags = 0U;  // 연결과 ARM 허가 상태를 준비한다.
    uint16_t crc;        // 완성할 패킷 CRC를 저장한다.

    if ((packet == NULL) || (user == NULL))
    {
        return false;
    }

    if (user->connected)
    {
        flags |= MANIPULATOR_FLAG_CONNECTED;  // 정상 CRSF 연결을 표시한다.
    }
    if (user->motion_armed)
    {
        flags |= MANIPULATOR_FLAG_MOTION_ARMED;  // 중립 재허가 완료를 표시한다.
    }
    if ((active_mode == ROBOT_MODE_ARM) && user->connected &&
        user->motion_armed && (user->sc == 1U) && (user->sd == 0U))
    {
        flags |= MANIPULATOR_FLAG_ARM_MODE;  // 유효한 SC ARM 조작 허가를 표시한다.
    }

    packet[MANIPULATOR_OFFSET_SYNC_0] = MANIPULATOR_PACKET_SYNC_0;    // 첫 번째 동기 바이트를 넣는다.
    packet[MANIPULATOR_OFFSET_SYNC_1] = MANIPULATOR_PACKET_SYNC_1;    // 두 번째 동기 바이트를 넣는다.
    packet[MANIPULATOR_OFFSET_VERSION] = MANIPULATOR_PACKET_VERSION;  // 프로토콜 버전을 넣는다.
    packet[MANIPULATOR_OFFSET_SEQUENCE] = sequence;                   // 패킷 순번을 넣는다.
    packet[MANIPULATOR_OFFSET_FLAGS] = flags;                         // 안전 상태 비트를 넣는다.

    packet[MANIPULATOR_OFFSET_SWITCHES] = ManipulatorLink_PackSwitches(user);  // 여섯 스위치를 묶는다.

    ManipulatorLink_WriteI16(&packet[MANIPULATOR_OFFSET_ROLL], user->roll);          // Roll 값을 넣는다.
    ManipulatorLink_WriteI16(&packet[MANIPULATOR_OFFSET_PITCH], user->pitch);        // Pitch 값을 넣는다.
    ManipulatorLink_WriteI16(&packet[MANIPULATOR_OFFSET_THROTTLE], user->throttle);  // Throttle 값을 넣는다.
    ManipulatorLink_WriteI16(&packet[MANIPULATOR_OFFSET_YAW], user->yaw);            // Yaw 값을 넣는다.

    crc = ManipulatorLink_Crc16CcittFalse(packet, MANIPULATOR_PACKET_CRC_SIZE);  // 헤더부터 Yaw까지 검사한다.
    packet[MANIPULATOR_OFFSET_CRC] = (uint8_t)(crc & 0xFFU);                     // CRC 하위 바이트를 넣는다.
    packet[MANIPULATOR_OFFSET_CRC + 1U] = (uint8_t)(crc >> 8U);                  // CRC 상위 바이트를 넣는다.
    return true;
}

/* 주기가 된 최신 조종 패킷을 UART 인터럽트로 송신한다. */
bool ManipulatorLink_Process(ManipulatorLink_Handle_t *handle,
                             const RobotUserCommand_t *user,
                             RobotControlMode_t active_mode,
                             uint32_t now_ms)
{
    HAL_StatusTypeDef status;  // 비동기 송신 시작 결과를 저장한다.

    if ((handle == NULL) || (handle->uart == NULL) || (user == NULL))
    {
        return false;
    }
    if ((now_ms - handle->last_tx_ms) < MANIPULATOR_TX_PERIOD_MS)
    {
        return false;  // 아직 5 ms 송신 주기가 되지 않았음을 알린다.
    }
    if (handle->tx_busy)
    {
        handle->tx_busy_skip_count++;  // 이전 패킷 완료 전이면 최신 주기만 건너뛴다.
        return false;
    }
    if (!ManipulatorLink_BuildPacket(handle->tx_packet,
                                     handle->sequence,
                                     user,
                                     active_mode))
    {
        return false;
    }

    handle->tx_busy = true;  // 완료 콜백보다 먼저 송신 상태를 표시한다.
    status = HAL_UART_Transmit_IT(handle->uart,
                                  handle->tx_packet,
                                  MANIPULATOR_PACKET_SIZE);  // 16바이트를 블로킹 없이 전송한다.
    if (status != HAL_OK)
    {
        handle->tx_busy = false;     // 실패한 송신 상태를 해제한다.
        handle->tx_error_count++;    // 송신 시작 오류를 기록한다.
        return false;
    }

    handle->last_tx_ms = now_ms;  // 정상 시작 시각을 다음 주기 기준으로 저장한다.
    handle->sequence++;           // 다음 패킷 순번으로 이동한다.
    return true;
}

/* 매니퓰레이터 UART 송신 완료를 기록한다. */
void ManipulatorLink_TxCpltCallback(ManipulatorLink_Handle_t *handle,
                                    UART_HandleTypeDef *uart)
{
    if ((handle == NULL) || (handle->uart == NULL) ||
        (uart != handle->uart))
    {
        return;
    }

    handle->tx_busy = false;  // 다음 패킷 송신을 허가한다.
    handle->tx_count++;       // 정상 완료 횟수를 기록한다.
}

/* 매니퓰레이터 UART 오류 후 다음 송신을 허가한다. */
void ManipulatorLink_ErrorCallback(ManipulatorLink_Handle_t *handle,
                                   UART_HandleTypeDef *uart)
{
    if ((handle == NULL) || (handle->uart == NULL) ||
        (uart != handle->uart))
    {
        return;
    }

    handle->tx_busy = false;   // 오류가 난 송신 상태를 해제한다.
    handle->tx_error_count++;  // UART 오류를 기록한다.
}
