#include "user_command/crsf_protocol.h"

#include <stddef.h>
#include <string.h>

#define CRSF_RC_PAYLOAD_SIZE 22U
#define CRSF_CRC_POLYNOMIAL  0xD5U

/* CRSF 주소 바이트 후보를 확인한다. */
static bool CRSF_Protocol_IsAddress(uint8_t byte)
{
    return (byte == 0xC8U) || (byte == 0xECU) ||
           (byte == 0xEAU) || (byte == 0xEEU);  // FC와 송수신기 주소를 허용한다.
}

/* 22 byte RC Payload를 16개 11 bit 채널로 해제한다. */
static void CRSF_Protocol_UnpackChannels(const uint8_t *payload,
                                         uint16_t channel[ROBOT_CRSF_CHANNEL_COUNT])
{
    uint32_t bit_buffer = 0U;  // 아직 사용하지 않은 비트를 저장한다.
    uint32_t bit_count = 0U;   // 버퍼의 유효 비트 수를 저장한다.
    uint32_t byte_index = 0U;  // Payload 읽기 위치를 저장한다.
    uint32_t channel_index;    // 채널 출력 위치를 저장한다.

    for (channel_index = 0U; channel_index < ROBOT_CRSF_CHANNEL_COUNT; ++channel_index)
    {
        while (bit_count < 11U)
        {
            bit_buffer |= (uint32_t)payload[byte_index++] << bit_count;  // 다음 byte를 이어 붙인다.
            bit_count += 8U;                                            // 유효 비트 수를 갱신한다.
        }

        channel[channel_index] = (uint16_t)(bit_buffer & 0x07FFU);  // 현재 11 bit 채널을 꺼낸다.
        bit_buffer >>= 11U;                                        // 사용한 비트를 제거한다.
        bit_count -= 11U;                                          // 남은 비트 수를 갱신한다.
    }
}

/* CRSF 프레임 해석 상태를 초기화한다. */
void CRSF_Protocol_Init(CRSF_Protocol_t *protocol)
{
    if (protocol == NULL)
    {
        return;
    }

    memset(protocol, 0, sizeof(*protocol));  // 이전 프레임 상태를 제거한다.
}

/* DVB-S2 다항식을 사용하는 CRSF CRC8을 계산한다. */
uint8_t CRSF_Protocol_Crc8(const uint8_t *data,
                           uint8_t length)
{
    uint8_t crc = 0U;     // 누적 CRC 값을 저장한다.
    uint8_t data_index;   // 입력 byte 위치를 저장한다.

    if (data == NULL)
    {
        return 0U;
    }

    for (data_index = 0U; data_index < length; ++data_index)
    {
        uint8_t bit;   // 현재 처리할 비트를 저장한다.

        crc ^= data[data_index];   // 다음 byte를 CRC에 반영한다.

        for (bit = 0U; bit < 8U; ++bit)
        {
            crc = ((crc & 0x80U) != 0U)
                ? (uint8_t)((crc << 1U) ^ CRSF_CRC_POLYNOMIAL)
                : (uint8_t)(crc << 1U);  // 한 비트씩 다항식을 적용한다.
        }
    }

    return crc;
}

/* 한 수신 바이트를 CRSF 상태기에 전달한다. */
bool CRSF_Protocol_ProcessByte(CRSF_Protocol_t *protocol,
                               uint8_t byte)
{
    if (protocol == NULL)
    {
        return false;
    }

    if (protocol->address == 0U)
    {
        if (CRSF_Protocol_IsAddress(byte))
        {
            protocol->address = byte;  // 새 프레임 주소를 저장한다.
        }
        return false;
    }

    if (protocol->length == 0U)
    {
        if ((byte < 2U) || (byte > (CRSF_FRAME_MAX_SIZE - 2U)))
        {
            protocol->length_error_count++;  // 잘못된 길이를 기록한다.
            protocol->address = CRSF_Protocol_IsAddress(byte) ? byte : 0U;  // 동기 후보를 보존한다.
            return false;
        }

        protocol->length = byte;  // Type+Payload+CRC 길이를 저장한다.
        protocol->index = 0U;     // 프레임 저장 위치를 초기화한다.
        return false;
    }

    protocol->frame[protocol->index++] = byte;  // Type 이후 데이터를 저장한다.

    if (protocol->index < protocol->length)
    {
        return false;
    }

    {
        const uint8_t crc = CRSF_Protocol_Crc8(protocol->frame,
                                               (uint8_t)(protocol->length - 1U));  // Type과 Payload의 CRC를 계산한다.
        const bool crc_ok = (crc == protocol->frame[protocol->length - 1U]);       // 수신 CRC와 비교한다.
        bool rc_updated = false;                                                   // RC 채널 갱신 여부를 저장한다.

        if (!crc_ok)
        {
            protocol->crc_error_count++;  // CRC 오류를 기록한다.
        }
        else if ((protocol->frame[0] == CRSF_FRAME_TYPE_RC_CHANNELS_PACKED) &&
                 (protocol->length == (CRSF_RC_PAYLOAD_SIZE + 2U)))
        {
            CRSF_Protocol_UnpackChannels(&protocol->frame[1], protocol->channel);  // 16개 RC 채널을 해제한다.
            protocol->frame_counter++;                                            // 정상 RC 프레임을 기록한다.
            rc_updated = true;                                                     // 채널 갱신을 알린다.
        }

        protocol->address = 0U;  // 다음 프레임을 위해 주소를 초기화한다.
        protocol->length = 0U;   // 다음 프레임을 위해 길이를 초기화한다.
        protocol->index = 0U;    // 다음 프레임을 위해 위치를 초기화한다.
        return rc_updated;
    }
}

/* 링 버퍼에 저장된 모든 CRSF 바이트를 처리한다. */
uint32_t CRSF_Protocol_ProcessReceiver(CRSF_Protocol_t *protocol,
                                       CRSF_Receiver_t *receiver)
{
    uint8_t byte;            // 처리할 수신 바이트를 저장한다.
    uint32_t update_count = 0U;  // 새 RC 프레임 수를 저장한다.

    if ((protocol == NULL) || (receiver == NULL))
    {
        return 0U;
    }

    while (CRSF_Receiver_PopByte(receiver, &byte))
    {
        if (CRSF_Protocol_ProcessByte(protocol, byte))
        {
            update_count++;  // 정상 RC 프레임을 기록한다.
        }
    }

    return update_count;
}
