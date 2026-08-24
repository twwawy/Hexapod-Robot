#ifndef CRSF_PROTOCOL_H
#define CRSF_PROTOCOL_H

#include "common/robot_config.h"
#include "user_command/crsf_receiver.h"

#include <stdbool.h>
#include <stdint.h>

#define CRSF_FRAME_MAX_SIZE       64U
#define CRSF_FRAME_TYPE_RC_CHANNELS_PACKED 0x16U

typedef struct
{
    uint8_t address;                         // 현재 프레임 주소를 저장한다.
    uint8_t length;                          // 현재 프레임 길이를 저장한다.
    uint8_t index;                           // 현재 수신 위치를 저장한다.
    uint8_t frame[CRSF_FRAME_MAX_SIZE];      // Type부터 CRC까지 저장한다.
    uint16_t channel[ROBOT_CRSF_CHANNEL_COUNT];  // 최근 16개 채널을 저장한다.
    uint32_t frame_counter;                  // 정상 RC 프레임 수를 저장한다.
    uint32_t crc_error_count;                // CRC 오류 수를 저장한다.
    uint32_t length_error_count;             // 길이 오류 수를 저장한다.
} CRSF_Protocol_t;

void CRSF_Protocol_Init(CRSF_Protocol_t *protocol);  // 프레임 해석 상태를 초기화한다.

bool CRSF_Protocol_ProcessByte(CRSF_Protocol_t *protocol,
                               uint8_t byte);        // 한 바이트를 처리하고 RC 갱신 여부를 반환한다.

uint32_t CRSF_Protocol_ProcessReceiver(CRSF_Protocol_t *protocol,
                                       CRSF_Receiver_t *receiver);  // 링 버퍼의 모든 바이트를 처리한다.

uint8_t CRSF_Protocol_Crc8(const uint8_t *data,
                           uint8_t length);           // CRSF CRC8 값을 계산한다.

#endif
