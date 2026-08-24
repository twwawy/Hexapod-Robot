#ifndef ROBOT_TELEMETRY_H
#define ROBOT_TELEMETRY_H

#include "common/robot_types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ROBOT_TELEMETRY_MAX_TEXT 180U

typedef enum
{
    ROBOT_TELEMETRY_STATUS = 0,  // 운용 상태 패킷을 나타낸다.
    ROBOT_TELEMETRY_JOINT,       // 관절각 패킷을 나타낸다.
    ROBOT_TELEMETRY_GPS          // GPS 패킷을 나타낸다.
} RobotTelemetry_Type_t;

typedef struct
{
    uint32_t sequence;            // 전체 패킷 순번을 저장한다.
    uint32_t last_status_ms;      // STATUS 전송 시각을 저장한다.
    uint32_t last_joint_ms;       // JOINT 전송 시각을 저장한다.
    uint32_t last_gps_ms;         // GPS 전송 시각을 저장한다.
    uint32_t last_packet_ms;      // 마지막 LoRa 패킷 시각을 저장한다.
    RobotTelemetry_Type_t next;   // 다음 우선 검사 패킷을 저장한다.
} RobotTelemetry_Handle_t;

void RobotTelemetry_Init(RobotTelemetry_Handle_t *handle);  // Telemetry 주기 상태를 초기화한다.

bool RobotTelemetry_BuildNext(RobotTelemetry_Handle_t *handle,
                              uint32_t now_ms,
                              RobotControlMode_t mode,
                              const RobotSafetyOutput_t *safety,
                              const RobotSensorSnapshot_t *sensor,
                              uint8_t relay_mask,
                              char *text,
                              size_t text_size);  // 전송 시각이 된 다음 패킷을 만든다.

#endif
