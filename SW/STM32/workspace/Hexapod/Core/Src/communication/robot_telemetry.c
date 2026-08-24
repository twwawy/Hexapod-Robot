#include "communication/robot_telemetry.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TELEMETRY_STATUS_PERIOD_MS 1000U
#define TELEMETRY_JOINT_PERIOD_MS  2000U
#define TELEMETRY_GPS_PERIOD_MS    2000U
#define TELEMETRY_PACKET_GAP_MS     500U

/* Telemetry 주기와 순번을 초기화한다. */
void RobotTelemetry_Init(RobotTelemetry_Handle_t *handle)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 전송 상태를 제거한다.
}

/* STATUS 문자열 패킷을 만든다. */
static int RobotTelemetry_BuildStatus(uint32_t sequence,
                                      uint32_t now_ms,
                                      RobotControlMode_t mode,
                                      const RobotSafetyOutput_t *safety,
                                      const RobotSensorSnapshot_t *sensor,
                                      uint8_t relay_mask,
                                      char *text,
                                      size_t text_size)
{
    uint8_t contact_mask = 0U;  // 발 접촉 상태를 비트로 저장한다.
    uint32_t leg;               // 접촉 상태를 읽을 다리 번호를 저장한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (sensor->foot_contact[leg])
        {
            contact_mask |= (uint8_t)(1U << leg);  // 접촉한 다리를 비트로 표시한다.
        }
    }

    return snprintf(text,
                    text_size,
                    "S,%lu,%lu,%u,%u,%u,%ld,%ld,%ld,%u,%u",
                    (unsigned long)sequence,
                    (unsigned long)now_ms,
                    (unsigned int)mode,
                    safety->rollover_fault ? 1U : 0U,
                    safety->controller_fault ? 1U : 0U,
                    (long)lroundf(sensor->imu.attitude_rad.roll * ROBOT_RAD_TO_DEG_F * 100.0f),
                    (long)lroundf(sensor->imu.attitude_rad.pitch * ROBOT_RAD_TO_DEG_F * 100.0f),
                    (long)lroundf(sensor->imu.attitude_rad.yaw * ROBOT_RAD_TO_DEG_F * 100.0f),
                    (unsigned int)contact_mask,
                    (unsigned int)relay_mask);  // 관제탑 STATUS 필드를 고정 순서로 만든다.
}

/* JOINT 문자열 패킷을 만든다. */
static int RobotTelemetry_BuildJoint(uint32_t sequence,
                                     uint32_t now_ms,
                                     const RobotSensorSnapshot_t *sensor,
                                     char *text,
                                     size_t text_size)
{
    size_t used;      // 현재 문자열 길이를 저장한다.
    uint32_t joint;   // 출력할 관절 번호를 저장한다.
    int written;      // 현재 snprintf 결과를 저장한다.

    written = snprintf(text, text_size, "J,%lu,%lu",
                       (unsigned long)sequence,
                       (unsigned long)now_ms);  // JOINT 공통 머리말을 만든다.

    if ((written < 0) || ((size_t)written >= text_size))
    {
        return -1;
    }

    used = (size_t)written;  // 다음 필드 위치를 저장한다.

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        written = snprintf(&text[used],
                           text_size - used,
                           ",%ld",
                           (long)lroundf(sensor->joint_angle_rad[joint] *
                                         ROBOT_RAD_TO_DEG_F * 100.0f));  // 관절각을 0.01도로 추가한다.

        if ((written < 0) || ((size_t)written >= (text_size - used)))
        {
            return -1;
        }

        used += (size_t)written;  // 다음 관절 필드 위치를 갱신한다.
    }

    return (int)used;
}

/* GPS 문자열 패킷을 만든다. */
static int RobotTelemetry_BuildGps(uint32_t sequence,
                                   uint32_t now_ms,
                                   const RobotSensorSnapshot_t *sensor,
                                   char *text,
                                   size_t text_size)
{
    return snprintf(text,
                    text_size,
                    "G,%lu,%lu,%lld,%lld,%ld,%u",
                    (unsigned long)sequence,
                    (unsigned long)now_ms,
                    (long long)llround(sensor->gps.latitude_deg * 1.0e7),
                    (long long)llround(sensor->gps.longitude_deg * 1.0e7),
                    (long)lroundf(sensor->gps.altitude_m * 1000.0f),
                    sensor->gps.valid ? 1U : 0U);  // GPS 좌표를 정수 단위로 만든다.
}

/* 전송 주기가 된 STATUS·JOINT·GPS 중 하나를 만든다. */
bool RobotTelemetry_BuildNext(RobotTelemetry_Handle_t *handle,
                              uint32_t now_ms,
                              RobotControlMode_t mode,
                              const RobotSafetyOutput_t *safety,
                              const RobotSensorSnapshot_t *sensor,
                              uint8_t relay_mask,
                              char *text,
                              size_t text_size)
{
    RobotTelemetry_Type_t type;  // 이번에 만들 패킷 종류를 저장한다.
    int length;                   // 생성한 문자열 길이를 저장한다.
    uint32_t attempt;             // 패킷 종류 검사 횟수를 저장한다.
    bool selected = false;        // 전송할 패킷 선택 여부를 저장한다.

    if ((handle == NULL) || (safety == NULL) || (sensor == NULL) ||
        (text == NULL) || (text_size == 0U))
    {
        return false;
    }

    if ((now_ms - handle->last_packet_ms) < TELEMETRY_PACKET_GAP_MS)
    {
        return false;  // 느린 LoRa의 연속 패킷 Burst를 막는다.
    }

    type = handle->next;  // 직전 패킷 다음 종류부터 검사한다.
    for (attempt = 0U; attempt < 3U; ++attempt)
    {
        if ((type == ROBOT_TELEMETRY_STATUS) &&
            ((now_ms - handle->last_status_ms) >= TELEMETRY_STATUS_PERIOD_MS))
        {
            handle->last_status_ms = now_ms;  // STATUS 전송 시각을 갱신한다.
            selected = true;                  // STATUS 선택을 표시한다.
            break;
        }
        if ((type == ROBOT_TELEMETRY_JOINT) &&
            ((now_ms - handle->last_joint_ms) >= TELEMETRY_JOINT_PERIOD_MS))
        {
            handle->last_joint_ms = now_ms;  // JOINT 전송 시각을 갱신한다.
            selected = true;                 // JOINT 선택을 표시한다.
            break;
        }
        if ((type == ROBOT_TELEMETRY_GPS) &&
            ((now_ms - handle->last_gps_ms) >= TELEMETRY_GPS_PERIOD_MS))
        {
            handle->last_gps_ms = now_ms;  // GPS 전송 시각을 갱신한다.
            selected = true;               // GPS 선택을 표시한다.
            break;
        }

        type = (RobotTelemetry_Type_t)(((uint32_t)type + 1U) % 3U);  // 다음 종류를 검사한다.
    }

    if (!selected)
    {
        return false;  // 아직 주기가 된 패킷이 없음을 알린다.
    }

    handle->last_packet_ms = now_ms;                              // LoRa 패킷 간격 기준을 갱신한다.
    handle->next = (RobotTelemetry_Type_t)(((uint32_t)type + 1U) % 3U);  // 다음 우선 종류를 저장한다.
    handle->sequence++;                                           // 새 패킷 순번을 할당한다.

    if (type == ROBOT_TELEMETRY_STATUS)
    {
        length = RobotTelemetry_BuildStatus(handle->sequence, now_ms, mode,
                                             safety, sensor, relay_mask,
                                             text, text_size);  // STATUS를 만든다.
    }
    else if (type == ROBOT_TELEMETRY_JOINT)
    {
        length = RobotTelemetry_BuildJoint(handle->sequence, now_ms,
                                            sensor, text, text_size);  // JOINT를 만든다.
    }
    else
    {
        length = RobotTelemetry_BuildGps(handle->sequence, now_ms,
                                          sensor, text, text_size);  // GPS를 만든다.
    }

    return (length > 0) && ((size_t)length < text_size) &&
           ((size_t)length <= ROBOT_TELEMETRY_MAX_TEXT);  // 패킷 길이 제한을 확인한다.
}
