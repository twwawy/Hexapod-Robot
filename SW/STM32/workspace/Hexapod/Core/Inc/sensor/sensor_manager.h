#ifndef SENSOR_MANAGER_H
#define SENSOR_MANAGER_H

#include "common/robot_types.h"
#include "sensor/foot_pressure.h"
#include "sensor/gps.h"
#include "sensor/imu.h"
#include "sensor/joint_feedback.h"
#include "sensor/mcp3008.h"

#include <stdbool.h>

typedef struct
{
    GPS_Handle_t *gps;                    // GPS 드라이버를 참조한다.
    IMU_Handle_t *imu;                    // IMU 드라이버를 참조한다.
    MCP3008_Handle_t *mcp3008;            // ADC 드라이버를 참조한다.
    JointFeedback_Handle_t joints;        // 관절센서 변환 상태를 저장한다.
    FootPressure_Handle_t pressure;       // 압력센서 변환 상태를 저장한다.
    MCP3008_Data_t adc;                   // 최근 ADC 값을 저장한다.
    RobotSensorSnapshot_t snapshot;       // 최근 센서 스냅샷을 저장한다.
    uint8_t contact_latched_mask;         // 제어 전까지 유지할 접촉 비트를 저장한다.
} SensorManager_Handle_t;

void SensorManager_Init(SensorManager_Handle_t *handle,
                        GPS_Handle_t *gps,
                        IMU_Handle_t *imu,
                        MCP3008_Handle_t *mcp3008);  // 실제 센서 드라이버를 연결한다.

bool SensorManager_Update(
    SensorManager_Handle_t *handle,
    const float pwm_angle_rad[ROBOT_JOINT_COUNT],
    bool pwm_valid);  // 최신 센서값과 PWM 예측으로 스냅샷을 갱신한다.

bool SensorManager_UpdatePressure(SensorManager_Handle_t *handle);  // 압력 6채널만 갱신한다.

uint8_t SensorManager_TakeContactLatch(SensorManager_Handle_t *handle);  // 접촉 비트를 한 번 꺼낸다.

bool SensorManager_GetSnapshot(const SensorManager_Handle_t *handle,
                               RobotSensorSnapshot_t *snapshot);  // 제어용 스냅샷을 반환한다.

#endif
