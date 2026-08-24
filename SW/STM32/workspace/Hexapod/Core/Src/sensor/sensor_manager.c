#include "sensor/sensor_manager.h"

#include <stddef.h>
#include <string.h>

/* 실제 센서 드라이버와 변환 모듈을 연결한다. */
void SensorManager_Init(SensorManager_Handle_t *handle,
                        GPS_Handle_t *gps,
                        IMU_Handle_t *imu,
                        MCP3008_Handle_t *mcp3008)
{
    if (handle == NULL)
    {
        return;
    }

    memset(handle, 0, sizeof(*handle));       // 이전 센서 상태를 제거한다.
    handle->gps = gps;                        // GPS 드라이버를 연결한다.
    handle->imu = imu;                        // IMU 드라이버를 연결한다.
    handle->mcp3008 = mcp3008;                // ADC 드라이버를 연결한다.
    JointFeedback_Init(&handle->joints);      // 관절 보정 테이블을 준비한다.
    FootPressure_Init(&handle->pressure);     // 압력센서 임계값을 준비한다.
}

/* GPS·WT931·MCP3008의 최신 실제 측정값을 스냅샷으로 만든다. */
bool SensorManager_Update(SensorManager_Handle_t *handle)
{
    GPS_Data_t gps_data;   // 최근 GPS 값을 저장한다.
    IMU_Data_t imu_data;   // 최근 IMU 값을 저장한다.
    uint32_t leg;          // ADC를 배치할 다리 번호를 저장한다.
    uint32_t joint;        // ADC를 배치할 관절 번호를 저장한다.
    bool adc_ok;           // ADC 전체 읽기 결과를 저장한다.

    if ((handle == NULL) || (handle->gps == NULL) ||
        (handle->imu == NULL) || (handle->mcp3008 == NULL))
    {
        return false;
    }

    (void)GPS_Process(handle->gps);                 // GPS 수신 버퍼를 해석한다.
    (void)IMU_Process(handle->imu);                 // IMU 수신 버퍼를 해석한다.
    adc_ok = (MCP3008_ReadAll(handle->mcp3008, &handle->adc) == HAL_OK);  // 24개 ADC 채널을 읽는다.

    if (GPS_GetLatest(handle->gps, &gps_data))
    {
        handle->snapshot.gps.latitude_deg = gps_data.latitude_deg;    // 위도를 갱신한다.
        handle->snapshot.gps.longitude_deg = gps_data.longitude_deg;  // 경도를 갱신한다.
        handle->snapshot.gps.altitude_m = gps_data.height_m;          // 고도를 갱신한다.
        handle->snapshot.gps.timestamp_ms = gps_data.mcu_time_ms;     // GPS 시각을 갱신한다.
        handle->snapshot.gps.valid = gps_data.position_valid;         // GPS 유효성을 갱신한다.
    }

    if (IMU_GetLatest(handle->imu, &imu_data))
    {
        handle->snapshot.imu.acceleration_mps2.x = imu_data.acceleration_mps2[0];       // X 가속도를 갱신한다.
        handle->snapshot.imu.acceleration_mps2.y = imu_data.acceleration_mps2[1];       // Y 가속도를 갱신한다.
        handle->snapshot.imu.acceleration_mps2.z = imu_data.acceleration_mps2[2];       // Z 가속도를 갱신한다.
        handle->snapshot.imu.angular_velocity_radps.x = imu_data.angular_velocity_radps[0];  // Roll 각속도를 갱신한다.
        handle->snapshot.imu.angular_velocity_radps.y = imu_data.angular_velocity_radps[1];  // Pitch 각속도를 갱신한다.
        handle->snapshot.imu.angular_velocity_radps.z = imu_data.angular_velocity_radps[2];  // Yaw 각속도를 갱신한다.
        handle->snapshot.imu.attitude_rad.roll = imu_data.euler_angle_rad[0];            // Roll을 갱신한다.
        handle->snapshot.imu.attitude_rad.pitch = imu_data.euler_angle_rad[1];           // Pitch를 갱신한다.
        handle->snapshot.imu.attitude_rad.yaw = imu_data.euler_angle_rad[2];             // Yaw를 갱신한다.
        handle->snapshot.imu.timestamp_ms = imu_data.mcu_time_ms;                        // IMU 시각을 갱신한다.
        handle->snapshot.imu.valid = IMU_HasNavigationData(&imu_data);                   // IMU 유효성을 갱신한다.
    }

    if (adc_ok)
    {
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            for (joint = 0U; joint < ROBOT_JOINTS_PER_LEG; ++joint)
            {
                const uint32_t index = leg * ROBOT_JOINTS_PER_LEG + joint;  // 관절 배열 위치를 계산한다.
                handle->snapshot.joint_raw[index] = handle->adc.leg_raw[leg][joint];  // 관절 raw 값을 배치한다.
            }

            handle->snapshot.pressure_raw[leg] =
                handle->adc.leg_raw[leg][MCP3008_LEG_PRESSURE];             // 압력 raw 값을 배치한다.
        }

        (void)JointFeedback_Convert(&handle->joints,
                                    handle->snapshot.joint_raw,
                                    handle->snapshot.joint_angle_rad);       // 관절각을 변환한다.
        FootPressure_Update(&handle->pressure,
                            handle->snapshot.pressure_raw,
                            handle->snapshot.foot_contact);                  // 접촉 상태를 갱신한다.
        handle->snapshot.timestamp_ms = handle->adc.mcu_time_ms;             // 스냅샷 시각을 갱신한다.
    }

    return adc_ok;
}

/* 가장 최근 센서 스냅샷을 복사한다. */
bool SensorManager_GetSnapshot(const SensorManager_Handle_t *handle,
                               RobotSensorSnapshot_t *snapshot)
{
    if ((handle == NULL) || (snapshot == NULL))
    {
        return false;
    }

    *snapshot = handle->snapshot;   // 제어기에 일관된 센서값을 전달한다.
    return true;
}
