#ifndef MEASUREMENT_DEBUG_H
#define MEASUREMENT_DEBUG_H

#include "common/robot_calibration.h"
#include "common/robot_types.h"

#include <stdbool.h>
#include <stdint.h>

#define MEASUREMENT_DEBUG_STAGE_COUNT 8U

typedef enum
{
    MEASUREMENT_DEBUG_INIT_OK = 0,          // 초기화 오류가 없음을 나타낸다.
    MEASUREMENT_DEBUG_INIT_INVALID_HANDLE,  // CubeMX Handle 오류를 나타낸다.
    MEASUREMENT_DEBUG_INIT_ADC,             // MCP3008 초기화 오류를 나타낸다.
    MEASUREMENT_DEBUG_INIT_GPS,             // GPS 수신 시작 오류를 나타낸다.
    MEASUREMENT_DEBUG_INIT_IMU              // WT931 수신 시작 오류를 나타낸다.
} MeasurementDebug_InitError_t;

typedef struct
{
    uint32_t current_stage;                                             // 현재 실측 단계 번호를 저장한다.
    bool stage_completed[MEASUREMENT_DEBUG_STAGE_COUNT];                // 단계별 사용자 완료를 저장한다.
    bool initialization_complete;                                      // 현재 단계 초기화 완료를 저장한다.
    MeasurementDebug_InitError_t initialization_error;                 // 현재 단계 초기화 오류를 저장한다.
    uint32_t last_sample_ms;                                           // 마지막 실측 시각을 저장한다.

    RobotSensorSnapshot_t latest_sensor;                                // 최근 GPS·IMU·관절·압력값을 저장한다.
    uint16_t adc_raw[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT];       // 최근 ADC 24채널 값을 저장한다.
    uint16_t adc_min[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT];       // ADC 채널별 최소값을 저장한다.
    uint16_t adc_max[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT];       // ADC 채널별 최대값을 저장한다.
    uint32_t sensor_sample_count;                                       // 정상 센서 측정 횟수를 저장한다.
    uint32_t sensor_error_count;                                        // 센서 읽기 실패 횟수를 저장한다.
    uint32_t gps_update_count;                                          // GPS 위치 갱신 횟수를 저장한다.
    uint32_t gps_rx_overflow_count;                                     // GPS 수신 버퍼 초과 횟수를 저장한다.
    uint32_t imu_frame_count;                                           // WT931 정상 프레임 수를 저장한다.
    uint32_t imu_checksum_error_count;                                  // WT931 체크섬 오류 수를 저장한다.
    uint32_t imu_rx_overflow_count;                                     // WT931 수신 버퍼 초과 횟수를 저장한다.
    uint32_t adc_update_count;                                          // ADC 24채널 갱신 횟수를 저장한다.
    uint32_t adc_driver_error_count;                                    // MCP3008 드라이버 오류 수를 저장한다.
    uint8_t adc_last_error_device;                                      // 마지막 실패 MCP3008 번호를 저장한다.
    uint8_t adc_last_error_channel;                                     // 마지막 실패 채널 번호를 저장한다.

    uint32_t imu_sample_count;                                         // WT931 수평 표본 수를 저장한다.
    double imu_euler_sum_rad[3];                                       // WT931 자세 누적값을 저장한다.

    bool adc_mapping_recorded[MCP3008_LEG_COUNT][MCP3008_LEG_INPUT_COUNT]; // ADC 입력별 매핑 확인을 저장한다.

    uint16_t joint_minimum_raw[ROBOT_JOINT_COUNT];                     // 최소 자세 관절 ADC를 저장한다.
    uint16_t joint_zero_raw[ROBOT_JOINT_COUNT];                        // 영점 자세 관절 ADC를 저장한다.
    uint16_t joint_maximum_raw[ROBOT_JOINT_COUNT];                     // 최대 자세 관절 ADC를 저장한다.
    bool joint_minimum_captured;                                       // 최소 자세 기록 여부를 저장한다.
    bool joint_zero_captured;                                          // 영점 자세 기록 여부를 저장한다.
    bool joint_maximum_captured;                                       // 최대 자세 기록 여부를 저장한다.

    uint32_t pressure_unloaded_sum[ROBOT_PRESSURE_COUNT];              // 압력센서 무부하 합계를 저장한다.
    uint32_t pressure_loaded_sum[ROBOT_PRESSURE_COUNT];                // 압력센서 접촉 합계를 저장한다.
    uint16_t pressure_unloaded_count;                                  // 무부하 표본 수를 저장한다.
    uint16_t pressure_loaded_count;                                    // 접촉 표본 수를 저장한다.

    uint16_t crsf_minimum[USER_COMMAND_USED_CHANNELS];                 // CRSF 채널별 최소 raw를 저장한다.
    uint16_t crsf_center[USER_COMMAND_USED_CHANNELS];                  // CRSF 채널별 중립 raw를 저장한다.
    uint16_t crsf_maximum[USER_COMMAND_USED_CHANNELS];                 // CRSF 채널별 최대 raw를 저장한다.
    bool crsf_center_captured;                                         // CRSF 중립 기록 여부를 저장한다.

    uint8_t active_servo_joint;                                        // 현재 실측 중인 서보 번호를 저장한다.
    uint16_t active_servo_pulse_us;                                    // 현재 실측 서보 Pulse를 저장한다.
    uint8_t relay_state_mask;                                          // 현재 릴레이 출력 비트를 저장한다.

    RobotCalibration_t calibration;                                    // 중앙 테이블에 옮길 최종 실측값을 저장한다.
} MeasurementDebug_t;

extern volatile MeasurementDebug_t g_measurement_debug;  // STM32CubeIDE에서 확인할 전역 실측값을 공개한다.

void MeasurementDebug_Reset(void);  // 모든 실측 디버그 값을 초기화한다.

#endif
