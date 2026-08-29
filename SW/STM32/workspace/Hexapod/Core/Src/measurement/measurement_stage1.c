#include "measurement/measurement_stage1.h"

#include "common/robot_config.h"
#include "low_control/relay.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "sensor/imu.h"

#include <stddef.h>

static IMU_Handle_t measurement_imu;                  // WT931 수신 상태를 저장한다.
static IMU_LevelCalibration_t measurement_level;      // Roll·Pitch 자동 영점 상태를 저장한다.
static MeasurementRunner_t measurement_runner;        // 현재 실측 단계를 저장한다.
static uint32_t measurement_last_frame_count;         // 마지막 처리 프레임 번호를 저장한다.
static bool measurement_initialized;                  // 1단계 실행 가능 여부를 저장한다.

/* 자동 영점 측정 상태와 최종 보정값을 디버거에 기록한다. */
static void MeasurementStage1_UpdateCalibrationDebug(void)
{
    IMU_Calibration_t applied;  // 현재 적용한 IMU 보정값을 저장한다.
    uint32_t axis;              // 갱신할 자세 축 번호를 저장한다.

    IMU_GetCalibration(&measurement_imu, &applied);  // 자동 보정이 반영된 값을 읽는다.
    g_measurement_debug.imu_calibration_state =
        (uint8_t)measurement_level.state;  // 현재 자동 보정 단계를 표시한다.
    g_measurement_debug.imu_calibration_sample_count =
        measurement_level.sample_count;   // 현재 정지 표본 수를 표시한다.
    for (axis = 0U; axis < 3U; ++axis)
    {
        g_measurement_debug.imu_euler_offset_rad[axis] =
            applied.euler_offset_rad[axis];  // 현재 축 Offset을 rad로 표시한다.
        g_measurement_debug.imu_euler_offset_deg[axis] =
            applied.euler_offset_rad[axis] * ROBOT_RAD_TO_DEG_F;  // 현재 축 Offset을 deg로 표시한다.
    }
    if (measurement_level.state != IMU_LEVEL_CALIBRATION_COMPLETE)
    {
        return;
    }

    for (axis = 0U; axis < 3U; ++axis)
    {
        g_measurement_debug.imu_axis_confirmed[axis] = true;  // 기존 축 방향과 자동 영점을 확인 완료로 표시한다.
    }
    g_measurement_debug.calibration.imu = applied;       // 중앙 표에 복사할 최종 IMU 값을 저장한다.
    g_measurement_debug.calibration.imu_calibrated = true;  // 최종 IMU 실측 완료를 표시한다.
    g_measurement_debug.imu_check_complete = true;       // 자동 영점 적용 완료를 표시한다.
}

/* USART3 WT931 각도 최종 확인을 준비한다. */
bool MeasurementStage1_Init(UART_HandleTypeDef *imu_uart)
{
    MeasurementRunner_Init(&measurement_runner);                   // 전역 디버그값을 초기화한다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 1단계 IMU 확인으로 이동한다.
    g_measurement_debug.calibration = g_robot_calibration;         // 기존 중앙 보정표를 측정 기준으로 복사한다.
    Relay_Init();                                                  // 실측 중 모든 서보 전원을 차단한다.
    measurement_initialized = false;                              // 초기화 완료 전 실행을 막는다.

    if (imu_uart == NULL)
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 잘못된 USART3 Handle을 기록한다.
        return false;
    }

    IMU_Init(&measurement_imu, imu_uart);  // USART3 WT931을 연결한다.
    IMU_SetCalibration(&measurement_imu,
                       &g_robot_calibration.imu);  // 기존 축 방향과 Offset에서 재측정을 시작한다.

    if (IMU_Start(&measurement_imu) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_IMU;  // WT931 수신 실패를 기록한다.
        return false;
    }

    IMU_LevelCalibration_Init(&measurement_level,
                              HAL_GetTick());                     // 수평 정지 자세 측정을 준비한다.
    measurement_last_frame_count = 0U;                            // 첫 각도 프레임을 받을 준비를 한다.
    measurement_initialized = true;                               // 주기 실행을 허용한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK;  // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;            // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 새 WT931 각도 프레임을 평균내 자동 영점을 기록한다. */
void MeasurementStage1_Process(void)
{
    IMU_Data_t data;  // 최근 WT931 각도 프레임을 저장한다.
    uint32_t axis;    // 갱신할 자세 축 번호를 저장한다.

    if (!measurement_initialized)
    {
        return;
    }

    (void)IMU_Process(&measurement_imu);  // 인터럽트 수신 바이트를 각도값으로 해석한다.
    (void)IMU_LevelCalibration_Update(&measurement_level,
                                      &measurement_imu,
                                      HAL_GetTick());  // 정지 표본으로 Roll·Pitch 영점을 갱신한다.
    MeasurementStage1_UpdateCalibrationDebug();  // 진행 상태와 최종 Offset을 표시한다.

    if (!IMU_GetLatest(&measurement_imu, &data) ||
        (data.frame_counter == measurement_last_frame_count))
    {
        return;
    }

    measurement_last_frame_count = data.frame_counter;                             // 새 프레임 처리를 기록한다.
    g_measurement_debug.last_sample_ms = data.mcu_time_ms;                         // 최근 각도 수신 시각을 갱신한다.
    g_measurement_debug.imu_frame_count = data.frame_counter;                      // 정상 각도 프레임 수를 갱신한다.
    g_measurement_debug.imu_checksum_error_count = data.checksum_error_count;      // 체크섬 오류 수를 갱신한다.
    g_measurement_debug.imu_rx_overflow_count = measurement_imu.rx_overflow_count; // 버퍼 초과 수를 갱신한다.

    for (axis = 0U; axis < 3U; ++axis)
    {
        g_measurement_debug.imu_euler_current_rad[axis] = data.euler_angle_rad[axis];  // 최근 rad 자세값을 갱신한다.
        g_measurement_debug.imu_euler_current_deg[axis] =
            data.euler_angle_rad[axis] / ROBOT_DEG_TO_RAD_F;                          // 확인하기 쉬운 deg로 변환한다.
    }
    g_measurement_debug.latest_sensor.imu.attitude_rad.roll = data.euler_angle_rad[0];   // 최근 Roll을 갱신한다.
    g_measurement_debug.latest_sensor.imu.attitude_rad.pitch = data.euler_angle_rad[1];  // 최근 Pitch를 갱신한다.
    g_measurement_debug.latest_sensor.imu.attitude_rad.yaw = data.euler_angle_rad[2];    // 최근 Yaw를 갱신한다.
    g_measurement_debug.latest_sensor.imu.timestamp_ms = data.mcu_time_ms;                // 최근 자세 시각을 갱신한다.
    g_measurement_debug.latest_sensor.imu.valid = IMU_HasNavigationData(&data);           // 각도 프레임 유효성을 갱신한다.

    if (g_measurement_debug.imu_check_complete &&
        !measurement_runner.completed[MEASUREMENT_STAGE_IMU])
    {
        (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // IMU 측정 완료 후 다음 단계로 이동한다.
    }
}

/* WT931의 1바이트 수신 완료를 드라이버에 전달한다. */
void MeasurementStage1_UartRxCallback(UART_HandleTypeDef *uart)
{
    IMU_RxCpltCallback(&measurement_imu, uart);  // USART3이면 다음 WT931 수신을 시작한다.
}

/* WT931의 UART 오류를 드라이버에서 복구한다. */
void MeasurementStage1_UartErrorCallback(UART_HandleTypeDef *uart)
{
    IMU_ErrorCallback(&measurement_imu, uart);  // USART3 WT931 수신을 복구한다.
}
