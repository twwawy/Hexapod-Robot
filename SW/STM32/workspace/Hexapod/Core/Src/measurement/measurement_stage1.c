#include "measurement/measurement_stage1.h"

#include "common/robot_config.h"
#include "low_control/relay.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "sensor/imu.h"

#include <stddef.h>

static IMU_Handle_t measurement_imu;                  // WT931 수신 상태를 저장한다.
static MeasurementRunner_t measurement_runner;        // 현재 실측 단계를 저장한다.
static uint32_t measurement_last_frame_count;         // 마지막 처리 프레임 번호를 저장한다.
static bool measurement_initialized;                  // 1단계 실행 가능 여부를 저장한다.

/* 확인 완료 시 전용 프로그램에서 설정한 WT931 값을 중앙 표 형식으로 기록한다. */
static void MeasurementStage1_UpdateConfirmation(void)
{
    uint32_t axis;       // 확인할 자세 축 번호를 저장한다.
    bool all_confirmed;  // 전체 자세 축 확인 여부를 저장한다.

    all_confirmed = true;  // 미확인 축을 찾기 전 완료로 가정한다.
    for (axis = 0U; axis < 3U; ++axis)
    {
        if (!g_measurement_debug.imu_axis_confirmed[axis])
        {
            all_confirmed = false;  // 한 축이라도 미확인이면 완료를 막는다.
        }
    }

    g_measurement_debug.imu_check_complete = all_confirmed;          // 최종 확인 상태를 갱신한다.
    g_measurement_debug.calibration.imu_calibrated = all_confirmed;  // 중앙 표 복사용 완료 상태를 갱신한다.
    if (!all_confirmed)
    {
        return;
    }

    for (axis = 0U; axis < 3U; ++axis)
    {
        g_measurement_debug.calibration.imu.acceleration_sign[axis] = 1;      // 장치에서 설정한 축을 그대로 사용한다.
        g_measurement_debug.calibration.imu.angular_velocity_sign[axis] = 1;  // 장치에서 설정한 축을 그대로 사용한다.
        g_measurement_debug.calibration.imu.euler_angle_sign[axis] = 1;       // 확인한 자세 방향을 그대로 사용한다.
        g_measurement_debug.calibration.imu.euler_offset_rad[axis] = 0.0f;    // 장치에서 설정한 영점을 그대로 사용한다.
    }
}

/* USART3 WT931 각도 최종 확인을 준비한다. */
bool MeasurementStage1_Init(UART_HandleTypeDef *imu_uart)
{
    uint32_t axis;  // 초기화할 자세 축 번호를 저장한다.

    MeasurementRunner_Init(&measurement_runner);                   // 전역 디버그값을 초기화한다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 1단계 IMU 확인으로 이동한다.
    Relay_Init();                                                  // 실측 중 모든 서보 전원을 차단한다.
    measurement_initialized = false;                              // 초기화 완료 전 실행을 막는다.

    if (imu_uart == NULL)
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 잘못된 USART3 Handle을 기록한다.
        return false;
    }

    IMU_Init(&measurement_imu, imu_uart);  // USART3 WT931을 연결한다.
    for (axis = 0U; axis < 3U; ++axis)
    {
        g_measurement_debug.imu_axis_confirmed[axis] = true;  // 확인 완료한 자세 축을 기록한다.
    }
    MeasurementStage1_UpdateConfirmation();  // 중앙 표와 같은 IMU 완료값을 디버거에 기록한다.

    if (IMU_Start(&measurement_imu) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_IMU;  // WT931 수신 실패를 기록한다.
        return false;
    }

    measurement_last_frame_count = 0U;                            // 첫 각도 프레임을 받을 준비를 한다.
    measurement_initialized = true;                               // 주기 실행을 허용한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK;  // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;            // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 새 WT931 각도 프레임과 사용자의 축 확인 상태를 기록한다. */
void MeasurementStage1_Process(void)
{
    IMU_Data_t data;  // 최근 WT931 각도 프레임을 저장한다.
    uint32_t axis;    // 갱신할 자세 축 번호를 저장한다.

    if (!measurement_initialized)
    {
        return;
    }

    (void)IMU_Process(&measurement_imu);     // 인터럽트 수신 바이트를 각도값으로 해석한다.
    MeasurementStage1_UpdateConfirmation();  // 사용자가 확인한 자세 축을 반영한다.

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
