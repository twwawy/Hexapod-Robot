#include "measurement/measurement_stage0.h"

#include "low_control/relay.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "measurement/sensor_raw_measurement.h"

#include <stddef.h>

#define MEASUREMENT_STAGE0_SAMPLE_PERIOD_MS 10U

static GPS_Handle_t measurement_gps;                    // GPS 수신 상태를 저장한다.
static IMU_Handle_t measurement_imu;                    // WT931 수신 상태를 저장한다.
static MCP3008_Handle_t measurement_adc;                // MCP3008 통신 상태를 저장한다.
static SensorManager_Handle_t measurement_sensors;      // 통합 센서 상태를 저장한다.
static SensorRawMeasurement_t measurement_sensor_raw;   // 0단계 실측값을 저장한다.
static MeasurementRunner_t measurement_runner;          // 현재 실측 단계를 저장한다.
static uint32_t measurement_last_sample_ms;              // 최근 샘플 실행 시각을 저장한다.
static bool measurement_initialized;                    // 0단계 실행 가능 여부를 저장한다.

/* GPS·WT931·MCP3008를 연결하고 인터럽트 수신을 시작한다. */
bool MeasurementStage0_Init(UART_HandleTypeDef *gps_uart,
                            UART_HandleTypeDef *imu_uart,
                            SPI_HandleTypeDef *adc_spi)
{
    MeasurementRunner_Init(&measurement_runner);  // 전역 디버그값과 0단계를 초기화한다.
    Relay_Init();                                  // 실측 중 모든 서보 전원을 차단한다.
    measurement_initialized = false;               // 초기화 완료 전 실행을 막는다.

    if ((gps_uart == NULL) || (imu_uart == NULL) || (adc_spi == NULL))
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 잘못된 CubeMX Handle을 기록한다.
        return false;
    }

    GPS_Init(&measurement_gps, gps_uart);  // USART2 GPS를 연결한다.
    IMU_Init(&measurement_imu, imu_uart);  // USART3 WT931을 연결한다.
    if (MCP3008_Init(&measurement_adc, adc_spi) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_ADC;  // ADC 초기화 실패를 기록한다.
        return false;
    }

    SensorManager_Init(&measurement_sensors,
                       &measurement_gps,
                       &measurement_imu,
                       &measurement_adc);                     // 세 센서를 하나의 스냅샷으로 연결한다.
    SensorRawMeasurement_Init(&measurement_sensor_raw,
                              &measurement_sensors);           // Raw 범위 기록을 준비한다.

    if (GPS_Start(&measurement_gps) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_GPS;  // GPS 수신 실패를 기록한다.
        return false;
    }
    if (IMU_Start(&measurement_imu) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_IMU;  // IMU 수신 실패를 기록한다.
        return false;
    }

    measurement_last_sample_ms = HAL_GetTick();                      // 첫 실행 기준 시각을 저장한다.
    measurement_initialized = true;                                  // 주기 실행을 허용한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK;  // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;               // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 10 ms마다 GPS·WT931·MCP3008 값을 전역 디버그 구조체에 기록한다. */
void MeasurementStage0_Process(void)
{
    const uint32_t now_ms = HAL_GetTick();  // 현재 실측 시각을 저장한다.

    if (!measurement_initialized ||
        ((now_ms - measurement_last_sample_ms) < MEASUREMENT_STAGE0_SAMPLE_PERIOD_MS))
    {
        return;
    }

    measurement_last_sample_ms = now_ms;                      // 이번 샘플 시각을 저장한다.
    g_measurement_debug.last_sample_ms = now_ms;               // 디버거 샘플 시각을 갱신한다.
    (void)SensorRawMeasurement_Sample(&measurement_sensor_raw); // 전체 센서값을 한 번 기록한다.
}

/* GPS와 WT931의 1바이트 수신 완료를 해당 드라이버에 전달한다. */
void MeasurementStage0_UartRxCallback(UART_HandleTypeDef *uart)
{
    GPS_RxCpltCallback(&measurement_gps, uart);  // USART2이면 다음 GPS 수신을 시작한다.
    IMU_RxCpltCallback(&measurement_imu, uart);  // USART3이면 다음 WT931 수신을 시작한다.
}

/* GPS와 WT931의 UART 오류를 해당 드라이버에서 복구한다. */
void MeasurementStage0_UartErrorCallback(UART_HandleTypeDef *uart)
{
    GPS_ErrorCallback(&measurement_gps, uart);  // USART2 GPS 수신을 복구한다.
    IMU_ErrorCallback(&measurement_imu, uart);  // USART3 WT931 수신을 복구한다.
}
