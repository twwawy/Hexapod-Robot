#include "measurement/measurement_stage0.h"

#include "low_control/relay.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "measurement/sensor_raw_measurement.h"
#include "user_command/crsf_protocol.h"
#include "user_command/crsf_receiver.h"

#include <stddef.h>

#define MEASUREMENT_STAGE0_SAMPLE_PERIOD_MS 10U
#define MEASUREMENT_STAGE0_CRSF_TIMEOUT_MS  500U

static GPS_Handle_t measurement_gps;                    // GPS 수신 상태를 저장한다.
static IMU_Handle_t measurement_imu;                    // WT931 수신 상태를 저장한다.
static MCP3008_Handle_t measurement_adc;                // MCP3008 통신 상태를 저장한다.
static SensorManager_Handle_t measurement_sensors;      // 통합 센서 상태를 저장한다.
static SensorRawMeasurement_t measurement_sensor_raw;   // 0단계 실측값을 저장한다.
static CRSF_Receiver_t measurement_crsf_receiver;        // USART6 조종기 수신 상태를 저장한다.
static CRSF_Protocol_t measurement_crsf_protocol;        // CRSF 프레임 해석 상태를 저장한다.
static MeasurementRunner_t measurement_runner;          // 현재 실측 단계를 저장한다.
static uint32_t measurement_last_sample_ms;              // 최근 샘플 실행 시각을 저장한다.
static uint32_t measurement_crsf_last_frame_ms;          // 최근 정상 조종기 프레임 시각을 저장한다.
static bool measurement_initialized;                    // 0단계 실행 가능 여부를 저장한다.

/* 최근 조종기 채널과 통신 상태를 디버그 구조체에 기록한다. */
static void MeasurementStage0_ProcessCrsf(uint32_t now_ms)
{
    const uint32_t frames = CRSF_Protocol_ProcessReceiver(&measurement_crsf_protocol,
                                                          &measurement_crsf_receiver);  // 대기 중인 CRSF 프레임을 처리한다.
    uint32_t channel;  // 복사할 조종기 채널 번호를 저장한다.

    g_measurement_debug.crsf_frame_count = measurement_crsf_protocol.frame_counter;            // 정상 프레임 수를 표시한다.
    g_measurement_debug.crsf_crc_error_count = measurement_crsf_protocol.crc_error_count;        // CRC 오류 수를 표시한다.
    g_measurement_debug.crsf_length_error_count = measurement_crsf_protocol.length_error_count;  // 길이 오류 수를 표시한다.
    g_measurement_debug.crsf_rx_overflow_count = measurement_crsf_receiver.overflow_count;       // 버퍼 초과 수를 표시한다.
    g_measurement_debug.crsf_uart_error_count = measurement_crsf_receiver.uart_error_count;      // UART 오류 수를 표시한다.

    if (frames > 0U)
    {
        for (channel = 0U; channel < ROBOT_CRSF_CHANNEL_COUNT; ++channel)
        {
            g_measurement_debug.crsf_current_raw[channel] =
                measurement_crsf_protocol.channel[channel];  // 최근 16채널 raw를 표시한다.
        }

        measurement_crsf_last_frame_ms = now_ms;         // 연결 Timeout 기준을 갱신한다.
        g_measurement_debug.crsf_last_frame_ms = now_ms;  // 마지막 정상 프레임 시각을 표시한다.
        g_measurement_debug.crsf_connected = true;        // 정상 조종기 연결을 표시한다.
    }
    else if ((now_ms - measurement_crsf_last_frame_ms) > MEASUREMENT_STAGE0_CRSF_TIMEOUT_MS)
    {
        g_measurement_debug.crsf_connected = false;  // 정상 프레임이 끊긴 상태를 표시한다.
    }
}

/* GPS·WT931·MCP3008·CRSF를 연결하고 인터럽트 수신을 시작한다. */
bool MeasurementStage0_Init(UART_HandleTypeDef *gps_uart,
                            UART_HandleTypeDef *imu_uart,
                            SPI_HandleTypeDef *adc_spi,
                            UART_HandleTypeDef *crsf_uart)
{
    MeasurementRunner_Init(&measurement_runner);  // 전역 디버그값과 0단계를 초기화한다.
    Relay_Init();                                  // 릴레이 출력을 초기화한다.
    Relay_AllOn();                                 // 센서 확인 중 여섯 서보 전원을 계속 공급한다.
    measurement_initialized = false;               // 초기화 완료 전 실행을 막는다.
    g_measurement_debug.stage0_relay_enable = true;  // 릴레이 상시 ON 상태를 표시한다.
    g_measurement_debug.relay_state_mask = Relay_GetStateMask();  // 실제 릴레이 상태를 표시한다.

    if ((gps_uart == NULL) || (imu_uart == NULL) ||
        (adc_spi == NULL) || (crsf_uart == NULL))
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 잘못된 CubeMX Handle을 기록한다.
        return false;
    }

    GPS_Init(&measurement_gps, gps_uart);  // USART2 GPS를 연결한다.
    IMU_Init(&measurement_imu, imu_uart);  // USART3 WT931을 연결한다.
    CRSF_Receiver_Init(&measurement_crsf_receiver, crsf_uart);  // USART6 조종기를 연결한다.
    CRSF_Protocol_Init(&measurement_crsf_protocol);             // CRSF 해석 상태를 초기화한다.
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
    if (CRSF_Receiver_Start(&measurement_crsf_receiver) != HAL_OK)
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 조종기 수신 시작 실패를 기록한다.
        return false;
    }

    measurement_last_sample_ms = HAL_GetTick();                       // 첫 센서 실행 기준 시각을 저장한다.
    measurement_crsf_last_frame_ms = measurement_last_sample_ms;      // 조종기 연결 Timeout 기준을 저장한다.
    g_measurement_debug.crsf_connected = false;                       // 첫 정상 프레임 전 연결을 해제한다.
    measurement_initialized = true;                                   // 주기 실행을 허용한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK;  // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;               // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 조종기를 처리하고 10 ms마다 GPS·WT931·MCP3008 값을 기록한다. */
void MeasurementStage0_Process(void)
{
    const uint32_t now_ms = HAL_GetTick();  // 현재 실측 시각을 저장한다.

    if (!measurement_initialized)
    {
        return;
    }

    MeasurementStage0_ProcessCrsf(now_ms);  // 수신된 조종기 채널을 즉시 갱신한다.
    if ((now_ms - measurement_last_sample_ms) < MEASUREMENT_STAGE0_SAMPLE_PERIOD_MS)
    {
        return;
    }

    measurement_last_sample_ms = now_ms;                      // 이번 샘플 시각을 저장한다.
    g_measurement_debug.last_sample_ms = now_ms;               // 디버거 샘플 시각을 갱신한다.
    (void)SensorRawMeasurement_Sample(&measurement_sensor_raw); // 전체 센서값을 한 번 기록한다.
}

/* 센서와 조종기의 1바이트 수신 완료를 해당 드라이버에 전달한다. */
void MeasurementStage0_UartRxCallback(UART_HandleTypeDef *uart)
{
    GPS_RxCpltCallback(&measurement_gps, uart);  // USART2이면 다음 GPS 수신을 시작한다.
    IMU_RxCpltCallback(&measurement_imu, uart);  // USART3이면 다음 WT931 수신을 시작한다.
    CRSF_Receiver_RxCpltCallback(&measurement_crsf_receiver, uart);  // USART6이면 다음 CRSF 수신을 시작한다.
}

/* 센서와 조종기의 UART 오류를 해당 드라이버에서 복구한다. */
void MeasurementStage0_UartErrorCallback(UART_HandleTypeDef *uart)
{
    GPS_ErrorCallback(&measurement_gps, uart);  // USART2 GPS 수신을 복구한다.
    IMU_ErrorCallback(&measurement_imu, uart);  // USART3 WT931 수신을 복구한다.
    CRSF_Receiver_ErrorCallback(&measurement_crsf_receiver, uart);  // USART6 CRSF 수신을 복구한다.
}
