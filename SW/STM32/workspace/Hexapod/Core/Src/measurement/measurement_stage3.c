#include "measurement/measurement_stage3.h"

#include "common/robot_calibration.h"
#include "low_control/relay.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "sensor/mcp3008.h"

#include <stddef.h>
#include <string.h>

#define MEASUREMENT_STAGE3_SAMPLE_PERIOD_MS  20U
#define MEASUREMENT_STAGE3_RELAY_TIME_MS  10000U

static MCP3008_Handle_t measurement_adc;                // MCP3008 통신 상태를 저장한다.
static MCP3008_Data_t measurement_adc_data;             // 최근 ADC 24채널 값을 저장한다.
static MeasurementRunner_t measurement_runner;          // 현재 실측 단계를 저장한다.
static uint32_t measurement_last_sample_ms;              // 최근 ADC 실행 시각을 저장한다.
static uint32_t measurement_relay_start_ms;              // 현재 릴레이 시작 시각을 저장한다.
static uint8_t measurement_relay_channel;                // 현재 ON인 릴레이 배열 번호를 저장한다.
static bool measurement_initialized;                     // 3단계 실행 가능 여부를 저장한다.

/* 현재 채널 하나만 켜고 시작 시각과 디버그값을 갱신한다. */
static void MeasurementStage3_ActivateRelay(uint32_t now_ms)
{
    Relay_AllOff();                                               // 이전 릴레이를 먼저 끈다.
    Relay_On((Relay_Channel_t)measurement_relay_channel);         // 현재 시험 릴레이 하나만 켠다.
    measurement_relay_start_ms = now_ms;                          // 새 릴레이 시작 시각을 저장한다.
    g_measurement_debug.relay_test_active_channel =
        (uint8_t)(measurement_relay_channel + 1U);                 // 현재 채널을 1~6으로 표시한다.
    g_measurement_debug.relay_test_elapsed_seconds = 0U;          // 새 채널 경과 시간을 초기화한다.
    g_measurement_debug.relay_state_mask = Relay_GetStateMask();  // 실제 한 채널 ON 상태를 표시한다.
}

/* 실측한 ADC 매핑으로 다리 1~6의 3번 관절값을 갱신한다. */
static void MeasurementStage3_UpdateLeg3Raw(void)
{
    uint32_t leg;  // 갱신할 다리 번호를 저장한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const MCP3008_InputMapping_t *mapping =
            &g_robot_calibration.adc[leg][MCP3008_LEG_JOINT_3];  // 해당 다리의 J3 채널을 선택한다.

        g_measurement_debug.relay_test_leg3_raw[leg] =
            measurement_adc_data.raw[mapping->device][mapping->channel];  // 다리별 J3 ADC를 표시한다.
    }
}

/* SPI1 ADC와 10초 릴레이 순환 시험을 준비한다. */
bool MeasurementStage3_Init(SPI_HandleTypeDef *adc_spi)
{
    MeasurementRunner_Init(&measurement_runner);                   // 전역 디버그값을 초기화한다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 0단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 1단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 2단계를 건너뛴다.
    g_measurement_debug.calibration = g_robot_calibration;         // 앞 단계의 중앙 설정값을 유지한다.
    memset(&measurement_adc_data, 0, sizeof(measurement_adc_data)); // 이전 ADC 값을 제거한다.
    Relay_Init();                                                  // 모든 릴레이를 OFF로 준비한다.
    measurement_initialized = false;                              // 초기화 완료 전 실행을 막는다.
    measurement_relay_channel = 0U;                               // INA1부터 시험한다.
    g_measurement_debug.relay_state_mask = 0U;                     // 릴레이 OFF 상태를 표시한다.
    g_measurement_debug.relay_test_active_channel = 0U;            // 초기화 중 활성 채널이 없음을 표시한다.
    g_measurement_debug.relay_test_elapsed_seconds = 0U;           // 경과 시간을 초기화한다.
    g_measurement_debug.relay_test_cycle_count = 0U;               // 전체 반복 횟수를 초기화한다.

    if (adc_spi == NULL)
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 잘못된 SPI1 Handle을 기록한다.
        return false;
    }
    if (MCP3008_Init(&measurement_adc, adc_spi) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_ADC;  // ADC 초기화 실패를 기록한다.
        return false;
    }

    measurement_last_sample_ms = HAL_GetTick();                        // 첫 ADC 실행 기준 시각을 저장한다.
    MeasurementStage3_ActivateRelay(measurement_last_sample_ms);       // INA1 시험을 즉시 시작한다.
    measurement_initialized = true;                                    // 주기 실행을 허용한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK;  // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;                 // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 다리 1~6의 J3 ADC를 표시하며 릴레이를 10초마다 순환한다. */
void MeasurementStage3_Process(void)
{
    const uint32_t now_ms = HAL_GetTick();  // 현재 실측 시각을 저장한다.

    if (!measurement_initialized)
    {
        return;
    }

    if ((now_ms - measurement_relay_start_ms) >= MEASUREMENT_STAGE3_RELAY_TIME_MS)
    {
        ++measurement_relay_channel;  // 다음 릴레이 채널로 이동한다.
        if (measurement_relay_channel >= RELAY_CHANNEL_COUNT)
        {
            measurement_relay_channel = 0U;                      // 여섯 채널 후 INA1로 돌아간다.
            ++g_measurement_debug.relay_test_cycle_count;        // 전체 한 바퀴 완료를 기록한다.
        }
        MeasurementStage3_ActivateRelay(now_ms);  // 다음 릴레이의 10초 시험을 시작한다.
    }
    else
    {
        g_measurement_debug.relay_test_elapsed_seconds =
            (uint8_t)((now_ms - measurement_relay_start_ms) / 1000U);  // 현재 채널 경과 초를 표시한다.
    }

    if ((now_ms - measurement_last_sample_ms) < MEASUREMENT_STAGE3_SAMPLE_PERIOD_MS)
    {
        return;
    }

    measurement_last_sample_ms = now_ms;  // 이번 ADC 실행 시각을 저장한다.
    if (MCP3008_ReadAll(&measurement_adc, &measurement_adc_data) != HAL_OK)
    {
        Relay_AllOff();                                                         // ADC 오류 시 모든 릴레이를 끈다.
        g_measurement_debug.relay_state_mask = 0U;                              // 실제 OFF 상태를 표시한다.
        g_measurement_debug.adc_driver_error_count = measurement_adc_data.error_count;        // ADC 오류 수를 갱신한다.
        g_measurement_debug.adc_last_error_device = measurement_adc_data.last_error_device;   // 실패 장치를 갱신한다.
        g_measurement_debug.adc_last_error_channel = measurement_adc_data.last_error_channel; // 실패 채널을 갱신한다.
        measurement_initialized = false;                                       // 오류 후 자동 재출력을 막는다.
        return;
    }

    g_measurement_debug.last_sample_ms = measurement_adc_data.mcu_time_ms;          // 최근 ADC 시각을 갱신한다.
    g_measurement_debug.adc_update_count = measurement_adc_data.update_counter;     // 정상 ADC 횟수를 갱신한다.
    g_measurement_debug.adc_driver_error_count = measurement_adc_data.error_count;  // ADC 오류 수를 갱신한다.
    MeasurementStage3_UpdateLeg3Raw();                                             // 다리 1~6의 J3 값을 갱신한다.
}
