#include "measurement/measurement_stage6.h"

#include "common/robot_calibration.h"
#include "low_control/relay.h"
#include "measurement/foot_pressure_measurement.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "sensor/mcp3008.h"

#include <stddef.h>
#include <string.h>

#define MEASUREMENT_STAGE6_SAMPLE_PERIOD_MS       10U
#define MEASUREMENT_STAGE6_WAIT_UNLOADED_MS    10000U
#define MEASUREMENT_STAGE6_CAPTURE_UNLOADED_MS 10000U
#define MEASUREMENT_STAGE6_WAIT_LOADED_MS      15000U
#define MEASUREMENT_STAGE6_CAPTURE_LOADED_MS    5000U
#define MEASUREMENT_STAGE6_MINIMUM_DIFFERENCE     10

static MCP3008_Handle_t measurement_adc;                 // MCP3008 통신 상태를 저장한다.
static MCP3008_Data_t measurement_adc_data;              // 최근 ADC 24채널 값을 저장한다.
static FootPressureMeasurement_t measurement_pressure;   // 압력센서 표본 누적값을 저장한다.
static FootPressure_Calibration_t measurement_table[ROBOT_PRESSURE_COUNT]; // 완성한 압력 보정표를 저장한다.
static MeasurementRunner_t measurement_runner;           // 현재 실측 단계를 저장한다.
static uint16_t measurement_pressure_raw[ROBOT_PRESSURE_COUNT]; // 최근 압력센서값을 저장한다.
static uint32_t measurement_loaded_sum[ROBOT_PRESSURE_COUNT]; // 다리별 누름 raw 합계를 저장한다.
static uint16_t measurement_loaded_count[ROBOT_PRESSURE_COUNT]; // 다리별 누름 표본 수를 저장한다.
static uint32_t measurement_last_sample_ms;               // 최근 ADC 실행 시각을 저장한다.
static uint32_t measurement_phase_start_ms;               // 현재 단계 시작 시각을 저장한다.
static uint8_t measurement_active_leg;                    // 현재 누를 다리 번호 0~5를 저장한다.
static MeasurementStage6_Phase_t measurement_phase;      // 현재 압력 측정 단계를 저장한다.
static bool measurement_initialized;                     // 6단계 실행 가능 여부를 저장한다.

/* 새 측정 단계와 시작 시각을 저장한다. */
static void MeasurementStage6_SetPhase(MeasurementStage6_Phase_t phase,
                                       uint32_t now_ms)
{
    measurement_phase = phase;                                     // 새 측정 단계를 저장한다.
    measurement_phase_start_ms = now_ms;                            // 새 단계 시작 시각을 저장한다.
    g_measurement_debug.pressure_measurement_phase = (uint8_t)phase; // Live Expressions 단계를 갱신한다.
    g_measurement_debug.pressure_phase_elapsed_seconds = 0U;         // 단계 경과 시간을 초기화한다.
}

/* 여섯 압력센서의 최근값과 평균을 디버거에 갱신한다. */
static void MeasurementStage6_UpdateDebugValues(void)
{
    uint32_t leg;  // 갱신할 다리 번호를 저장한다.

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        uint16_t unloaded_average = 0U;  // 현재 무부하 평균을 저장한다.
        uint16_t loaded_average = 0U;    // 현재 접촉 평균을 저장한다.

        if (measurement_pressure.unloaded_count > 0U)
        {
            unloaded_average =
                (uint16_t)(measurement_pressure.unloaded_sum[leg] /
                           measurement_pressure.unloaded_count);  // 무부하 누적값을 평균낸다.
        }
        if (measurement_loaded_count[leg] > 0U)
        {
            loaded_average =
                (uint16_t)(measurement_loaded_sum[leg] /
                           measurement_loaded_count[leg]);        // 다리별 누름값을 평균낸다.
        }

        g_measurement_debug.pressure_current_raw[leg] =
            measurement_pressure_raw[leg];                         // 최근 압력값을 표시한다.
        g_measurement_debug.pressure_unloaded_average[leg] =
            unloaded_average;                                      // 무부하 평균을 표시한다.
        g_measurement_debug.pressure_loaded_average[leg] =
            loaded_average;                                        // 접촉 평균을 표시한다.
        g_measurement_debug.pressure_loaded_sample_count[leg] =
            measurement_loaded_count[leg];                         // 다리별 누름 표본 수를 표시한다.
        g_measurement_debug.pressure_difference_raw[leg] =
            (measurement_loaded_count[leg] == 0U) ? 0 :
            (int16_t)((int32_t)loaded_average -
                      (int32_t)unloaded_average);                   // 측정된 다리의 변화량을 표시한다.
    }
}

/* 현재 ADC 결과에서 여섯 압력센서 채널을 선택한다. */
static bool MeasurementStage6_ReadPressure(void)
{
    uint32_t leg;      // 읽을 다리 번호를 저장한다.
    uint32_t device;   // 복사할 MCP3008 번호를 저장한다.
    uint32_t channel;  // 복사할 ADC 채널 번호를 저장한다.

    if (MCP3008_ReadAll(&measurement_adc, &measurement_adc_data) != HAL_OK)
    {
        g_measurement_debug.adc_driver_error_count = measurement_adc_data.error_count;        // ADC 오류 수를 갱신한다.
        g_measurement_debug.adc_last_error_device = measurement_adc_data.last_error_device;   // 실패 장치를 갱신한다.
        g_measurement_debug.adc_last_error_channel = measurement_adc_data.last_error_channel; // 실패 채널을 갱신한다.
        return false;
    }

    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            g_measurement_debug.adc_raw[device][channel] =
                measurement_adc_data.raw[device][channel];  // 최근 ADC 24채널을 표시한다.
        }
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        const MCP3008_InputMapping_t *mapping =
            &g_robot_calibration.adc[leg][MCP3008_LEG_PRESSURE];  // 현재 다리 압력 채널을 선택한다.

        if ((mapping->device >= MCP3008_DEVICE_COUNT) ||
            (mapping->channel >= MCP3008_CHANNEL_COUNT))
        {
            return false;
        }
        measurement_pressure_raw[leg] =
            measurement_adc_data.raw[mapping->device][mapping->channel];  // 실측한 배선에서 압력값을 읽는다.
    }

    g_measurement_debug.last_sample_ms = measurement_adc_data.mcu_time_ms;       // 최근 ADC 시각을 표시한다.
    g_measurement_debug.adc_update_count = measurement_adc_data.update_counter;  // 정상 ADC 횟수를 표시한다.
    MeasurementStage6_UpdateDebugValues();                                       // 최근 압력값을 표시한다.
    return true;
}

/* 두 자세의 평균값으로 압력 임계값을 생성한다. */
static bool MeasurementStage6_BuildTable(void)
{
    uint32_t leg;  // 검사할 다리 번호를 저장한다.

    if (measurement_pressure.unloaded_count == 0U)
    {
        g_measurement_debug.pressure_measurement_error_leg = 0U;  // 전체 표본 부족을 표시한다.
        return false;
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        const uint16_t unloaded = g_measurement_debug.pressure_unloaded_average[leg]; // 무부하 평균을 읽는다.
        const uint16_t loaded = g_measurement_debug.pressure_loaded_average[leg];     // 누름 평균을 읽는다.
        const int32_t difference = (int32_t)loaded - (int32_t)unloaded;                // 다리별 접촉 변화량을 계산한다.

        if ((measurement_loaded_count[leg] == 0U) ||
            ((difference > -MEASUREMENT_STAGE6_MINIMUM_DIFFERENCE) &&
             (difference < MEASUREMENT_STAGE6_MINIMUM_DIFFERENCE)))
        {
            g_measurement_debug.pressure_measurement_error_leg =
                (uint8_t)(leg + 1U);  // 변화가 부족한 다리 번호를 표시한다.
            return false;
        }

        measurement_table[leg].active_high = (difference > 0);  // 누를 때 raw 증가 여부를 저장한다.
        measurement_table[leg].release_threshold =
            (uint16_t)((int32_t)unloaded + difference / 20);     // 무부하에서 5% 지점을 해제값으로 만든다.
        measurement_table[leg].contact_threshold =
            (uint16_t)((int32_t)unloaded + difference / 10);     // 무부하에서 10% 지점을 접촉값으로 만든다.
        measurement_table[leg].calibrated = true;                // 실측 완료를 표시한다.
        g_measurement_debug.calibration.pressure[leg] =
            measurement_table[leg];                              // 중앙 표 복사용 임계값을 갱신한다.
    }

    return true;
}

/* SPI1 ADC와 자동 압력센서 측정을 준비한다. */
bool MeasurementStage6_Init(SPI_HandleTypeDef *adc_spi)
{
    uint32_t skip;    // 건너뛸 완료 단계 수를 저장한다.
    uint32_t now_ms;  // 초기화 완료 시각을 저장한다.

    MeasurementRunner_Init(&measurement_runner);  // 전역 디버그값을 초기화한다.
    for (skip = 0U; skip < (uint32_t)MEASUREMENT_STAGE_FOOT_PRESSURE; ++skip)
    {
        (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 0~5단계를 건너뛴다.
    }
    g_measurement_debug.calibration = g_robot_calibration;   // 앞 단계의 중앙 설정값을 유지한다.
    memset(&measurement_adc_data, 0, sizeof(measurement_adc_data)); // 이전 ADC 결과를 제거한다.
    memset(measurement_pressure_raw, 0, sizeof(measurement_pressure_raw)); // 이전 압력값을 제거한다.
    memset(measurement_loaded_sum, 0, sizeof(measurement_loaded_sum)); // 이전 누름 합계를 제거한다.
    memset(measurement_loaded_count, 0, sizeof(measurement_loaded_count)); // 이전 누름 표본 수를 제거한다.
    memset(measurement_table, 0, sizeof(measurement_table));  // 이전 압력 보정표를 제거한다.
    FootPressureMeasurement_Init(&measurement_pressure);     // 무부하와 접촉 누적값을 준비한다.
    Relay_Init();                                            // 릴레이 출력을 안전하게 준비한다.
    Relay_AllOff();                                          // 실측 중 모든 서보 전원을 차단한다.
    g_measurement_debug.relay_state_mask = 0U;                // 전체 릴레이 OFF를 표시한다.
    g_measurement_debug.pressure_measurement_complete = false; // 미완료 상태로 시작한다.
    g_measurement_debug.pressure_measurement_failed = false;   // 정상 상태로 시작한다.
    g_measurement_debug.pressure_measurement_error_leg = 0U;   // 오류 다리 표시를 제거한다.
    g_measurement_debug.pressure_measurement_active_leg = 0U;  // 무부하 단계에는 대상 다리가 없음을 표시한다.
    measurement_active_leg = 0U;                               // 첫 누름 대상을 1번 다리로 준비한다.
    measurement_initialized = false;                          // 초기화 완료 전 실행을 막는다.

    if ((adc_spi == NULL) || !g_robot_calibration.adc_mapping_calibrated)
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // Handle 또는 ADC 배치 오류를 기록한다.
        return false;
    }
    if (MCP3008_Init(&measurement_adc, adc_spi) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_ADC;  // ADC 초기화 실패를 기록한다.
        return false;
    }

    now_ms = HAL_GetTick();                                      // 자동 측정 시작 시각을 읽는다.
    measurement_last_sample_ms = now_ms;                         // 첫 ADC 실행 기준 시각을 저장한다.
    MeasurementStage6_SetPhase(MEASUREMENT_STAGE6_WAIT_UNLOADED,
                               now_ms);                           // 무부하 준비 시간부터 시작한다.
    measurement_initialized = true;                              // 주기 실행을 허용한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK; // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;           // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 10 ms마다 압력값을 읽고 두 자세의 평균과 임계값을 만든다. */
void MeasurementStage6_Process(void)
{
    const uint32_t now_ms = HAL_GetTick();  // 현재 실측 시각을 저장한다.
    const uint32_t elapsed_ms = now_ms - measurement_phase_start_ms; // 현재 단계 경과 시간을 계산한다.

    if (!measurement_initialized ||
        ((now_ms - measurement_last_sample_ms) < MEASUREMENT_STAGE6_SAMPLE_PERIOD_MS))
    {
        return;
    }

    measurement_last_sample_ms = now_ms;  // 이번 ADC 실행 시각을 저장한다.
    g_measurement_debug.pressure_phase_elapsed_seconds =
        (uint8_t)(elapsed_ms / 1000U);      // 현재 단계 경과 초를 표시한다.
    if (!MeasurementStage6_ReadPressure())
    {
        MeasurementStage6_SetPhase(MEASUREMENT_STAGE6_FAILED, now_ms); // ADC 오류 상태로 전환한다.
        g_measurement_debug.pressure_measurement_failed = true;         // 측정 실패를 표시한다.
        measurement_initialized = false;                                // 오류 후 자동 재실행을 막는다.
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE6_WAIT_UNLOADED)
    {
        if (elapsed_ms >= MEASUREMENT_STAGE6_WAIT_UNLOADED_MS)
        {
            MeasurementStage6_SetPhase(MEASUREMENT_STAGE6_CAPTURE_UNLOADED,
                                       now_ms);  // 무부하 평균 측정을 시작한다.
        }
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE6_CAPTURE_UNLOADED)
    {
        FootPressureMeasurement_AddUnloaded(&measurement_pressure,
                                             measurement_pressure_raw);  // 무부하 표본을 누적한다.
        MeasurementStage6_UpdateDebugValues();                           // 새 평균값을 표시한다.
        if (elapsed_ms >= MEASUREMENT_STAGE6_CAPTURE_UNLOADED_MS)
        {
            g_measurement_debug.pressure_measurement_active_leg =
                (uint8_t)(measurement_active_leg + 1U);        // 처음 누를 다리를 표시한다.
            MeasurementStage6_SetPhase(MEASUREMENT_STAGE6_WAIT_LOADED,
                                       now_ms);  // 표시된 발을 누를 준비 시간을 시작한다.
        }
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE6_WAIT_LOADED)
    {
        if (elapsed_ms >= MEASUREMENT_STAGE6_WAIT_LOADED_MS)
        {
            MeasurementStage6_SetPhase(MEASUREMENT_STAGE6_CAPTURE_LOADED,
                                       now_ms);  // 접촉 평균 측정을 시작한다.
        }
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE6_CAPTURE_LOADED)
    {
        measurement_loaded_sum[measurement_active_leg] +=
            measurement_pressure_raw[measurement_active_leg];  // 표시된 발의 누름값만 누적한다.
        ++measurement_loaded_count[measurement_active_leg];     // 현재 다리의 표본 수를 증가시킨다.
        g_measurement_debug.pressure_loaded_sum[measurement_active_leg] =
            measurement_loaded_sum[measurement_active_leg];     // 현재 다리의 누름 합계를 표시한다.
        g_measurement_debug.pressure_loaded_count =
            measurement_loaded_count[measurement_active_leg];   // 현재 다리의 표본 수를 표시한다.
        MeasurementStage6_UpdateDebugValues();                   // 새 평균과 변화량을 표시한다.
        if (elapsed_ms < MEASUREMENT_STAGE6_CAPTURE_LOADED_MS)
        {
            return;
        }

        if (measurement_active_leg + 1U < ROBOT_PRESSURE_COUNT)
        {
            ++measurement_active_leg;  // 다음 다리로 이동한다.
            g_measurement_debug.pressure_measurement_active_leg =
                (uint8_t)(measurement_active_leg + 1U);  // 다음에 누를 다리를 표시한다.
            g_measurement_debug.pressure_loaded_count = 0U;  // 다음 다리 표본 수를 초기화한다.
            MeasurementStage6_SetPhase(MEASUREMENT_STAGE6_WAIT_LOADED,
                                       now_ms);  // 이전 발을 놓고 다음 발을 누를 시간을 시작한다.
            return;
        }

        if (!MeasurementStage6_BuildTable())
        {
            MeasurementStage6_SetPhase(MEASUREMENT_STAGE6_FAILED, now_ms); // 부족한 변화량을 오류로 표시한다.
            g_measurement_debug.pressure_measurement_failed = true;         // 측정 실패를 표시한다.
            measurement_initialized = false;                                // 실패 결과를 유지한다.
            return;
        }

        g_measurement_debug.pressure_measurement_complete = true;  // 여섯 압력센서 보정 완료를 표시한다.
        MeasurementStage6_SetPhase(MEASUREMENT_STAGE6_COMPLETE,
                                   now_ms);                          // 완료 상태를 유지한다.
        (void)MeasurementRunner_CompleteCurrent(&measurement_runner); // 다음 실측 단계로 이동한다.
        measurement_initialized = false;                            // 완료 결과를 고정한다.
    }
}
