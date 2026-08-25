#include "measurement/measurement_stage2.h"

#include "common/robot_calibration.h"
#include "low_control/relay.h"
#include "measurement/adc_mapping_measurement.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "sensor/mcp3008.h"

#include <stddef.h>
#include <string.h>

#define MEASUREMENT_STAGE2_SAMPLE_PERIOD_MS       10U
#define MEASUREMENT_STAGE2_SETTLE_TIME_MS       1500U
#define MEASUREMENT_STAGE2_MINIMUM_RANGE          80U
#define MEASUREMENT_STAGE2_DOMINANCE_MARGIN       20U
#define MEASUREMENT_STAGE2_STABLE_SAMPLE_COUNT     5U

static MCP3008_Handle_t measurement_adc;                       // MCP3008 통신 상태를 저장한다.
static MCP3008_Data_t measurement_adc_data;                    // 최근 ADC 24채널 값을 저장한다.
static AdcMappingMeasurement_t measurement_mapping;            // 확인한 ADC 배선을 저장한다.
static MeasurementRunner_t measurement_runner;                 // 현재 실측 단계를 저장한다.
static bool measurement_channel_used[MCP3008_DEVICE_COUNT]
                                    [MCP3008_CHANNEL_COUNT];    // 이미 기록한 ADC 채널을 저장한다.
static uint16_t measurement_minimum[MCP3008_DEVICE_COUNT]
                                   [MCP3008_CHANNEL_COUNT];     // 현재 대상의 채널별 최소값을 저장한다.
static uint16_t measurement_maximum[MCP3008_DEVICE_COUNT]
                                   [MCP3008_CHANNEL_COUNT];     // 현재 대상의 채널별 최대값을 저장한다.
static uint32_t measurement_last_sample_ms;                     // 최근 ADC 실행 시각을 저장한다.
static uint32_t measurement_settle_until_ms;                    // 다음 대상의 대기 종료 시각을 저장한다.
static uint8_t measurement_target_index;                        // 현재 논리 센서 번호 0~23을 저장한다.
static uint8_t measurement_stable_device;                       // 연속 확인 중인 MCP3008 번호를 저장한다.
static uint8_t measurement_stable_channel;                      // 연속 확인 중인 ADC 채널을 저장한다.
static uint8_t measurement_stable_count;                        // 같은 후보의 연속 확인 횟수를 저장한다.
static bool measurement_initialized;                            // 2단계 실행 가능 여부를 저장한다.
static bool measurement_range_ready;                            // 현재 대상의 움직임 감지를 허용한다.

/* 현재 논리 센서 번호를 사용자 확인용 다리와 입력 번호로 표시한다. */
static void MeasurementStage2_UpdateTarget(void)
{
    g_measurement_debug.adc_mapping_target_leg =
        (uint8_t)((measurement_target_index / MCP3008_LEG_INPUT_COUNT) + 1U);  // 다리 번호를 1~6으로 표시한다.
    g_measurement_debug.adc_mapping_target_input =
        (uint8_t)((measurement_target_index % MCP3008_LEG_INPUT_COUNT) + 1U);  // 입력 번호를 1~4로 표시한다.
}

/* 현재 ADC 값부터 새 대상 센서의 변화 폭을 측정한다. */
static void MeasurementStage2_ResetRange(void)
{
    uint32_t device;   // 초기화할 MCP3008 번호를 저장한다.
    uint32_t channel;  // 초기화할 ADC 채널 번호를 저장한다.

    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            const uint16_t raw = measurement_adc_data.raw[device][channel];  // 현재값을 새 기준으로 읽는다.

            measurement_minimum[device][channel] = raw;                // 현재값을 최소값으로 시작한다.
            measurement_maximum[device][channel] = raw;                // 현재값을 최대값으로 시작한다.
            g_measurement_debug.adc_range[device][channel] = 0U;       // 이전 대상의 변화 폭을 제거한다.
        }
    }

    measurement_stable_device = MCP3008_INVALID_INDEX;                   // 이전 연속 후보 장치를 제거한다.
    measurement_stable_channel = MCP3008_INVALID_INDEX;                  // 이전 연속 후보 채널을 제거한다.
    measurement_stable_count = 0U;                                       // 이전 연속 확인 횟수를 제거한다.
    g_measurement_debug.adc_mapping_candidate_device = 0U;               // 아직 후보 장치가 없음을 표시한다.
    g_measurement_debug.adc_mapping_candidate_channel = MCP3008_INVALID_INDEX; // 아직 후보 채널이 없음을 표시한다.
    g_measurement_debug.adc_mapping_candidate_range = 0U;                // 아직 후보 변화가 없음을 표시한다.
    g_measurement_debug.adc_mapping_ambiguous = false;                   // 새 대상을 정상 상태로 시작한다.
    g_measurement_debug.adc_mapping_waiting_motion = true;               // 이제 표시된 센서를 움직이도록 알린다.
    measurement_range_ready = true;                                     // 변화 폭 추적을 허용한다.
}

/* 다음 논리 센서를 표시하고 이전 센서가 멈출 시간을 기다린다. */
static void MeasurementStage2_PrepareNext(uint32_t now_ms)
{
    MeasurementStage2_UpdateTarget();                              // 다음 다리와 센서 번호를 표시한다.
    measurement_settle_until_ms = now_ms + MEASUREMENT_STAGE2_SETTLE_TIME_MS; // 이전 움직임이 멈출 시간을 둔다.
    measurement_range_ready = false;                               // 대기 중 변화 감지를 막는다.
    g_measurement_debug.adc_mapping_waiting_motion = false;        // 아직 움직이지 않도록 표시한다.
}

/* 미사용 채널 중 가장 큰 변화와 두 번째 변화를 찾는다. */
static void MeasurementStage2_FindCandidate(uint8_t *best_device,
                                             uint8_t *best_channel,
                                             uint16_t *best_range,
                                             uint16_t *second_range)
{
    uint32_t device;   // 검사할 MCP3008 번호를 저장한다.
    uint32_t channel;  // 검사할 ADC 채널 번호를 저장한다.

    *best_device = MCP3008_INVALID_INDEX;  // 후보 장치를 미검출로 시작한다.
    *best_channel = MCP3008_INVALID_INDEX; // 후보 채널을 미검출로 시작한다.
    *best_range = 0U;                      // 최대 변화 폭을 초기화한다.
    *second_range = 0U;                    // 두 번째 변화 폭을 초기화한다.

    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            const uint16_t range = g_measurement_debug.adc_range[device][channel];  // 현재 채널 변화 폭을 읽는다.

            if (measurement_channel_used[device][channel])
            {
                continue;  // 이미 기록한 채널은 다음 후보에서 제외한다.
            }
            if (range > *best_range)
            {
                *second_range = *best_range;      // 이전 최대값을 두 번째 값으로 내린다.
                *best_range = range;              // 새 최대 변화 폭을 저장한다.
                *best_device = (uint8_t)device;   // 새 후보 장치를 저장한다.
                *best_channel = (uint8_t)channel; // 새 후보 채널을 저장한다.
            }
            else if (range > *second_range)
            {
                *second_range = range;  // 두 번째로 큰 변화 폭을 갱신한다.
            }
        }
    }
}

/* 새 ADC 값으로 미사용 채널의 움직임 범위를 갱신한다. */
static void MeasurementStage2_UpdateRange(void)
{
    uint32_t device;   // 갱신할 MCP3008 번호를 저장한다.
    uint32_t channel;  // 갱신할 ADC 채널 번호를 저장한다.

    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            const uint16_t raw = measurement_adc_data.raw[device][channel];  // 최근 채널값을 읽는다.

            g_measurement_debug.adc_raw[device][channel] = raw;  // 최근 ADC 값을 디버거에 갱신한다.
            if (measurement_channel_used[device][channel])
            {
                continue;  // 완료한 채널의 변화 폭은 다시 추적하지 않는다.
            }
            if (raw < measurement_minimum[device][channel])
            {
                measurement_minimum[device][channel] = raw;  // 새 최소값을 저장한다.
            }
            if (raw > measurement_maximum[device][channel])
            {
                measurement_maximum[device][channel] = raw;  // 새 최대값을 저장한다.
            }
            g_measurement_debug.adc_range[device][channel] =
                (uint16_t)(measurement_maximum[device][channel] -
                           measurement_minimum[device][channel]);  // 채널 변화 폭을 갱신한다.
        }
    }
}

/* 확실하게 구분된 후보를 현재 논리 센서에 기록한다. */
static bool MeasurementStage2_TryCapture(uint32_t now_ms)
{
    uint8_t best_device;   // 가장 큰 변화의 MCP3008 번호를 저장한다.
    uint8_t best_channel;  // 가장 큰 변화의 ADC 채널을 저장한다.
    uint16_t best_range;   // 가장 큰 변화 폭을 저장한다.
    uint16_t second_range; // 두 번째 변화 폭을 저장한다.
    uint8_t leg;           // 기록할 다리 배열 번호를 저장한다.
    uint8_t input;         // 기록할 다리 입력 번호를 저장한다.
    bool dominant;         // 후보 채널 구분 여부를 저장한다.

    MeasurementStage2_FindCandidate(&best_device,
                                    &best_channel,
                                    &best_range,
                                    &second_range);  // 현재 가장 큰 두 변화를 찾는다.
    g_measurement_debug.adc_mapping_candidate_device =
        (best_device == MCP3008_INVALID_INDEX) ? 0U : (uint8_t)(best_device + 1U); // MCP3008 번호를 1~3으로 표시한다.
    g_measurement_debug.adc_mapping_candidate_channel = best_channel;              // ADC 채널 번호를 표시한다.
    g_measurement_debug.adc_mapping_candidate_range = best_range;                  // 최대 변화 폭을 표시한다.

    dominant = (best_range >= MEASUREMENT_STAGE2_MINIMUM_RANGE) &&
               (best_range >= (uint16_t)(second_range + MEASUREMENT_STAGE2_DOMINANCE_MARGIN)); // 한 채널의 확실한 움직임을 검사한다.
    g_measurement_debug.adc_mapping_ambiguous =
        (best_range >= MEASUREMENT_STAGE2_MINIMUM_RANGE) && !dominant;  // 여러 채널이 함께 움직였는지 표시한다.
    if (!dominant)
    {
        measurement_stable_count = 0U;  // 불확실한 후보의 연속 기록을 제거한다.
        return false;
    }

    if ((best_device != measurement_stable_device) ||
        (best_channel != measurement_stable_channel))
    {
        measurement_stable_device = best_device;    // 새 연속 후보 장치를 저장한다.
        measurement_stable_channel = best_channel;  // 새 연속 후보 채널을 저장한다.
        measurement_stable_count = 1U;               // 새 후보 확인을 시작한다.
        return false;
    }
    if (measurement_stable_count < MEASUREMENT_STAGE2_STABLE_SAMPLE_COUNT)
    {
        ++measurement_stable_count;  // 같은 후보의 연속 확인 횟수를 늘린다.
        return false;
    }

    leg = (uint8_t)(measurement_target_index / MCP3008_LEG_INPUT_COUNT);    // 다리 배열 번호를 계산한다.
    input = (uint8_t)(measurement_target_index % MCP3008_LEG_INPUT_COUNT);  // 입력 배열 번호를 계산한다.
    if (!AdcMappingMeasurement_Record(&measurement_mapping,
                                      leg,
                                      (MCP3008_LegInput_t)input,
                                      best_device,
                                      best_channel))
    {
        return false;
    }

    measurement_channel_used[best_device][best_channel] = true;        // 기록한 채널의 재사용을 막는다.
    ++measurement_target_index;                                       // 다음 논리 센서로 이동한다.
    g_measurement_debug.adc_mapping_completed_count = measurement_target_index; // 완료 개수를 갱신한다.
    if (measurement_target_index >= (MCP3008_LEG_COUNT * MCP3008_LEG_INPUT_COUNT))
    {
        g_measurement_debug.adc_mapping_complete =
            AdcMappingMeasurement_IsComplete(&measurement_mapping);    // 중복 없는 24채널 완료를 확인한다.
        g_measurement_debug.adc_mapping_waiting_motion = false;        // 추가 센서 동작을 막는다.
        g_measurement_debug.calibration.adc_mapping_calibrated =
            g_measurement_debug.adc_mapping_complete;                  // 중앙 표 복사용 완료 상태를 갱신한다.
        Relay_AllOff();                                                // 실측 완료 후 서보 전원을 차단한다.
        g_measurement_debug.relay_state_mask = 0U;                     // 릴레이 OFF 상태를 표시한다.
        if (g_measurement_debug.adc_mapping_complete)
        {
            (void)MeasurementRunner_CompleteCurrent(&measurement_runner); // 완료하고 다음 실측 단계로 이동한다.
        }
        return true;
    }

    MeasurementStage2_PrepareNext(now_ms);  // 다음 센서가 안정될 시간을 시작한다.
    return true;
}

/* SPI1 MCP3008과 자동 ADC 매핑 상태를 준비한다. */
bool MeasurementStage2_Init(SPI_HandleTypeDef *adc_spi)
{
    MeasurementRunner_Init(&measurement_runner);                   // 전역 디버그값을 초기화한다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 0단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 1단계를 건너뛴다.
    g_measurement_debug.calibration = g_robot_calibration;         // 앞 단계의 중앙 설정값을 유지한다.
    memset(&measurement_adc_data, 0, sizeof(measurement_adc_data)); // 이전 ADC 값을 제거한다.
    memset(measurement_channel_used, 0, sizeof(measurement_channel_used)); // 채널 사용 기록을 제거한다.
    measurement_initialized = false;                               // 초기화 완료 전 실행을 막는다.
    measurement_range_ready = false;                               // 첫 ADC 기준값 전 감지를 막는다.
    measurement_target_index = 0U;                                 // 다리 1의 J1부터 시작한다.
    g_measurement_debug.adc_mapping_completed_count = 0U;          // 완료 센서 수를 초기화한다.
    g_measurement_debug.adc_mapping_complete = false;              // 매핑 미완료 상태로 둔다.
    g_measurement_debug.adc_mapping_waiting_motion = false;        // 첫 안정화 전 동작을 막는다.
    g_measurement_debug.adc_mapping_ambiguous = false;             // 채널 구분 정상 상태로 시작한다.
    Relay_Init();                                                  // PWM 없이 릴레이를 준비한다.

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

    AdcMappingMeasurement_Init(&measurement_mapping);                  // 논리 입력 기록을 준비한다.
    Relay_AllOn();                                                     // PWM 없이 서보 각도센서 전원을 공급한다.
    g_measurement_debug.relay_state_mask = Relay_GetStateMask();       // 실제 릴레이 ON 상태를 표시한다.
    measurement_last_sample_ms = HAL_GetTick();                        // 첫 ADC 실행 기준 시각을 저장한다.
    MeasurementStage2_PrepareNext(measurement_last_sample_ms);         // 첫 센서의 안정화 시간을 시작한다.
    measurement_initialized = true;                                    // 주기 실행을 허용한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK;  // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;                 // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 10 ms마다 ADC를 읽고 표시된 센서의 실제 채널을 자동 기록한다. */
void MeasurementStage2_Process(void)
{
    const uint32_t now_ms = HAL_GetTick();  // 현재 실측 시각을 저장한다.
    uint32_t device;                        // 디버거에 복사할 MCP3008 번호를 저장한다.
    uint32_t channel;                       // 디버거에 복사할 ADC 채널 번호를 저장한다.

    if (!measurement_initialized || g_measurement_debug.adc_mapping_complete ||
        ((now_ms - measurement_last_sample_ms) < MEASUREMENT_STAGE2_SAMPLE_PERIOD_MS))
    {
        return;
    }

    measurement_last_sample_ms = now_ms;  // 이번 ADC 실행 시각을 저장한다.
    if (MCP3008_ReadAll(&measurement_adc, &measurement_adc_data) != HAL_OK)
    {
        g_measurement_debug.adc_driver_error_count = measurement_adc_data.error_count;       // ADC 오류 수를 갱신한다.
        g_measurement_debug.adc_last_error_device = measurement_adc_data.last_error_device;  // 실패 장치를 갱신한다.
        g_measurement_debug.adc_last_error_channel = measurement_adc_data.last_error_channel;// 실패 채널을 갱신한다.
        return;
    }

    g_measurement_debug.last_sample_ms = measurement_adc_data.mcu_time_ms;          // 최근 ADC 시각을 갱신한다.
    g_measurement_debug.adc_update_count = measurement_adc_data.update_counter;     // 정상 ADC 횟수를 갱신한다.
    g_measurement_debug.adc_driver_error_count = measurement_adc_data.error_count;  // ADC 오류 수를 갱신한다.
    g_measurement_debug.adc_last_error_device = measurement_adc_data.last_error_device; // 오류 장치를 갱신한다.
    g_measurement_debug.adc_last_error_channel = measurement_adc_data.last_error_channel; // 오류 채널을 갱신한다.
    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            g_measurement_debug.adc_raw[device][channel] =
                measurement_adc_data.raw[device][channel];  // 최근 ADC 24채널을 표시한다.
        }
    }

    if (!measurement_range_ready)
    {
        if ((int32_t)(now_ms - measurement_settle_until_ms) >= 0)
        {
            MeasurementStage2_ResetRange();  // 안정된 현재값부터 대상 움직임을 감지한다.
        }
        return;
    }

    MeasurementStage2_UpdateRange();         // 미사용 채널의 변화 폭을 갱신한다.
    (void)MeasurementStage2_TryCapture(now_ms); // 확실한 단일 후보를 자동 기록한다.
}
