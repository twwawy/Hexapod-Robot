#include "measurement/measurement_stage7.h"

#include "common/robot_calibration.h"
#include "low_control/relay.h"
#include "measurement/crsf_calibration_measurement.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "user_command/crsf_protocol.h"
#include "user_command/crsf_receiver.h"

#include <stddef.h>
#include <string.h>

#define MEASUREMENT_STAGE7_REQUIRED_CONNECTION_FRAMES  20U
#define MEASUREMENT_STAGE7_WAIT_CENTER_MS           10000U
#define MEASUREMENT_STAGE7_CAPTURE_CENTER_MS         3000U
#define MEASUREMENT_STAGE7_WAIT_TARGET_MS           10000U
#define MEASUREMENT_STAGE7_CAPTURE_TARGET_MS         3000U
#define MEASUREMENT_STAGE7_FRAME_TIMEOUT_MS           500U
#define MEASUREMENT_STAGE7_MINIMUM_SPAN_RAW            100U

typedef struct
{
    uint8_t channel;   // 조작할 채널 배열 번호를 저장한다.
    int8_t position;   // 짐벌 방향 또는 스위치 논리 위치를 저장한다.
} MeasurementStage7_Target_t;

static const MeasurementStage7_Target_t measurement_targets[] =
{
    {0U, -1}, {0U,  1},  // CH1 Roll의 왼쪽과 오른쪽을 측정한다.
    {1U, -1}, {1U,  1},  // CH2 Pitch의 뒤쪽과 앞쪽을 측정한다.
    {2U, -1}, {2U,  1},  // CH3 Throttle의 아래쪽과 위쪽을 측정한다.
    {3U, -1}, {3U,  1},  // CH4 Yaw의 왼쪽과 오른쪽을 측정한다.
    {4U,  0}, {4U,  1},  // CH5 SA의 OFF와 ON을 측정한다.
    {5U,  0}, {5U,  1}, {5U,  2},  // CH6 SB의 위·가운데·아래를 측정한다.
    {6U,  0}, {6U,  1}, {6U,  2},  // CH7 SC의 위·가운데·아래를 측정한다.
    {7U,  0}, {7U,  1},  // CH8 SD의 OFF와 ON을 측정한다.
    {8U,  0}, {8U,  1},  // CH9 SE의 해제와 누름을 측정한다.
    {9U,  0}, {9U,  1}, {9U,  2}   // CH10 S1의 왼쪽·중앙·오른쪽을 측정한다.
};

#define MEASUREMENT_STAGE7_TARGET_COUNT \
    ((uint8_t)(sizeof(measurement_targets) / sizeof(measurement_targets[0])))

static CRSF_Receiver_t measurement_receiver;                         // USART6 CRSF 수신 상태를 저장한다.
static CRSF_Protocol_t measurement_protocol;                         // CRSF 프레임 해석 상태를 저장한다.
static CrsfCalibrationMeasurement_t measurement_crsf;                // CH1~CH10 보정용 raw를 저장한다.
static UserCommand_ChannelCalibration_t measurement_table[USER_COMMAND_USED_CHANNELS]; // 완성한 채널표를 저장한다.
static MeasurementRunner_t measurement_runner;                       // 현재 실측 단계를 저장한다.
static uint8_t measurement_switch_map[USER_COMMAND_USED_CHANNELS][3]; // raw 위치별 논리 스위치값을 저장한다.
static int8_t measurement_direction[USER_COMMAND_USED_CHANNELS];      // 채널별 양의 방향을 저장한다.
static uint32_t measurement_center_sum[4];                            // 네 짐벌 중립 raw 합계를 저장한다.
static uint32_t measurement_target_sum;                               // 현재 위치 raw 합계를 저장한다.
static uint32_t measurement_phase_start_ms;                           // 현재 단계 시작 시각을 저장한다.
static uint32_t measurement_last_frame_ms;                            // 마지막 정상 RC 프레임 시각을 저장한다.
static uint16_t measurement_center_count;                             // 네 짐벌 중립 표본 수를 저장한다.
static uint16_t measurement_target_count;                             // 현재 위치 표본 수를 저장한다.
static uint8_t measurement_target_index;                              // 현재 측정 지시 번호를 저장한다.
static MeasurementStage7_Phase_t measurement_phase;                  // 현재 CRSF 측정 단계를 저장한다.
static bool measurement_initialized;                                 // 7단계 실행 가능 여부를 저장한다.

/* 새 측정 단계와 시작 시각을 저장한다. */
static void MeasurementStage7_SetPhase(MeasurementStage7_Phase_t phase,
                                       uint32_t now_ms)
{
    measurement_phase = phase;                                 // 새 측정 단계를 저장한다.
    measurement_phase_start_ms = now_ms;                        // 새 단계 시작 시각을 저장한다.
    g_measurement_debug.crsf_measurement_phase = (uint8_t)phase; // Live Expressions 단계를 갱신한다.
    g_measurement_debug.crsf_phase_elapsed_seconds = 0U;         // 단계 경과 시간을 초기화한다.
}

/* 현재 조작할 채널과 위치를 디버거에 표시한다. */
static void MeasurementStage7_UpdateTarget(void)
{
    const MeasurementStage7_Target_t *target =
        &measurement_targets[measurement_target_index];  // 현재 측정 지시를 선택한다.

    g_measurement_debug.crsf_target_channel =
        (uint8_t)(target->channel + 1U);                  // 채널 번호를 1~10으로 표시한다.
    g_measurement_debug.crsf_target_position = target->position; // 목표 방향 또는 위치를 표시한다.
    g_measurement_debug.crsf_target_sample_count = 0U;    // 이전 위치의 표본 수를 제거한다.
    g_measurement_debug.crsf_target_average_raw = 0U;     // 이전 위치의 평균값을 제거한다.
}

/* 스위치의 측정 raw 순서로 논리값 대응표를 만든다. */
static bool MeasurementStage7_BuildSwitchMap(uint8_t channel,
                                              uint8_t position_count)
{
    uint8_t first;   // 비교할 첫 논리 위치를 저장한다.
    uint8_t second;  // 비교할 둘째 논리 위치를 저장한다.

    measurement_switch_map[channel][0] = 0U;  // 기본 Low 논리값을 준비한다.
    measurement_switch_map[channel][1] = 1U;  // 기본 Mid 논리값을 준비한다.
    measurement_switch_map[channel][2] = 2U;  // 기본 High 논리값을 준비한다.

    if (position_count == 2U)
    {
        const uint16_t logical_zero = g_measurement_debug.crsf_position_raw[channel][0]; // 논리 0 raw를 읽는다.
        const uint16_t logical_one = g_measurement_debug.crsf_position_raw[channel][1];  // 논리 1 raw를 읽는다.

        if (logical_zero == logical_one)
        {
            return false;
        }
        if (logical_zero < logical_one)
        {
            measurement_switch_map[channel][0] = 0U;  // 낮은 raw를 논리 0으로 둔다.
            measurement_switch_map[channel][2] = 1U;  // 높은 raw를 논리 1로 둔다.
        }
        else
        {
            measurement_switch_map[channel][0] = 1U;  // 낮은 raw를 논리 1로 둔다.
            measurement_switch_map[channel][2] = 0U;  // 높은 raw를 논리 0으로 둔다.
        }
        return true;
    }

    for (first = 0U; first < 3U; ++first)
    {
        uint8_t rank = 0U;  // 현재 논리 위치의 raw 순위를 저장한다.

        for (second = 0U; second < 3U; ++second)
        {
            if (g_measurement_debug.crsf_position_raw[channel][second] <
                g_measurement_debug.crsf_position_raw[channel][first])
            {
                ++rank;  // 더 작은 raw 개수로 정렬 위치를 계산한다.
            }
        }
        measurement_switch_map[channel][rank] = first;  // raw 순위에 사용자가 지정한 논리값을 넣는다.
    }

    return (g_measurement_debug.crsf_position_raw[channel][0] !=
            g_measurement_debug.crsf_position_raw[channel][1]) &&
           (g_measurement_debug.crsf_position_raw[channel][1] !=
            g_measurement_debug.crsf_position_raw[channel][2]) &&
           (g_measurement_debug.crsf_position_raw[channel][0] !=
            g_measurement_debug.crsf_position_raw[channel][2]);  // 세 위치가 모두 다른지 확인한다.
}

/* 기록한 모든 위치로 CH1~CH10 보정표를 만든다. */
static bool MeasurementStage7_BuildTable(void)
{
    uint32_t channel;  // 보정표를 만들 채널 번호를 저장한다.

    measurement_crsf.center_captured = true;  // 네 짐벌 중립 측정 완료를 표시한다.
    g_measurement_debug.crsf_center_captured = true;  // 디버거에도 중립 완료를 표시한다.

    for (channel = 0U; channel < 4U; ++channel)
    {
        const uint16_t negative = g_measurement_debug.crsf_position_raw[channel][0]; // 음의 끝점 raw를 읽는다.
        const uint16_t positive = g_measurement_debug.crsf_position_raw[channel][2]; // 양의 끝점 raw를 읽는다.
        const uint16_t raw_min = (negative < positive) ? negative : positive;         // 숫자가 작은 끝점을 선택한다.
        const uint16_t raw_max = (negative > positive) ? negative : positive;         // 숫자가 큰 끝점을 선택한다.

        if (((uint16_t)(raw_max - raw_min) < MEASUREMENT_STAGE7_MINIMUM_SPAN_RAW) ||
            (measurement_crsf.center[channel] <= raw_min) ||
            (measurement_crsf.center[channel] >= raw_max))
        {
            g_measurement_debug.crsf_measurement_error_channel =
                (uint8_t)(channel + 1U);  // 잘못된 짐벌 채널을 표시한다.
            return false;
        }

        measurement_crsf.minimum[channel] = raw_min;                  // 짐벌 최소 raw를 저장한다.
        measurement_crsf.maximum[channel] = raw_max;                  // 짐벌 최대 raw를 저장한다.
        measurement_direction[channel] = (positive > negative) ? 1 : -1; // 사용자가 지정한 양의 방향을 저장한다.
        g_measurement_debug.crsf_minimum[channel] = raw_min;           // 최종 최소값을 표시한다.
        g_measurement_debug.crsf_maximum[channel] = raw_max;           // 최종 최대값을 표시한다.
        g_measurement_debug.crsf_calibration_direction[channel] =
            measurement_direction[channel];                           // 최종 방향을 표시한다.
    }

    for (channel = 4U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        const uint8_t position_count =
            ((channel == 5U) || (channel == 6U) || (channel == 9U)) ? 3U : 2U; // SB·SC·S1만 세 위치로 처리한다.
        uint16_t raw_min = UINT16_MAX;  // 현재 스위치 최소 raw를 준비한다.
        uint16_t raw_max = 0U;          // 현재 스위치 최대 raw를 준비한다.
        uint8_t position;               // 검사할 논리 위치를 저장한다.

        for (position = 0U; position < position_count; ++position)
        {
            const uint16_t raw = g_measurement_debug.crsf_position_raw[channel][position]; // 현재 위치 raw를 읽는다.

            if (raw < raw_min)
            {
                raw_min = raw;  // 새 최소값을 저장한다.
            }
            if (raw > raw_max)
            {
                raw_max = raw;  // 새 최대값을 저장한다.
            }
        }

        if (((uint16_t)(raw_max - raw_min) < MEASUREMENT_STAGE7_MINIMUM_SPAN_RAW) ||
            !MeasurementStage7_BuildSwitchMap((uint8_t)channel, position_count))
        {
            g_measurement_debug.crsf_measurement_error_channel =
                (uint8_t)(channel + 1U);  // 잘못된 스위치 채널을 표시한다.
            return false;
        }

        measurement_crsf.minimum[channel] = raw_min;                      // 스위치 최소 raw를 저장한다.
        measurement_crsf.maximum[channel] = raw_max;                      // 스위치 최대 raw를 저장한다.
        measurement_direction[channel] = 1;                               // 스위치 방향은 대응표에서 처리한다.
        g_measurement_debug.crsf_minimum[channel] = raw_min;               // 최종 최소값을 표시한다.
        g_measurement_debug.crsf_center[channel] =
            (uint16_t)(((uint32_t)raw_min + raw_max) / 2U);                // 스위치 경계 중심을 표시한다.
        g_measurement_debug.crsf_maximum[channel] = raw_max;               // 최종 최대값을 표시한다.
        g_measurement_debug.crsf_calibration_direction[channel] = 1;       // 스위치 방향 고정값을 표시한다.
    }

    return CrsfCalibrationMeasurement_Build(&measurement_crsf,
                                            measurement_direction,
                                            measurement_switch_map,
                                            measurement_table);  // 최종 CRSF 보정표를 생성한다.
}

/* 현재 정상 프레임의 16채널 raw와 통신 상태를 디버거에 복사한다. */
static void MeasurementStage7_UpdateReceiverDebug(uint32_t now_ms)
{
    uint32_t channel;  // 복사할 CRSF 채널 번호를 저장한다.

    for (channel = 0U; channel < ROBOT_CRSF_CHANNEL_COUNT; ++channel)
    {
        g_measurement_debug.crsf_current_raw[channel] =
            measurement_protocol.channel[channel];  // 최근 16채널 raw를 표시한다.
    }

    measurement_last_frame_ms = now_ms;                               // 마지막 정상 프레임 시각을 저장한다.
    g_measurement_debug.crsf_last_frame_ms = now_ms;                   // 마지막 정상 프레임 시각을 표시한다.
    g_measurement_debug.crsf_frame_count = measurement_protocol.frame_counter; // 정상 프레임 수를 표시한다.
    g_measurement_debug.crsf_crc_error_count = measurement_protocol.crc_error_count; // CRC 오류 수를 표시한다.
    g_measurement_debug.crsf_length_error_count = measurement_protocol.length_error_count; // 길이 오류 수를 표시한다.
    g_measurement_debug.crsf_rx_overflow_count = measurement_receiver.overflow_count; // 버퍼 초과 수를 표시한다.
    g_measurement_debug.crsf_uart_error_count = measurement_receiver.uart_error_count; // UART 오류 수를 표시한다.
    g_measurement_debug.crsf_connected = true;                         // 정상 연결 상태를 표시한다.
}

/* 중립 또는 현재 목표 위치의 새 프레임을 평균에 추가한다. */
static void MeasurementStage7_AddSample(void)
{
    if (measurement_phase == MEASUREMENT_STAGE7_CAPTURE_CENTER)
    {
        uint32_t channel;  // 중립 표본을 추가할 짐벌 채널을 저장한다.

        if (measurement_center_count == UINT16_MAX)
        {
            return;
        }
        for (channel = 0U; channel < 4U; ++channel)
        {
            measurement_center_sum[channel] += measurement_protocol.channel[channel]; // 짐벌 중립 raw를 누적한다.
        }
        ++measurement_center_count;  // 중립 표본 수를 증가시킨다.
    }
    else if (measurement_phase == MEASUREMENT_STAGE7_CAPTURE_TARGET)
    {
        const uint8_t channel = measurement_targets[measurement_target_index].channel; // 현재 대상 채널을 선택한다.

        if (measurement_target_count == UINT16_MAX)
        {
            return;
        }
        measurement_target_sum += measurement_protocol.channel[channel];  // 현재 위치 raw를 누적한다.
        ++measurement_target_count;                                       // 현재 위치 표본 수를 증가시킨다.
        g_measurement_debug.crsf_target_sample_count = measurement_target_count; // 표본 수를 표시한다.
        g_measurement_debug.crsf_target_average_raw =
            (uint16_t)(measurement_target_sum / measurement_target_count); // 진행 중 평균값을 표시한다.
    }
}

/* 네 짐벌 중립 평균을 보정 상태에 기록한다. */
static bool MeasurementStage7_CaptureCenter(void)
{
    uint32_t channel;  // 기록할 짐벌 채널 번호를 저장한다.

    if (measurement_center_count == 0U)
    {
        return false;
    }

    for (channel = 0U; channel < 4U; ++channel)
    {
        measurement_crsf.center[channel] =
            (uint16_t)(measurement_center_sum[channel] / measurement_center_count); // 중립 평균을 저장한다.
        g_measurement_debug.crsf_center[channel] = measurement_crsf.center[channel]; // 중립 평균을 표시한다.
    }
    measurement_crsf.center_captured = true;              // 중립 측정 완료를 기록한다.
    g_measurement_debug.crsf_center_captured = true;       // 중립 측정 완료를 표시한다.
    return true;
}

/* 현재 목표 위치의 평균 raw를 위치표에 기록한다. */
static bool MeasurementStage7_CaptureTarget(void)
{
    const MeasurementStage7_Target_t *target =
        &measurement_targets[measurement_target_index];  // 현재 측정 지시를 선택한다.
    uint8_t position_index;                              // 위치표 열 번호를 저장한다.
    uint16_t average;                                    // 현재 위치 평균 raw를 저장한다.

    if (measurement_target_count == 0U)
    {
        return false;
    }

    average = (uint16_t)(measurement_target_sum / measurement_target_count); // 현재 위치 평균을 계산한다.
    position_index = (target->position < 0) ? 0U :
                     (target->position > 1) ? 2U :
                     (uint8_t)target->position;  // -1/0은 첫 열, 1은 둘째 또는 양의 축 끝, 2는 셋째 열로 배치한다.
    if ((target->channel < 4U) && (target->position > 0))
    {
        position_index = 2U;  // 짐벌 양의 끝점을 셋째 열에 저장한다.
    }

    g_measurement_debug.crsf_position_raw[target->channel][position_index] = average; // 위치 평균을 기록한다.
    g_measurement_debug.crsf_target_average_raw = average;                           // 완료 평균을 표시한다.
    return true;
}

/* USART6 수신기와 CH1~CH10 자동 실측을 준비한다. */
bool MeasurementStage7_Init(UART_HandleTypeDef *crsf_uart)
{
    uint32_t skip;     // 건너뛸 완료 단계 수를 저장한다.
    uint32_t channel;  // 초기화할 채널 번호를 저장한다.

    MeasurementRunner_Init(&measurement_runner);  // 전역 디버그값을 초기화한다.
    for (skip = 0U; skip < (uint32_t)MEASUREMENT_STAGE_CRSF; ++skip)
    {
        (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 0~6단계를 건너뛴다.
    }
    g_measurement_debug.calibration = g_robot_calibration;  // 앞 단계의 중앙 설정값을 유지한다.
    CrsfCalibrationMeasurement_Init(&measurement_crsf);     // CRSF 보정용 상태를 준비한다.
    CRSF_Receiver_Init(&measurement_receiver, crsf_uart);    // USART6 링 버퍼를 준비한다.
    CRSF_Protocol_Init(&measurement_protocol);               // CRSF 프레임 해석기를 준비한다.
    memset(measurement_table, 0, sizeof(measurement_table)); // 이전 최종 채널표를 제거한다.
    memset(measurement_switch_map, 0, sizeof(measurement_switch_map)); // 이전 스위치 대응을 제거한다.
    memset(measurement_direction, 0, sizeof(measurement_direction));   // 이전 방향 결과를 제거한다.
    memset(measurement_center_sum, 0, sizeof(measurement_center_sum)); // 이전 중립 합계를 제거한다.
    Relay_Init();                                             // 릴레이 출력을 안전하게 준비한다.
    Relay_AllOff();                                           // CRSF 실측 중 서보 전원을 차단한다.
    g_measurement_debug.relay_state_mask = 0U;                 // 전체 릴레이 OFF를 표시한다.
    g_measurement_debug.crsf_measurement_complete = false;     // 미완료 상태로 시작한다.
    g_measurement_debug.crsf_measurement_failed = false;       // 정상 상태로 시작한다.
    g_measurement_debug.crsf_measurement_error_channel = 0U;   // 오류 채널 표시를 제거한다.
    g_measurement_debug.crsf_connected = false;                // 첫 정상 프레임 전 연결을 해제한다.
    g_measurement_debug.crsf_target_channel = 0U;              // 연결 대기 중 대상 채널을 제거한다.
    g_measurement_debug.crsf_target_position = 0;              // 연결 대기 중 목표 위치를 제거한다.
    measurement_center_count = 0U;                             // 중립 표본 수를 초기화한다.
    measurement_target_count = 0U;                             // 목표 위치 표본 수를 초기화한다.
    measurement_target_sum = 0U;                               // 목표 위치 합계를 초기화한다.
    measurement_target_index = 0U;                             // 첫 Roll 음의 방향부터 준비한다.
    measurement_last_frame_ms = HAL_GetTick();                 // 연결 Timeout 기준을 준비한다.
    measurement_initialized = false;                           // 수신 시작 전 실행을 막는다.

    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        g_measurement_debug.crsf_position_raw[channel][0] = 0U;  // 첫 위치값을 초기화한다.
        g_measurement_debug.crsf_position_raw[channel][1] = 0U;  // 둘째 위치값을 초기화한다.
        g_measurement_debug.crsf_position_raw[channel][2] = 0U;  // 셋째 위치값을 초기화한다.
    }

    if (crsf_uart == NULL)
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 잘못된 USART6 Handle을 기록한다.
        return false;
    }
    if (CRSF_Receiver_Start(&measurement_receiver) != HAL_OK)
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // USART6 수신 시작 실패를 기록한다.
        return false;
    }

    MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_WAIT_CONNECTION,
                               HAL_GetTick());                    // 정상 프레임 연결부터 기다린다.
    measurement_initialized = true;                              // 주기 실행을 허용한다.
    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK; // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;           // 디버거에 준비 완료를 표시한다.
    return true;
}

/* CRSF 프레임을 처리하고 지시한 위치별 보정값을 자동 생성한다. */
void MeasurementStage7_Process(void)
{
    const uint32_t now_ms = HAL_GetTick();  // 현재 실측 시각을 저장한다.
    const uint32_t elapsed_ms = now_ms - measurement_phase_start_ms; // 현재 단계 경과 시간을 계산한다.
    const uint32_t frames = CRSF_Protocol_ProcessReceiver(&measurement_protocol,
                                                          &measurement_receiver); // 대기 중인 CRSF 프레임을 처리한다.

    if (!measurement_initialized)
    {
        return;
    }

    if (frames > 0U)
    {
        MeasurementStage7_UpdateReceiverDebug(now_ms);  // 새 채널값과 통신 상태를 표시한다.
        MeasurementStage7_AddSample();                  // 측정 단계이면 새 표본을 누적한다.
    }
    else if ((now_ms - measurement_last_frame_ms) > MEASUREMENT_STAGE7_FRAME_TIMEOUT_MS)
    {
        g_measurement_debug.crsf_connected = false;  // 정상 프레임 Timeout을 표시한다.
    }

    g_measurement_debug.crsf_phase_elapsed_seconds =
        (uint8_t)(elapsed_ms / 1000U);  // 현재 단계 경과 초를 표시한다.

    if (measurement_phase == MEASUREMENT_STAGE7_WAIT_CONNECTION)
    {
        if (measurement_protocol.frame_counter >= MEASUREMENT_STAGE7_REQUIRED_CONNECTION_FRAMES)
        {
            MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_WAIT_CENTER, now_ms); // 네 짐벌 중립 준비를 시작한다.
        }
        return;
    }

    if (!g_measurement_debug.crsf_connected)
    {
        MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_FAILED, now_ms); // 측정 중 연결 끊김을 오류로 표시한다.
        g_measurement_debug.crsf_measurement_failed = true;             // CRSF 실측 실패를 표시한다.
        measurement_initialized = false;                                // 실패 결과를 유지한다.
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE7_WAIT_CENTER)
    {
        if (elapsed_ms >= MEASUREMENT_STAGE7_WAIT_CENTER_MS)
        {
            memset(measurement_center_sum, 0, sizeof(measurement_center_sum)); // 중립 평균 합계를 초기화한다.
            measurement_center_count = 0U;                                // 중립 표본 수를 초기화한다.
            MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_CAPTURE_CENTER,
                                       now_ms);                            // 네 짐벌 중립 측정을 시작한다.
        }
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE7_CAPTURE_CENTER)
    {
        if (elapsed_ms < MEASUREMENT_STAGE7_CAPTURE_CENTER_MS)
        {
            return;
        }
        if (!MeasurementStage7_CaptureCenter())
        {
            MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_FAILED, now_ms); // 중립 표본 부족을 표시한다.
            g_measurement_debug.crsf_measurement_failed = true;             // CRSF 실측 실패를 표시한다.
            measurement_initialized = false;                                // 실패 결과를 유지한다.
            return;
        }

        MeasurementStage7_UpdateTarget();                              // 첫 Roll 음의 방향을 표시한다.
        MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_WAIT_TARGET,
                                   now_ms);                            // 첫 입력 준비 시간을 시작한다.
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE7_WAIT_TARGET)
    {
        if (elapsed_ms >= MEASUREMENT_STAGE7_WAIT_TARGET_MS)
        {
            measurement_target_sum = 0U;                              // 현재 위치 합계를 초기화한다.
            measurement_target_count = 0U;                            // 현재 위치 표본 수를 초기화한다.
            MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_CAPTURE_TARGET,
                                       now_ms);                        // 현재 위치 측정을 시작한다.
        }
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE7_CAPTURE_TARGET)
    {
        if (elapsed_ms < MEASUREMENT_STAGE7_CAPTURE_TARGET_MS)
        {
            return;
        }
        if (!MeasurementStage7_CaptureTarget())
        {
            MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_FAILED, now_ms); // 현재 위치 표본 부족을 표시한다.
            g_measurement_debug.crsf_measurement_failed = true;             // CRSF 실측 실패를 표시한다.
            measurement_initialized = false;                                // 실패 결과를 유지한다.
            return;
        }

        ++measurement_target_index;  // 다음 지시 위치로 이동한다.
        if (measurement_target_index < MEASUREMENT_STAGE7_TARGET_COUNT)
        {
            MeasurementStage7_UpdateTarget();                              // 다음 채널과 위치를 표시한다.
            MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_WAIT_TARGET,
                                       now_ms);                            // 다음 위치 준비 시간을 시작한다.
            return;
        }

        if (!MeasurementStage7_BuildTable())
        {
            MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_FAILED, now_ms); // 잘못된 채널 결과를 표시한다.
            g_measurement_debug.crsf_measurement_failed = true;             // CRSF 보정 실패를 표시한다.
            measurement_initialized = false;                                // 실패 결과를 유지한다.
            return;
        }

        g_measurement_debug.crsf_measurement_complete = true;  // CH1~CH10 보정 완료를 표시한다.
        MeasurementStage7_SetPhase(MEASUREMENT_STAGE7_COMPLETE,
                                   now_ms);                      // 완료 상태를 유지한다.
        (void)MeasurementRunner_CompleteCurrent(&measurement_runner); // 전체 실측 완료로 이동한다.
        measurement_initialized = false;                        // 완료 결과를 고정한다.
    }
}

/* USART6 수신 완료를 CRSF 링 버퍼에 전달한다. */
void MeasurementStage7_UartRxCallback(UART_HandleTypeDef *uart)
{
    CRSF_Receiver_RxCpltCallback(&measurement_receiver, uart);  // 해당 USART6이면 다음 수신을 건다.
}

/* USART6 수신 오류를 CRSF 드라이버에 전달한다. */
void MeasurementStage7_UartErrorCallback(UART_HandleTypeDef *uart)
{
    CRSF_Receiver_ErrorCallback(&measurement_receiver, uart);  // 해당 USART6이면 수신을 복구한다.
}
