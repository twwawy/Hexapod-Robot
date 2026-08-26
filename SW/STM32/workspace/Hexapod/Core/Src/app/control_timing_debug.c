#include "app/control_timing_debug.h"

#include "stm32f4xx_hal.h"

#include <stddef.h>
#include <string.h>

volatile ControlTimingDebug_t g_control_timing_debug;  // CubeIDE에서 확인할 시간 통계를 저장한다.

/* CPU Cycle 차이를 us로 변환한다. */
static uint32_t ControlTimingDebug_CyclesToUs(uint32_t cycles)
{
    const uint32_t cycles_per_us = g_control_timing_debug.cpu_hz / 1000000U;  // 1 us의 Cycle 수를 계산한다.

    return (cycles_per_us > 0U) ? (cycles / cycles_per_us) : 0U;  // 정수 us로 변환한다.
}

/* 기존 통계를 지우고 측정 설정을 유지한다. */
static void ControlTimingDebug_Reset(void)
{
    const bool enabled = g_control_timing_debug.enabled;  // DWT 활성 상태를 보존한다.
    const uint32_t cpu_hz = g_control_timing_debug.cpu_hz;  // CPU Clock을 보존한다.

    memset((void *)&g_control_timing_debug, 0, sizeof(g_control_timing_debug));  // 이전 통계를 제거한다.
    g_control_timing_debug.enabled = enabled;                                    // DWT 활성 상태를 복원한다.
    g_control_timing_debug.cpu_hz = cpu_hz;                                      // CPU Clock을 복원한다.
    g_control_timing_debug.expected_interval_us = CONTROL_TIMING_EXPECTED_US;     // 목표 제어 주기를 복원한다.
}

/* 이상 순간을 순환 버퍼에 저장한다. */
static void ControlTimingDebug_SaveEvent(uint8_t reason_flags,
                                         uint32_t timestamp_ms,
                                         uint32_t control_count,
                                         uint32_t missed_control_count,
                                         RobotControlMode_t active_mode,
                                         const RobotUserCommand_t *user,
                                         const RobotDroneOutput_t *drone,
                                         uint32_t gait_phase_index,
                                         float gait_phase_time_s)
{
    const uint32_t index = g_control_timing_debug.event_write_index;  // 이번 기록 위치를 선택한다.
    volatile ControlTimingEvent_t *event = &g_control_timing_debug.event[index];  // 기록 항목을 연결한다.

    event->timestamp_ms = timestamp_ms;                                      // 이상 발생 시각을 기록한다.
    event->control_count = control_count;                                    // 제어 횟수를 기록한다.
    event->missed_control_count = missed_control_count;                      // 누락 요청 수를 기록한다.
    event->interval_us = g_control_timing_debug.interval_us;                 // 실제 제어 간격을 기록한다.
    event->execution_us = g_control_timing_debug.execution_us;               // 전체 제어 시간을 기록한다.
    event->sensor_us = g_control_timing_debug.sensor_us;                     // 센서 시간을 기록한다.
    event->algorithm_us = g_control_timing_debug.algorithm_us;               // 알고리즘 시간을 기록한다.
    event->output_us = g_control_timing_debug.output_us;                     // 출력 시간을 기록한다.
    event->process_us = g_control_timing_debug.process_us;                   // 통신 시간을 기록한다.
    event->crsf_age_ms = g_control_timing_debug.crsf_age_ms;                 // 조종기 명령 나이를 기록한다.
    event->crsf_overflow_count = g_control_timing_debug.crsf_overflow_count; // 버퍼 초과 수를 기록한다.
    event->crsf_buffer_used = g_control_timing_debug.crsf_buffer_used;       // 버퍼 사용량을 기록한다.
    event->reason_flags = reason_flags;                                      // 기록 원인을 저장한다.
    event->active_mode = (uint8_t)active_mode;                               // 운용 모드를 저장한다.
    event->throttle = (user != NULL) ? user->throttle : 0;                   // 전후 입력을 저장한다.
    event->yaw = (user != NULL) ? user->yaw : 0;                             // 회전 입력을 저장한다.
    event->vx_user_mps = (drone != NULL) ? drone->vx_user_mps : 0.0f;        // 사용자 X속도를 저장한다.
    event->vy_user_mps = (drone != NULL) ? drone->vy_user_mps : 0.0f;        // 사용자 Y속도를 저장한다.
    event->wz_user_radps = (drone != NULL) ? drone->wz_user_radps : 0.0f;    // 사용자 회전속도를 저장한다.
    event->vx_candidate_mps = g_control_timing_debug.vx_candidate_mps;       // PI 이후 X속도를 저장한다.
    event->vy_candidate_mps = g_control_timing_debug.vy_candidate_mps;       // PI 이후 Y속도를 저장한다.
    event->wz_candidate_radps = g_control_timing_debug.wz_candidate_radps;   // PI 이후 회전속도를 저장한다.
    event->vx_accepted_mps = g_control_timing_debug.vx_accepted_mps;         // 적용 X속도를 저장한다.
    event->vy_accepted_mps = g_control_timing_debug.vy_accepted_mps;         // 적용 Y속도를 저장한다.
    event->wz_accepted_radps = g_control_timing_debug.wz_accepted_radps;     // 적용 회전속도를 저장한다.
    event->gait_phase_index = gait_phase_index;                              // Tripod 위상을 저장한다.
    event->gait_phase_time_s = gait_phase_time_s;                            // 위상 시간을 저장한다.

    g_control_timing_debug.latest_event_index = index;                       // 최근 기록 위치를 공개한다.
    g_control_timing_debug.latest_reason_flags = reason_flags;               // 최근 기록 원인을 공개한다.
    g_control_timing_debug.event_write_index =
        (index + 1U) % CONTROL_TIMING_EVENT_COUNT;                            // 다음 순환 위치를 선택한다.
    if (g_control_timing_debug.event_count < CONTROL_TIMING_EVENT_COUNT)
    {
        g_control_timing_debug.event_count++;                                // 유효 기록 수를 증가시킨다.
    }
}

/* DWT Cycle Counter와 통계를 초기화한다. */
void ControlTimingDebug_Init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // DWT Trace 기능을 활성화한다.
    DWT->CYCCNT = 0U;                                // Cycle Counter를 초기화한다.
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;             // CPU Cycle 측정을 시작한다.

    memset((void *)&g_control_timing_debug, 0, sizeof(g_control_timing_debug));  // 시간 통계를 초기화한다.
    g_control_timing_debug.enabled =
        ((DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) != 0U);                           // 실제 DWT 활성 여부를 저장한다.
    g_control_timing_debug.cpu_hz = SystemCoreClock;                            // 현재 CPU Clock을 저장한다.
    g_control_timing_debug.expected_interval_us = CONTROL_TIMING_EXPECTED_US;   // 5 ms 목표 주기를 저장한다.
}

/* 현재 CPU Cycle Counter를 읽는다. */
uint32_t ControlTimingDebug_ReadCycle(void)
{
    return g_control_timing_debug.enabled ? DWT->CYCCNT : 0U;  // DWT가 켜졌을 때만 Cycle을 반환한다.
}

/* 통신 처리 시간과 CRSF 상태를 기록한다. */
void ControlTimingDebug_RecordProcess(uint32_t start_cycle,
                                      uint32_t end_cycle,
                                      uint32_t crsf_age_ms,
                                      uint16_t crsf_buffer_used,
                                      uint32_t crsf_overflow_count,
                                      uint32_t crsf_uart_error_count,
                                      uint32_t crsf_crc_error_count)
{
    const uint32_t process_us =
        ControlTimingDebug_CyclesToUs(end_cycle - start_cycle);  // 통신 처리 시간을 계산한다.

    g_control_timing_debug.process_us = process_us;                         // 최근 통신 시간을 저장한다.
    g_control_timing_debug.crsf_age_ms = crsf_age_ms;                       // 입력 프레임 나이를 저장한다.
    g_control_timing_debug.crsf_buffer_used = crsf_buffer_used;             // 버퍼 사용량을 저장한다.
    g_control_timing_debug.crsf_overflow_count = crsf_overflow_count;       // 버퍼 초과 수를 저장한다.
    g_control_timing_debug.crsf_uart_error_count = crsf_uart_error_count;   // UART 오류 수를 저장한다.
    g_control_timing_debug.crsf_crc_error_count = crsf_crc_error_count;     // CRC 오류 수를 저장한다.

    if (process_us > g_control_timing_debug.process_max_us)
    {
        g_control_timing_debug.process_max_us = process_us;  // 최대 통신 시간을 갱신한다.
    }
    if (process_us > CONTROL_TIMING_PROCESS_WARNING_US)
    {
        g_control_timing_debug.process_warning_count++;      // 긴 통신 처리를 기록한다.
    }
    if (crsf_buffer_used > g_control_timing_debug.crsf_buffer_max_used)
    {
        g_control_timing_debug.crsf_buffer_max_used = crsf_buffer_used;  // 최대 버퍼 사용량을 갱신한다.
    }
    if (crsf_age_ms > g_control_timing_debug.crsf_age_max_ms)
    {
        g_control_timing_debug.crsf_age_max_ms = crsf_age_ms;  // 최대 입력 지연을 갱신한다.
    }
}

/* 실제 제어 시작 간격을 기록한다. */
void ControlTimingDebug_BeginControl(uint32_t start_cycle)
{
    uint32_t interval_us;  // 실제 제어 시작 간격을 저장한다.

    if (g_control_timing_debug.reset_request)
    {
        ControlTimingDebug_Reset();  // 디버거의 통계 초기화 요청을 처리한다.
    }

    if (!g_control_timing_debug.interval_valid)
    {
        g_control_timing_debug.last_control_cycle = start_cycle;  // 첫 제어 시각만 저장한다.
        g_control_timing_debug.interval_valid = true;              // 다음 제어부터 간격을 측정한다.
        return;
    }

    interval_us = ControlTimingDebug_CyclesToUs(
        start_cycle - g_control_timing_debug.last_control_cycle);  // 실제 시작 간격을 계산한다.
    g_control_timing_debug.last_control_cycle = start_cycle;       // 이번 시작 시각을 보존한다.
    g_control_timing_debug.interval_us = interval_us;              // 최근 시작 간격을 저장한다.

    if ((g_control_timing_debug.interval_min_us == 0U) ||
        (interval_us < g_control_timing_debug.interval_min_us))
    {
        g_control_timing_debug.interval_min_us = interval_us;  // 최소 시작 간격을 갱신한다.
    }
    if (interval_us > g_control_timing_debug.interval_max_us)
    {
        g_control_timing_debug.interval_max_us = interval_us;  // 최대 시작 간격을 갱신한다.
    }
    if (interval_us < CONTROL_TIMING_SHORT_US)
    {
        g_control_timing_debug.interval_short_count++;  // 지나치게 짧은 시작 간격을 기록한다.
    }
    if (interval_us > CONTROL_TIMING_LONG_US)
    {
        g_control_timing_debug.interval_long_count++;   // 긴 시작 간격을 기록한다.
    }
    if (interval_us > CONTROL_TIMING_VERY_LONG_US)
    {
        g_control_timing_debug.interval_very_long_count++;  // 매우 긴 시작 간격을 기록한다.
    }
}

/* 센서·알고리즘·출력 구간 시간을 기록한다. */
void ControlTimingDebug_RecordBreakdown(uint32_t sensor_cycles,
                                        uint32_t algorithm_cycles,
                                        uint32_t output_cycles)
{
    g_control_timing_debug.sensor_us =
        ControlTimingDebug_CyclesToUs(sensor_cycles);       // 센서 구간 시간을 저장한다.
    g_control_timing_debug.algorithm_us =
        ControlTimingDebug_CyclesToUs(algorithm_cycles);    // 알고리즘 구간 시간을 저장한다.
    g_control_timing_debug.output_us =
        ControlTimingDebug_CyclesToUs(output_cycles);       // 출력 구간 시간을 저장한다.

    if (g_control_timing_debug.sensor_us > g_control_timing_debug.sensor_max_us)
    {
        g_control_timing_debug.sensor_max_us = g_control_timing_debug.sensor_us;  // 최대 센서 시간을 갱신한다.
    }
    if (g_control_timing_debug.algorithm_us > g_control_timing_debug.algorithm_max_us)
    {
        g_control_timing_debug.algorithm_max_us = g_control_timing_debug.algorithm_us;  // 최대 알고리즘 시간을 갱신한다.
    }
    if (g_control_timing_debug.output_us > g_control_timing_debug.output_max_us)
    {
        g_control_timing_debug.output_max_us = g_control_timing_debug.output_us;  // 최대 출력 시간을 갱신한다.
    }
}

/* PI 후보와 작업공간 채택 속도를 기록한다. */
void ControlTimingDebug_RecordSignals(const RobotBodyTwist_t *candidate,
                                      const RobotBodyTwist_t *accepted)
{
    if ((candidate == NULL) || (accepted == NULL))
    {
        return;
    }

    g_control_timing_debug.vx_candidate_mps = candidate->vx;   // PI 이후 X속도를 저장한다.
    g_control_timing_debug.vy_candidate_mps = candidate->vy;   // PI 이후 Y속도를 저장한다.
    g_control_timing_debug.wz_candidate_radps = candidate->wz; // PI 이후 회전속도를 저장한다.
    g_control_timing_debug.vx_accepted_mps = accepted->vx;     // 작업공간 적용 X속도를 저장한다.
    g_control_timing_debug.vy_accepted_mps = accepted->vy;     // 작업공간 적용 Y속도를 저장한다.
    g_control_timing_debug.wz_accepted_radps = accepted->wz;   // 작업공간 적용 회전속도를 저장한다.
}

/* 전체 실행 시간과 이상 순간을 기록한다. */
void ControlTimingDebug_EndControl(uint32_t start_cycle,
                                   uint32_t end_cycle,
                                   uint32_t timestamp_ms,
                                   uint32_t control_count,
                                   uint32_t missed_control_count,
                                   RobotControlMode_t active_mode,
                                   const RobotUserCommand_t *user,
                                   const RobotDroneOutput_t *drone,
                                   uint32_t gait_phase_index,
                                   float gait_phase_time_s)
{
    uint8_t reason_flags = 0U;  // 이번 이상 기록 원인을 모은다.
    const bool interval_measured =
        (g_control_timing_debug.control_sample_count > 0U);  // 첫 제어 이후의 간격인지 확인한다.
    const uint32_t execution_us =
        ControlTimingDebug_CyclesToUs(end_cycle - start_cycle);  // 전체 제어 시간을 계산한다.

    g_control_timing_debug.execution_us = execution_us;                     // 최근 전체 시간을 저장한다.
    g_control_timing_debug.timer_missed_count = missed_control_count;       // 누락 요청 수를 저장한다.
    g_control_timing_debug.control_sample_count++;                          // 측정 제어 횟수를 증가시킨다.
    g_control_timing_debug.throttle = (user != NULL) ? user->throttle : 0;  // 전후 입력을 저장한다.
    g_control_timing_debug.yaw = (user != NULL) ? user->yaw : 0;            // 회전 입력을 저장한다.
    g_control_timing_debug.vx_user_mps =
        (drone != NULL) ? drone->vx_user_mps : 0.0f;                        // 사용자 X속도를 저장한다.
    g_control_timing_debug.vy_user_mps =
        (drone != NULL) ? drone->vy_user_mps : 0.0f;                        // 사용자 Y속도를 저장한다.
    g_control_timing_debug.wz_user_radps =
        (drone != NULL) ? drone->wz_user_radps : 0.0f;                      // 사용자 회전속도를 저장한다.

    if (execution_us > g_control_timing_debug.execution_max_us)
    {
        g_control_timing_debug.execution_max_us = execution_us;  // 최대 전체 시간을 갱신한다.
    }
    if (execution_us > CONTROL_TIMING_EXECUTION_WARNING_US)
    {
        g_control_timing_debug.execution_warning_count++;  // 마감에 가까운 실행을 기록한다.
        reason_flags |= CONTROL_TIMING_REASON_EXECUTION_NEAR;
    }
    if (execution_us >= CONTROL_TIMING_EXPECTED_US)
    {
        g_control_timing_debug.execution_overrun_count++;  // 5 ms 초과 실행을 기록한다.
        reason_flags |= CONTROL_TIMING_REASON_EXECUTION_OVER;
    }
    if (interval_measured &&
        (g_control_timing_debug.interval_us < CONTROL_TIMING_SHORT_US))
    {
        reason_flags |= CONTROL_TIMING_REASON_INTERVAL_SHORT;  // 짧은 시작 간격을 표시한다.
    }
    if (interval_measured &&
        (g_control_timing_debug.interval_us > CONTROL_TIMING_LONG_US))
    {
        reason_flags |= CONTROL_TIMING_REASON_INTERVAL_LONG;   // 긴 시작 간격을 표시한다.
    }
    if (g_control_timing_debug.process_us > CONTROL_TIMING_PROCESS_WARNING_US)
    {
        reason_flags |= CONTROL_TIMING_REASON_PROCESS_LONG;    // 긴 통신 처리를 표시한다.
    }
    if (missed_control_count != g_control_timing_debug.previous_missed_count)
    {
        reason_flags |= CONTROL_TIMING_REASON_TIMER_MISSED;    // 새 중복 요청을 표시한다.
    }
    if (g_control_timing_debug.crsf_overflow_count !=
        g_control_timing_debug.previous_overflow_count)
    {
        reason_flags |= CONTROL_TIMING_REASON_CRSF_OVERFLOW;   // 새 버퍼 초과를 표시한다.
    }
    if (g_control_timing_debug.capture_next)
    {
        reason_flags |= CONTROL_TIMING_REASON_MANUAL_CAPTURE;  // 디버거 강제 기록을 표시한다.
        g_control_timing_debug.capture_next = false;            // 한 번의 요청만 소비한다.
    }

    g_control_timing_debug.previous_missed_count = missed_control_count;  // 누락 비교 기준을 갱신한다.
    g_control_timing_debug.previous_overflow_count =
        g_control_timing_debug.crsf_overflow_count;                       // 버퍼 비교 기준을 갱신한다.

    if (reason_flags != 0U)
    {
        ControlTimingDebug_SaveEvent(reason_flags,
                                     timestamp_ms,
                                     control_count,
                                     missed_control_count,
                                     active_mode,
                                     user,
                                     drone,
                                     gait_phase_index,
                                     gait_phase_time_s);  // 이상 순간의 전체 상태를 보존한다.
    }
}

/* ISR에서 중복 Timer 요청 수만 짧게 기록한다. */
void ControlTimingDebug_RecordTimerMissed(uint32_t missed_control_count)
{
    g_control_timing_debug.timer_missed_count = missed_control_count;  // 최신 중복 요청 수를 저장한다.
}
