#ifndef CONTROL_TIMING_DEBUG_H
#define CONTROL_TIMING_DEBUG_H

#include "common/robot_types.h"

#include <stdbool.h>
#include <stdint.h>

#define CONTROL_TIMING_EVENT_COUNT          16U     // 최근 이상 16개를 보존한다.
#define CONTROL_TIMING_EXPECTED_US          5000U   // 목표 제어 주기를 정한다.
#define CONTROL_TIMING_SHORT_US             4000U   // 짧은 제어 간격 기준을 정한다.
#define CONTROL_TIMING_LONG_US              6000U   // 긴 제어 간격 기준을 정한다.
#define CONTROL_TIMING_VERY_LONG_US         10000U  // 매우 긴 제어 간격 기준을 정한다.
#define CONTROL_TIMING_EXECUTION_WARNING_US 4500U   // 마감 임박 기준을 정한다.
#define CONTROL_TIMING_PROCESS_WARNING_US   3000U   // 긴 통신 처리 기준을 정한다.

#define CONTROL_TIMING_REASON_INTERVAL_SHORT 0x01U  // 짧은 제어 간격을 표시한다.
#define CONTROL_TIMING_REASON_INTERVAL_LONG  0x02U  // 긴 제어 간격을 표시한다.
#define CONTROL_TIMING_REASON_EXECUTION_NEAR 0x04U  // 마감 임박 실행을 표시한다.
#define CONTROL_TIMING_REASON_EXECUTION_OVER 0x08U  // 제어 주기 초과를 표시한다.
#define CONTROL_TIMING_REASON_PROCESS_LONG   0x10U  // 긴 통신 처리를 표시한다.
#define CONTROL_TIMING_REASON_TIMER_MISSED   0x20U  // 중복 Timer 요청을 표시한다.
#define CONTROL_TIMING_REASON_CRSF_OVERFLOW  0x40U  // CRSF 버퍼 초과를 표시한다.
#define CONTROL_TIMING_REASON_MANUAL_CAPTURE 0x80U  // 강제 기록 요청을 표시한다.

typedef struct
{
    uint32_t timestamp_ms;            // 이상 동작 시각을 저장한다.
    uint32_t control_count;           // 완료 제어 횟수를 저장한다.
    uint32_t missed_control_count;    // 누락된 Timer 요청 수를 저장한다.
    uint32_t interval_us;             // 실제 제어 시작 간격을 저장한다.
    uint32_t execution_us;            // 전체 제어 실행 시간을 저장한다.
    uint32_t sensor_us;               // 센서 갱신 시간을 저장한다.
    uint32_t algorithm_us;            // 제어 알고리즘 시간을 저장한다.
    uint32_t output_us;               // IK와 출력 시간을 저장한다.
    uint32_t process_us;              // 직전 통신 처리 시간을 저장한다.
    uint32_t crsf_age_ms;             // 마지막 조종기 명령 나이를 저장한다.
    uint32_t crsf_overflow_count;     // CRSF 버퍼 초과 횟수를 저장한다.
    uint16_t crsf_buffer_used;        // CRSF 버퍼 사용량을 저장한다.
    uint8_t reason_flags;             // 기록 원인 비트를 저장한다.
    uint8_t active_mode;              // 현재 운용 모드를 저장한다.
    int16_t throttle;                 // 현재 전후 입력을 저장한다.
    int16_t yaw;                      // 현재 회전 입력을 저장한다.
    float vx_user_mps;                // 조종기 변환 X속도를 저장한다.
    float vy_user_mps;                // 조종기 변환 Y속도를 저장한다.
    float wz_user_radps;              // 조종기 변환 회전속도를 저장한다.
    float vx_candidate_mps;           // PI 이후 X속도를 저장한다.
    float vy_candidate_mps;           // PI 이후 Y속도를 저장한다.
    float wz_candidate_radps;         // PI 이후 회전속도를 저장한다.
    float vx_accepted_mps;            // 작업공간 적용 X속도를 저장한다.
    float vy_accepted_mps;            // 작업공간 적용 Y속도를 저장한다.
    float wz_accepted_radps;          // 작업공간 적용 회전속도를 저장한다.
    uint32_t gait_phase_index;        // 현재 Tripod 위상을 저장한다.
    float gait_phase_time_s;          // 현재 위상 경과시간을 저장한다.
} ControlTimingEvent_t;

typedef struct
{
    bool enabled;                     // DWT 측정 활성화를 저장한다.
    bool interval_valid;              // 제어 간격 측정 시작 여부를 저장한다.
    bool capture_next;                // 다음 제어 강제 기록 요청을 저장한다.
    bool reset_request;               // 통계 초기화 요청을 저장한다.
    uint32_t cpu_hz;                  // 현재 CPU Clock을 저장한다.
    uint32_t expected_interval_us;     // 목표 제어 주기를 저장한다.
    uint32_t last_control_cycle;       // 직전 제어 시작 Cycle을 저장한다.
    uint32_t interval_us;              // 최근 실제 제어 간격을 저장한다.
    uint32_t execution_us;             // 최근 전체 제어 시간을 저장한다.
    uint32_t sensor_us;                // 최근 센서 갱신 시간을 저장한다.
    uint32_t algorithm_us;             // 최근 알고리즘 시간을 저장한다.
    uint32_t output_us;                // 최근 IK와 출력 시간을 저장한다.
    uint32_t process_us;               // 최근 통신 처리 시간을 저장한다.
    uint32_t crsf_age_ms;              // 최근 조종기 명령 나이를 저장한다.
    uint16_t crsf_buffer_used;          // 현재 CRSF 버퍼 사용량을 저장한다.
    uint16_t crsf_buffer_max_used;      // 최대 CRSF 버퍼 사용량을 저장한다.
    uint32_t crsf_overflow_count;       // CRSF 버퍼 초과 횟수를 저장한다.
    uint32_t crsf_uart_error_count;     // CRSF UART 오류 횟수를 저장한다.
    uint32_t crsf_crc_error_count;      // CRSF CRC 오류 횟수를 저장한다.
    uint32_t interval_min_us;           // 최소 제어 간격을 저장한다.
    uint32_t interval_max_us;           // 최대 제어 간격을 저장한다.
    uint32_t execution_max_us;          // 최대 전체 제어 시간을 저장한다.
    uint32_t sensor_max_us;             // 최대 센서 시간을 저장한다.
    uint32_t algorithm_max_us;          // 최대 알고리즘 시간을 저장한다.
    uint32_t output_max_us;             // 최대 IK와 출력 시간을 저장한다.
    uint32_t process_max_us;            // 최대 통신 처리 시간을 저장한다.
    uint32_t crsf_age_max_ms;           // 최대 조종기 명령 나이를 저장한다.
    uint32_t control_sample_count;       // 측정한 제어 횟수를 저장한다.
    uint32_t interval_short_count;       // 4 ms 미만 제어 간격 수를 저장한다.
    uint32_t interval_long_count;        // 6 ms 초과 제어 간격 수를 저장한다.
    uint32_t interval_very_long_count;   // 10 ms 초과 제어 간격 수를 저장한다.
    uint32_t execution_warning_count;    // 4.5 ms 초과 실행 수를 저장한다.
    uint32_t execution_overrun_count;    // 5 ms 초과 실행 수를 저장한다.
    uint32_t process_warning_count;      // 3 ms 초과 통신 처리 수를 저장한다.
    uint32_t timer_missed_count;         // 중복 Timer 요청 수를 저장한다.
    uint32_t previous_missed_count;      // 직전 확인한 중복 요청 수를 저장한다.
    uint32_t previous_overflow_count;    // 직전 확인한 버퍼 초과 수를 저장한다.
    int16_t throttle;                    // 최근 전후 입력을 저장한다.
    int16_t yaw;                         // 최근 회전 입력을 저장한다.
    float vx_user_mps;                   // 최근 조종기 변환 X속도를 저장한다.
    float vy_user_mps;                   // 최근 조종기 변환 Y속도를 저장한다.
    float wz_user_radps;                 // 최근 조종기 변환 회전속도를 저장한다.
    float vx_candidate_mps;              // 최근 PI 이후 X속도를 저장한다.
    float vy_candidate_mps;              // 최근 PI 이후 Y속도를 저장한다.
    float wz_candidate_radps;            // 최근 PI 이후 회전속도를 저장한다.
    float vx_accepted_mps;               // 최근 작업공간 적용 X속도를 저장한다.
    float vy_accepted_mps;               // 최근 작업공간 적용 Y속도를 저장한다.
    float wz_accepted_radps;             // 최근 작업공간 적용 회전속도를 저장한다.
    uint32_t event_write_index;           // 다음 이상 기록 위치를 저장한다.
    uint32_t event_count;                 // 유효 이상 기록 수를 저장한다.
    uint32_t latest_event_index;          // 가장 최근 이상 기록 위치를 저장한다.
    uint8_t latest_reason_flags;          // 가장 최근 기록 원인을 저장한다.
    ControlTimingEvent_t event[CONTROL_TIMING_EVENT_COUNT];  // 최근 이상 제어를 순환 저장한다.
} ControlTimingDebug_t;

extern volatile ControlTimingDebug_t g_control_timing_debug;  // CubeIDE 확인용 시간 통계를 공개한다.

void ControlTimingDebug_Init(void);  // DWT Cycle Counter와 통계를 준비한다.

uint32_t ControlTimingDebug_ReadCycle(void);  // 현재 CPU Cycle을 읽는다.

void ControlTimingDebug_RecordProcess(uint32_t start_cycle,
                                      uint32_t end_cycle,
                                      uint32_t crsf_age_ms,
                                      uint16_t crsf_buffer_used,
                                      uint32_t crsf_overflow_count,
                                      uint32_t crsf_uart_error_count,
                                      uint32_t crsf_crc_error_count);  // 통신 처리와 CRSF 상태를 기록한다.

void ControlTimingDebug_BeginControl(uint32_t start_cycle);  // 실제 제어 시작 간격을 기록한다.

void ControlTimingDebug_RecordBreakdown(uint32_t sensor_cycles,
                                        uint32_t algorithm_cycles,
                                        uint32_t output_cycles);  // 제어 구간별 실행 시간을 기록한다.

void ControlTimingDebug_RecordSignals(const RobotBodyTwist_t *candidate,
                                      const RobotBodyTwist_t *accepted);  // PI 전후 속도 명령을 기록한다.

void ControlTimingDebug_EndControl(uint32_t start_cycle,
                                   uint32_t end_cycle,
                                   uint32_t timestamp_ms,
                                   uint32_t control_count,
                                   uint32_t missed_control_count,
                                   RobotControlMode_t active_mode,
                                   const RobotUserCommand_t *user,
                                   const RobotDroneOutput_t *drone,
                                   uint32_t gait_phase_index,
                                   float gait_phase_time_s);  // 전체 실행 시간과 이상 순간을 기록한다.

void ControlTimingDebug_RecordTimerMissed(uint32_t missed_control_count);  // ISR의 중복 요청 수를 기록한다.

#endif
