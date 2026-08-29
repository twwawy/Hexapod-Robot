#ifndef GAIT_MANAGER_H
#define GAIT_MANAGER_H

#include "common/robot_types.h"

typedef struct
{
    uint32_t phase_index;                        // 현재 Tripod 위상 번호를 저장한다.
    uint32_t start_wait_count;                   // 첫 보행 입력 대기 주기를 저장한다.
    uint32_t start_preview_count;                // 첫 위상 분산 검사 주기를 저장한다.
    uint32_t phase_cycle_count;                  // 현재 위상 제어 주기를 저장한다.
    float phase_time_s;                          // 현재 위상 경과 시간을 저장한다.
    float late_landing_time_s[ROBOT_LEG_COUNT];  // 다리별 Late Landing 탐색 시간을 저장한다.
    uint8_t support_recovery_mask;               // 재접촉을 기다리는 기존 지지발을 저장한다.
    bool airborne_seen[ROBOT_LEG_COUNT];         // Swing 중 비접촉 확인을 저장한다.
    bool landed[ROBOT_LEG_COUNT];                // Swing 착지 확인을 저장한다.
    bool initialized;                            // 정상 보행 초기화 여부를 저장한다.
    bool run_enable;                             // 내부 보행 활성화를 저장한다.
    bool stop_pending;                           // 현재 Swing 착지 후 정지 요청을 저장한다.
    bool resume_phase;                           // 정지 후 다음 Tripod 재개 여부를 저장한다.
    bool next_phase_enable;                      // 25 ms 전에 확정한 보행 Enable을 저장한다.
    bool next_phase_locked;                      // 다음 위상 결정 완료 여부를 저장한다.
    bool late_landing_hold;                      // 탐색 한계 후 보행 정지를 저장한다.
    bool support_recovery_active;                // 지지발 재착지 일시정지를 저장한다.
} GaitManager_Handle_t;

void GaitManager_Init(GaitManager_Handle_t *handle);  // Tripod 상태를 정지로 초기화한다.

RobotGaitPhase_t GaitManager_Step(GaitManager_Handle_t *handle,
                                  bool normal_mode_enable,
                                  bool tripod_enable,
                                  bool phase_validation_done,
                                  bool phase_validation_accepted,
                                  RobotTripodMode_t tripod_mode,
                                  float recovery_progress,
                                  const bool contact[ROBOT_LEG_COUNT]);  // 확정 접촉으로 다리 상태를 갱신한다.

RobotGaitPhase_t GaitManager_StepContacts(
    GaitManager_Handle_t *handle,
    bool normal_mode_enable,
    bool tripod_enable,
    bool phase_validation_done,
    bool phase_validation_accepted,
    RobotTripodMode_t tripod_mode,
    float recovery_progress,
    const bool contact[ROBOT_LEG_COUNT],
    const bool contact_raw[ROBOT_LEG_COUNT]);  // 접촉 후보와 확정값으로 다리 상태를 갱신한다.

#endif
