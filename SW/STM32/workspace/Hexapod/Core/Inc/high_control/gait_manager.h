#ifndef GAIT_MANAGER_H
#define GAIT_MANAGER_H

#include "common/robot_types.h"

typedef struct
{
    uint32_t phase_index;                    // 현재 Tripod 위상 번호를 저장한다.
    float phase_time_s;                      // 현재 위상 경과 시간을 저장한다.
    bool airborne_seen[ROBOT_LEG_COUNT];     // Swing 중 비접촉 확인을 저장한다.
    bool landed[ROBOT_LEG_COUNT];            // Swing 착지 확인을 저장한다.
    bool initialized;                        // 정상 보행 초기화 여부를 저장한다.
    bool run_enable;                         // 내부 보행 활성화를 저장한다.
    bool stop_pending;                       // 현재 Swing 착지 후 정지 요청을 저장한다.
} GaitManager_Handle_t;

void GaitManager_Init(GaitManager_Handle_t *handle);  // Tripod 상태를 정지로 초기화한다.

RobotGaitPhase_t GaitManager_Step(GaitManager_Handle_t *handle,
                                  bool tripod_enable,
                                  RobotTripodMode_t tripod_mode,
                                  float recovery_progress,
                                  const bool contact[ROBOT_LEG_COUNT]);  // 다리별 상태와 진행률을 갱신한다.

#endif
