#ifndef ROBOT_BRINGUP_H
#define ROBOT_BRINGUP_H

#include "common/robot_config.h"

#include <stdbool.h>
#include <stdint.h>

#define ROBOT_BRINGUP_STAGE              1U                                 // 현재 실기 시험 단계를 선택한다.
#define ROBOT_BRINGUP_STAGE_MIN          1U                                 // 첫 시험 단계를 정의한다.
#define ROBOT_BRINGUP_STAGE_MAX          6U                                 // 최종 운용 단계를 정의한다.
#define ROBOT_BRINGUP_LOW_SPEED_MPS      0.05f                              // 저속 보행 시험 한계를 정의한다.
#define ROBOT_BRINGUP_LOW_YAW_RATE_RADPS (15.0f * ROBOT_DEG_TO_RAD_F)        // 저속 회전 시험 한계를 정의한다.

#if (ROBOT_BRINGUP_STAGE < ROBOT_BRINGUP_STAGE_MIN) || \
    (ROBOT_BRINGUP_STAGE > ROBOT_BRINGUP_STAGE_MAX)
#error "ROBOT_BRINGUP_STAGE must be between 1 and 6"
#endif

typedef struct
{
    uint8_t stage;                // 현재 시험 단계를 저장한다.
    float linear_limit_mps;       // 현재 선속도 한계를 저장한다.
    float yaw_limit_radps;        // 현재 회전속도 한계를 저장한다.
    bool neutral_output_active;   // 0도 고정 출력 여부를 저장한다.
    bool stand_landing_allowed;   // 서기·착지 허가 여부를 저장한다.
    bool correction_allowed;      // 보정 모드 허가 여부를 저장한다.
    bool walking_allowed;         // 보행 모드 허가 여부를 저장한다.
    bool relay_enabled;           // 실제 릴레이 출력 여부를 저장한다.
} RobotBringupStatus_t;

#endif
