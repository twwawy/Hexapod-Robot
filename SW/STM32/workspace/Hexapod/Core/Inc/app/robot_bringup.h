#ifndef ROBOT_BRINGUP_H
#define ROBOT_BRINGUP_H

#include "common/robot_config.h"

#include <stdbool.h>
#include <stdint.h>

#define ROBOT_BRINGUP_STAGE              6U                                 // 현재 실기 시험 단계를 선택한다.
#define ROBOT_BRINGUP_STAGE_MIN          1U                                 // 첫 시험 단계를 정의한다.
#define ROBOT_BRINGUP_STAGE_MAX          6U                                 // 최종 운용 단계를 정의한다.
#define ROBOT_BRINGUP_LOW_SPEED_MPS      (ROBOT_MAX_LINEAR_SPEED_MPS * 0.5f) // 최대 선속도의 50%를 정의한다.
#define ROBOT_BRINGUP_LOW_YAW_RATE_RADPS (ROBOT_MAX_YAW_RATE_RADPS * 0.5f)   // 최대 회전속도의 50%를 정의한다.
#define ROBOT_BRINGUP_FAKE_RC_DELAY_MS   1000U                              // 임시 조종기 전원 허가 대기시간을 정의한다.
#define ROBOT_BRINGUP_STAGE3_STAND_MS    2000U                              // 자동 서기 시작 시각을 정의한다.
#define ROBOT_BRINGUP_STAGE3_LANDING_MS  10000U                             // 일반 U3 자동 착지 시각을 정의한다.
#define ROBOT_BRINGUP_PRESSURE_CALIBRATION 0U                               // U3 압력 자동 보정 사용 여부를 정의한다.

#if (ROBOT_BRINGUP_STAGE < ROBOT_BRINGUP_STAGE_MIN) || \
    (ROBOT_BRINGUP_STAGE > ROBOT_BRINGUP_STAGE_MAX)
#error "ROBOT_BRINGUP_STAGE must be between 1 and 6"
#endif

typedef struct
{
    uint8_t stage;                    // 현재 시험 단계를 저장한다.
    float linear_limit_mps;           // 현재 선속도 한계를 저장한다.
    float yaw_limit_radps;            // 현재 회전속도 한계를 저장한다.
    bool neutral_output_active;       // 0도 고정 출력 여부를 저장한다.
    bool stand_landing_allowed;       // 서기·착지 허가 여부를 저장한다.
    bool correction_allowed;          // 보정 모드 허가 여부를 저장한다.
    bool walking_allowed;             // 보행 모드 허가 여부를 저장한다.
    bool simulated_rc_active;         // 임시 조종기 사용 여부를 저장한다.
    uint8_t simulated_rc_phase;       // 자동 시험 진행 단계를 저장한다.
    uint32_t simulated_rc_elapsed_ms; // 자동 시험 경과시간을 저장한다.
    bool relay_enabled;               // 실제 릴레이 출력 여부를 저장한다.
} RobotBringupStatus_t;

extern volatile bool g_robot_bringup_emergency_stop;  // 디버거용 임시 긴급정지를 공개한다.

#endif
