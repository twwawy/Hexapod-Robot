#ifndef FOOT_TRAJECTORY_H
#define FOOT_TRAJECTORY_H

#include "common/robot_types.h"

typedef struct
{
    RobotVec3_t start[ROBOT_LEG_COUNT];       // 검사 시작 순간의 발 위치를 저장한다.
    RobotVec3_t nominal[ROBOT_LEG_COUNT];     // 잔차 적용 전 착지 목표를 저장한다.
    RobotVec3_t target[ROBOT_LEG_COUNT];      // 검사와 실행이 공유할 착지 목표를 저장한다.
    float nominal_height[ROBOT_LEG_COUNT];   // 잔차 적용 전 Swing 높이를 저장한다.
    float height[ROBOT_LEG_COUNT];           // 검사와 실행이 공유할 Swing 높이를 저장한다.
    RobotBodyTwist_t twist;                  // 기본 계획을 만든 위상 속도를 저장한다.
    uint16_t plan_id;                        // 기본 계획의 변경 번호를 저장한다.
    uint8_t swing_mask;                      // 이번 계획에 잔차를 적용할 다리를 저장한다.
    bool valid;                             // 공개 가능한 계획의 존재를 저장한다.
} FootTrajectory_Plan_t;

typedef struct
{
    RobotVec3_t body_offset_m;                       // 보정 모드 몸체 Offset을 저장한다.
    RobotVec3_t memory[ROBOT_LEG_COUNT];             // 다리별 연속 발 위치를 저장한다.
    RobotVec3_t swing_start[ROBOT_LEG_COUNT];        // 사용자 정의 Swing 시작점을 저장한다.
    RobotVec3_t recovery_start[ROBOT_LEG_COUNT];     // 복구 Swing 시작점을 저장한다.

    RobotBodyTwist_t phase_twist;                     // 현재 위상에 고정한 보행 속도를 저장한다.
    FootTrajectory_Plan_t pending_plan;                // 검증을 통과한 다음 위상 계획을 저장한다.
    FootTrajectory_Plan_t active_plan;                 // 현재 Swing에 고정한 계획을 저장한다.
    uint16_t active_plan_id;                           // 실제 이륙에 사용한 계획 번호를 저장한다.
    uint8_t active_plan_mask;                          // 실제 잔차를 적용한 다리를 저장한다.
    bool active_plan_valid;                           // 실제 RL 위상 적용 이력을 저장한다.

    float landing_target_z[ROBOT_LEG_COUNT];         // 다리별 정상 착지 Z를 저장한다.
    float previous_progress[ROBOT_LEG_COUNT];        // 위상 정지 중 지지발 적분을 막을 진행률을 저장한다.
    float landing_z_error[ROBOT_LEG_COUNT];          // 다리별 PWM 명령 착지 Z 오차를 저장한다.
    float common_z_recovery_remaining[2];            // 두 Tripod의 남은 Z 복구량을 저장한다.
    float common_z_recovery_total[2];                // 두 Tripod의 S-curve 전체 복구량을 저장한다.
    float common_z_recovery_progress[2];             // 두 Tripod의 S-curve 진행률을 저장한다.
    float swing_resume_progress[ROBOT_LEG_COUNT];    // 정지 후 Swing 재개 진행률을 저장한다.
    RobotLegState_t previous_state[ROBOT_LEG_COUNT];  // HOLD를 제외한 직전 다리 상태를 저장한다.
    uint8_t previous_swing_mask;                     // 직전 Swing 다리 비트를 저장한다.

    bool adapted_stance[ROBOT_LEG_COUNT];            // 접촉 적응 Stance 여부를 저장한다.
    bool custom_swing[ROBOT_LEG_COUNT];              // 연속 시작점 사용 여부를 저장한다.
    bool landing_z_error_valid[ROBOT_LEG_COUNT];     // 다리별 착지 Z 오차 수집 여부를 저장한다.
    bool touchdown_pending[ROBOT_LEG_COUNT];         // 접촉 확인 후 첫 명령 Z 반영 대기를 저장한다.
    bool swing_resume_active[ROBOT_LEG_COUNT];       // 정지 위치 기반 Swing 재개를 저장한다.
    bool phase_twist_valid;                          // 현재 위상 속도 고정 여부를 저장한다.
    bool initialized;                                // 기본 발 위치 초기화 여부를 저장한다.
} FootTrajectory_Handle_t;

void FootTrajectory_Init(FootTrajectory_Handle_t *handle);  // 발 위치와 보정 Offset을 초기화한다.

void FootTrajectory_BuildPlan(FootTrajectory_Plan_t *plan,
                               const RobotVec3_t feet[ROBOT_LEG_COUNT],
                               const RobotVec3_t *body_offset_m,
                               const RobotBodyTwist_t *twist,
                               RobotGaitPattern_t pattern,
                               uint8_t swing_mask);  // 다음 위상의 기본 착지점과 높이를 계산한다.

bool FootTrajectory_SetPlan(FootTrajectory_Handle_t *handle,
                             const FootTrajectory_Plan_t *plan);  // 검증된 계획을 다음 이륙까지 보관한다.

void FootTrajectory_CancelPlan(FootTrajectory_Handle_t *handle);  // 진행 중 Swing을 보존하며 대기 계획을 취소한다.

bool FootTrajectory_LatchTouchdown(FootTrajectory_Handle_t *handle,
                                   uint8_t leg,
                                   const RobotVec3_t *commanded_foot_body,
                                   const RobotEuler_t *posture_rad);  // 접촉 순간 PWM 명령 발 위치를 고정한다.

bool FootTrajectory_LatchCommandedTouchdown(
    FootTrajectory_Handle_t *handle,
    uint8_t leg);  // 접촉 순간의 직전 궤적 명령 위치를 고정한다.

void FootTrajectory_UpdateCommandedLanding(FootTrajectory_Handle_t *handle,
                                            const RobotGaitPhase_t *gait,
                                            bool common_z_enable);  // 접촉 확인 후 고정한 궤적 Z 오차를 반영한다.

RobotFootTargets_t FootTrajectory_Step(FootTrajectory_Handle_t *handle,
                                       const RobotBodyTwist_t *twist,
                                       const RobotDroneOutput_t *drone,
                                       const RobotGaitPhase_t *gait,
                                       const RobotEuler_t *posture_rad);  // 여섯 발의 연속 궤적을 계산한다.

#endif
