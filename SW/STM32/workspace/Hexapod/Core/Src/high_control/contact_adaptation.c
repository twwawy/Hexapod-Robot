#include "high_control/contact_adaptation.h"

#include <math.h>
#include <stddef.h>

/* Swing 진행률 50% 이후 Stance 전환을 Early Landing으로 판단한다. */
bool ContactAdaptation_IsEarlyLanding(RobotLegState_t previous_state,
                                      RobotLegState_t current_state,
                                      float progress)
{
    return (previous_state == ROBOT_LEG_SWING) &&
           (current_state == ROBOT_LEG_STANCE) &&
           (progress >= ROBOT_EARLY_LANDING_PROGRESS);  // 조기 접촉 조건을 함께 검사한다.
}

/* 발을 아래와 다리 안쪽으로 이동시켜 Late Landing을 탐색한다. */
void ContactAdaptation_ApplyLateLanding(RobotVec3_t *position,
                                        float leg_angle_rad)
{
    if (position == NULL)
    {
        return;
    }

    position->x -= ROBOT_LATE_INWARD_SPEED_MPS * cosf(leg_angle_rad) *
                   ROBOT_CONTROL_PERIOD_S;  // 다리 안쪽 X 이동을 적용한다.
    position->y -= ROBOT_LATE_INWARD_SPEED_MPS * sinf(leg_angle_rad) *
                   ROBOT_CONTROL_PERIOD_S;  // 다리 안쪽 Y 이동을 적용한다.
    position->z -= ROBOT_LATE_LANDING_SPEED_MPS * ROBOT_CONTROL_PERIOD_S;  // 발을 아래로 내린다.
}
