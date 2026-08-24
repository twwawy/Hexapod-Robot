#ifndef CONTACT_ADAPTATION_H
#define CONTACT_ADAPTATION_H

#include "common/robot_types.h"

bool ContactAdaptation_IsEarlyLanding(RobotLegState_t previous_state,
                                      RobotLegState_t current_state,
                                      float progress);  // Swing 중 조기 접촉 전환을 확인한다.

void ContactAdaptation_ApplyLateLanding(RobotVec3_t *position,
                                        float leg_angle_rad);  // Late Landing 탐색 이동을 적용한다.

#endif
