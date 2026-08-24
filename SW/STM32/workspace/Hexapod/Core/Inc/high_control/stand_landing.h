#ifndef STAND_LANDING_H
#define STAND_LANDING_H

#include "common/robot_types.h"

void StandLanding_Calculate(bool stand_enable,
                            bool landing_enable,
                            float posture_progress,
                            RobotVec3_t delta_body[ROBOT_LEG_COUNT]);  // 서기·착지 발 위치 변화량을 계산한다.

#endif
