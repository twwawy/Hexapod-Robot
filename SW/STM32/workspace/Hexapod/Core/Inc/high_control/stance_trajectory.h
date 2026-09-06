#ifndef STANCE_TRAJECTORY_H
#define STANCE_TRAJECTORY_H

#include "common/robot_types.h"

RobotVec3_t StanceTrajectory_Advance(const RobotVec3_t *start,
                                    const RobotBodyTwist_t *twist,
                                    float duration_s);  // 일정한 몸체 이동의 반대 변환을 적용한다.

RobotVec3_t StanceTrajectory_Interpolate(float progress,
                                         const RobotVec3_t *start,
                                         const RobotVec3_t *end);  // Stance 선형 궤적을 계산한다.

#endif
