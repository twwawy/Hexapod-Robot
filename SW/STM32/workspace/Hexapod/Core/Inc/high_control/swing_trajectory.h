#ifndef SWING_TRAJECTORY_H
#define SWING_TRAJECTORY_H

#include "common/robot_types.h"

RobotVec3_t SwingTrajectory_Calculate(float progress,
                                      const RobotVec3_t *start,
                                      const RobotVec3_t *end,
                                      float height_m,
                                      float radial_offset_m,
                                      float leg_angle_rad);  // Quintic-Bezier Swing 궤적을 계산한다.

#endif
