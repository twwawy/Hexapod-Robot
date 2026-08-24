#ifndef KINEMATICS_TEST_H
#define KINEMATICS_TEST_H

#include <stdbool.h>

bool KinematicsTest_Run(float tolerance_m, float tolerance_rad);  // 여섯 다리 좌표변환과 FK·IK 왕복을 검사한다.

#endif
