#ifndef SAFETY_TEST_H
#define SAFETY_TEST_H

#include <stdbool.h>

bool SafetyTest_Run(void);  // 전복·비유한 IMU·IK 실패와 영구 Latch를 검사한다.

#endif
