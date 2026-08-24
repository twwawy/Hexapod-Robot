#include "high_control/swing_trajectory.h"

#include <math.h>
#include <stddef.h>

/* Quintic 진행률과 Cubic Bezier를 사용해 Swing 발 위치를 계산한다. */
RobotVec3_t SwingTrajectory_Calculate(float progress,
                                      const RobotVec3_t *start,
                                      const RobotVec3_t *end,
                                      float height_m,
                                      float radial_offset_m,
                                      float leg_angle_rad)
{
    RobotVec3_t output = {0.0f, 0.0f, 0.0f};  // 기본 결과를 0으로 준비한다.
    float limited;      // 제한된 진행률을 저장한다.
    float scaled;       // Quintic 진행률을 저장한다.
    float one_minus;    // Bezier 반대 진행률을 저장한다.
    float b0;           // Bezier 첫 계수를 저장한다.
    float b1;           // Bezier 둘째 계수를 저장한다.
    float b2;           // Bezier 셋째 계수를 저장한다.
    float b3;           // Bezier 넷째 계수를 저장한다.
    float radial_bulge; // Swing 방사 오프셋을 저장한다.

    if ((start == NULL) || (end == NULL))
    {
        return output;
    }

    limited = fminf(fmaxf(progress, 0.0f), 1.0f);  // 진행률을 0~1로 제한한다.
    scaled = 10.0f * limited * limited * limited -
             15.0f * limited * limited * limited * limited +
             6.0f * limited * limited * limited * limited * limited;  // Quintic 진행률을 계산한다.
    one_minus = 1.0f - scaled;                       // Bezier 반대 진행률을 계산한다.
    b0 = one_minus * one_minus * one_minus;         // 첫 Bezier 계수를 계산한다.
    b1 = 3.0f * one_minus * one_minus * scaled;     // 둘째 Bezier 계수를 계산한다.
    b2 = 3.0f * one_minus * scaled * scaled;        // 셋째 Bezier 계수를 계산한다.
    b3 = scaled * scaled * scaled;                  // 넷째 Bezier 계수를 계산한다.

    output.x = (b0 + b1) * start->x + (b2 + b3) * end->x;  // Swing X를 계산한다.
    output.y = (b0 + b1) * start->y + (b2 + b3) * end->y;  // Swing Y를 계산한다.
    output.z = b0 * start->z +
               b1 * (start->z + (4.0f / 3.0f) * height_m) +
               b2 * (end->z + (4.0f / 3.0f) * height_m) +
               b3 * end->z;  // 최고점이 height와 일치하는 Bezier Z를 계산한다.
    radial_bulge = radial_offset_m * 4.0f * scaled * (1.0f - scaled);  // 중간 방사 이동을 계산한다.
    output.x += radial_bulge * cosf(leg_angle_rad);  // 다리 바깥 방향 X를 추가한다.
    output.y += radial_bulge * sinf(leg_angle_rad);  // 다리 바깥 방향 Y를 추가한다.
    return output;
}
