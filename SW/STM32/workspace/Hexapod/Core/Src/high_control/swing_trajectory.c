#include "high_control/swing_trajectory.h"

#include <math.h>
#include <stddef.h>

/* Cubic Smoothstep과 포물선으로 Swing 발 위치를 계산한다. */
RobotVec3_t SwingTrajectory_Calculate(float progress,
                                      const RobotVec3_t *start,
                                      const RobotVec3_t *end,
                                      float height_m,
                                      float radial_offset_m,
                                      float leg_angle_rad)
{
    RobotVec3_t output = {0.0f, 0.0f, 0.0f};  // 기본 결과를 0으로 준비한다.
    float limited;      // 제한된 진행률을 저장한다.
    float scaled;       // Smoothstep 진행률을 저장한다.
    float height_curve; // Swing 높이 곡선을 저장한다.
    float radial_bulge; // Swing 방사 오프셋을 저장한다.

    if ((start == NULL) || (end == NULL))
    {
        return output;
    }

    limited = fminf(fmaxf(progress, 0.0f), 1.0f);  // 진행률을 0~1로 제한한다.
    scaled = limited * limited * (3.0f - 2.0f * limited);  // Cubic Smoothstep을 계산한다.
    height_curve = 4.0f * height_m * scaled * (1.0f - scaled);  // 최고점 높이를 계산한다.
    output.x = start->x + (end->x - start->x) * scaled;  // Swing X를 계산한다.
    output.y = start->y + (end->y - start->y) * scaled;  // Swing Y를 계산한다.
    output.z = start->z + (end->z - start->z) * scaled + height_curve;  // Swing Z를 계산한다.
    radial_bulge = radial_offset_m * 4.0f * scaled * (1.0f - scaled);  // 중간 방사 이동을 계산한다.
    output.x += radial_bulge * cosf(leg_angle_rad);  // 다리 바깥 방향 X를 추가한다.
    output.y += radial_bulge * sinf(leg_angle_rad);  // 다리 바깥 방향 Y를 추가한다.
    return output;
}

static float AdaptiveQuintic(float x)
{
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    return x*x*x*(10.0f + x*(-15.0f + 6.0f*x));
}

RobotVec3_t SwingTrajectory_CalculateAdaptive(float progress,
    const RobotVec3_t *start, const RobotVec3_t *end, float clearance,
    float apex_phase, float transfer_phase)
{
    RobotVec3_t out = {0};
    if (!start || !end) return out;
    const float apex = fminf(fmaxf(apex_phase, 0.3f), 0.7f);
    const float transfer = fminf(fmaxf(transfer_phase, 0.35f), 0.65f);
    const float xy = AdaptiveQuintic((progress - (transfer - 0.25f))/0.5f);
    const float up = AdaptiveQuintic(progress/(apex - 0.2f));
    const float down = AdaptiveQuintic((progress - (apex + 0.2f))/(0.8f-apex));
    const float top = fmaxf(start->z, end->z) + clearance;
    out.x = start->x + xy*(end->x-start->x);
    out.y = start->y + xy*(end->y-start->y);
    out.z = start->z + up*(top-start->z) + down*(end->z-top);
    return out;
}
