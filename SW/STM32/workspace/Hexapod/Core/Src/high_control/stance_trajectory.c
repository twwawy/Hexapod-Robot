#include "high_control/stance_trajectory.h"

#include <math.h>
#include <stddef.h>

/* 시작점과 끝점 사이의 Stance 선형 궤적을 계산한다. */
RobotVec3_t StanceTrajectory_Interpolate(float progress,
                                         const RobotVec3_t *start,
                                         const RobotVec3_t *end)
{
    RobotVec3_t output = {0.0f, 0.0f, 0.0f};  // 기본 결과를 0으로 준비한다.
    const float limited = fminf(fmaxf(progress, 0.0f), 1.0f);  // 진행률을 0~1로 제한한다.

    if ((start == NULL) || (end == NULL))
    {
        return output;
    }

    output.x = start->x + limited * (end->x - start->x);  // Stance X를 보간한다.
    output.y = start->y + limited * (end->y - start->y);  // Stance Y를 보간한다.
    output.z = start->z + limited * (end->z - start->z);  // Stance Z를 보간한다.
    return output;
}
