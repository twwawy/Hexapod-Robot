#include "high_control/stance_trajectory.h"

#include <math.h>
#include <stddef.h>

/* 고정된 몸체 속도로 지지발을 이동해 회전 적분 오차를 줄인다. */
RobotVec3_t StanceTrajectory_Advance(const RobotVec3_t *start,
                                    const RobotBodyTwist_t *twist,
                                    float duration_s)
{
    RobotVec3_t output = {0};  // 잘못된 입력의 기본 출력을 준비한다.

    if ((start == NULL) || (twist == NULL))
    {
        return output;
    }

    const float angle = twist->wz * duration_s;  // 몸체 회전량을 계산한다.
    const float sine = sinf(angle);              // 역회전에 사용할 사인값을 계산한다.
    const float cosine = cosf(angle);            // 역회전에 사용할 코사인값을 계산한다.
    const float sinc = (fabsf(angle) < 0.001f)
                     ? 1.0f - angle * angle / 6.0f
                     : sine / angle;  // 작은 회전에서도 평행이동 정밀도를 유지한다.
    const float cosc = (fabsf(angle) < 0.001f)
                     ? angle * (0.5f - angle * angle / 24.0f)
                     : (1.0f - cosine) / angle;  // 작은 각도의 나눗셈 손실을 줄인다.

    output.x = cosine * start->x + sine * start->y - duration_s *
               (twist->vx * sinc + twist->vy * cosc);  // 몸체 이동 반대 방향의 X를 계산한다.
    output.y = -sine * start->x + cosine * start->y + duration_s *
               (twist->vx * cosc - twist->vy * sinc);  // 몸체 이동 반대 방향의 Y를 계산한다.
    output.z = start->z - twist->vz * duration_s;       // 지지발의 수직 위치를 계산한다.
    return output;
}

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
