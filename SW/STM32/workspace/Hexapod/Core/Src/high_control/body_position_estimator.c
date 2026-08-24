#include "high_control/body_position_estimator.h"

#include "high_control/leg_kinematics.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* Body 좌표 벡터를 ZYX 자세로 World 좌표에 회전한다. */
static RobotVec3_t BodyPositionEstimator_RotateToWorld(const RobotVec3_t *body,
                                                       const RobotEuler_t *attitude)
{
    RobotVec3_t world;                         // 회전한 World 벡터를 저장한다.
    const float cr = cosf(attitude->roll);    // Roll Cosine을 계산한다.
    const float sr = sinf(attitude->roll);    // Roll Sine을 계산한다.
    const float cp = cosf(attitude->pitch);   // Pitch Cosine을 계산한다.
    const float sp = sinf(attitude->pitch);   // Pitch Sine을 계산한다.
    const float cy = cosf(attitude->yaw);     // Yaw Cosine을 계산한다.
    const float sy = sinf(attitude->yaw);     // Yaw Sine을 계산한다.

    world.x = cy * cp * body->x +
              (cy * sp * sr - sy * cr) * body->y +
              (cy * sp * cr + sy * sr) * body->z;  // World X를 계산한다.
    world.y = sy * cp * body->x +
              (sy * sp * sr + cy * cr) * body->y +
              (sy * sp * cr - cy * sr) * body->z;  // World Y를 계산한다.
    world.z = -sp * body->x + cp * sr * body->y + cp * cr * body->z;  // World Z를 계산한다.
    return world;
}

/* 두 3차원 후보 위치 사이 거리를 계산한다. */
static float BodyPositionEstimator_Distance(const RobotVec3_t *a,
                                            const RobotVec3_t *b)
{
    const float dx = a->x - b->x;  // X 차이를 계산한다.
    const float dy = a->y - b->y;  // Y 차이를 계산한다.
    const float dz = a->z - b->z;  // Z 차이를 계산한다.
    return sqrtf(dx * dx + dy * dy + dz * dz);  // 3차원 거리를 반환한다.
}

/* 몸체 위치 Anchor와 Slip 상태를 초기화한다. */
void BodyPositionEstimator_Init(BodyPositionEstimator_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));  // 이전 추정 상태를 제거한다.
    }
}

/* 접촉 중인 Stance 발 Anchor로 몸체 위치와 Slip을 추정한다. */
BodyPositionEstimator_Output_t BodyPositionEstimator_Step(
    BodyPositionEstimator_Handle_t *handle,
    const float joint_angle_rad[ROBOT_JOINT_COUNT],
    const RobotGaitPhase_t *gait,
    const bool contact[ROBOT_LEG_COUNT],
    const RobotEuler_t *attitude_rad)
{
    BodyPositionEstimator_Output_t output;  // 이번 위치 추정 결과를 저장한다.
    RobotVec3_t candidate[ROBOT_LEG_COUNT]; // 다리별 몸체 위치 후보를 저장한다.
    bool raw_valid[ROBOT_LEG_COUNT];        // 다리별 원시 유효성을 저장한다.
    bool accepted[ROBOT_LEG_COUNT];         // 평균에 사용할 후보를 저장한다.
    RobotVec3_t accepted_sum = {0.0f, 0.0f, 0.0f};  // 채택 후보 합계를 저장한다.
    uint32_t raw_valid_count = 0U;          // 원시 유효 다리 수를 저장한다.
    uint32_t leg;                           // 처리할 다리 번호를 저장한다.

    memset(&output, 0, sizeof(output));     // 기본 결과를 0으로 준비한다.
    memset(candidate, 0, sizeof(candidate));// 후보 배열을 초기화한다.
    memset(raw_valid, 0, sizeof(raw_valid));// 원시 유효성을 초기화한다.
    memset(accepted, 0, sizeof(accepted));  // 평균 채택 상태를 초기화한다.

    if ((handle == NULL) || (joint_angle_rad == NULL) ||
        (gait == NULL) || (contact == NULL) || (attitude_rad == NULL))
    {
        return output;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t foot_body;   // FK Body 발 위치를 저장한다.
        RobotVec3_t foot_world;  // 회전한 World 발 벡터를 저장한다.
        const bool valid = (gait->state[leg] == ROBOT_LEG_STANCE) && contact[leg];  // Stance Contact를 확인한다.

        (void)LegKinematics_Forward((uint8_t)leg,
                                    &joint_angle_rad[leg * ROBOT_JOINTS_PER_LEG],
                                    &foot_body);  // 현재 관절각의 FK를 계산한다.
        foot_world = BodyPositionEstimator_RotateToWorld(&foot_body, attitude_rad);  // FK 벡터를 World로 회전한다.

        if (valid && !handle->was_valid[leg])
        {
            handle->anchor_world[leg].x = handle->body_position_world.x + foot_world.x;  // 새 World X Anchor를 만든다.
            handle->anchor_world[leg].y = handle->body_position_world.y + foot_world.y;  // 새 World Y Anchor를 만든다.
            handle->anchor_world[leg].z = handle->body_position_world.z + foot_world.z;  // 새 World Z Anchor를 만든다.
        }

        if (valid)
        {
            candidate[leg].x = handle->anchor_world[leg].x - foot_world.x;  // Anchor에서 Body X 후보를 계산한다.
            candidate[leg].y = handle->anchor_world[leg].y - foot_world.y;  // Anchor에서 Body Y 후보를 계산한다.
            candidate[leg].z = handle->anchor_world[leg].z - foot_world.z;  // Anchor에서 Body Z 후보를 계산한다.
            raw_valid[leg] = true;                                        // 원시 후보를 유효로 표시한다.
            raw_valid_count++;                                            // 원시 유효 다리 수를 늘린다.
        }

        handle->was_valid[leg] = valid;  // 다음 주기 Stance 진입 검출을 위해 저장한다.
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        uint32_t other;          // 비교할 다른 다리 번호를 저장한다.
        bool has_neighbor = false;  // 0.05 m 안의 정상 후보 여부를 저장한다.

        if (!raw_valid[leg])
        {
            handle->slip_suspect[leg] = 0U;  // Stance 밖에서 의심 횟수를 제거한다.
            handle->slip_latched[leg] = false; // 다음 Stance를 위해 Latch를 해제한다.
            continue;
        }

        if (handle->slip_latched[leg])
        {
            continue;
        }

        if (raw_valid_count < 2U)
        {
            accepted[leg] = true;             // 단일 후보는 비교 없이 사용한다.
            handle->slip_suspect[leg] = 0U;   // 의심 횟수를 제거한다.
            continue;
        }

        for (other = 0U; other < ROBOT_LEG_COUNT; ++other)
        {
            if ((other == leg) || !raw_valid[other] || handle->slip_latched[other])
            {
                continue;
            }

            if (BodyPositionEstimator_Distance(&candidate[leg], &candidate[other]) <=
                ROBOT_SLIP_DISTANCE_M)
            {
                has_neighbor = true;  // 정상 거리의 다른 후보를 찾는다.
                break;
            }
        }

        if (has_neighbor)
        {
            accepted[leg] = true;             // 가까운 후보를 평균에 채택한다.
            handle->slip_suspect[leg] = 0U;   // 의심 횟수를 제거한다.
        }
        else
        {
            if (handle->slip_suspect[leg] < ROBOT_SLIP_CONFIRM_SAMPLES)
            {
                handle->slip_suspect[leg]++;  // 고립 후보 연속 횟수를 늘린다.
            }

            if (handle->slip_suspect[leg] >= ROBOT_SLIP_CONFIRM_SAMPLES)
            {
                handle->slip_latched[leg] = true;  // 5회 연속 고립을 Slip으로 Latch한다.
            }
        }
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (accepted[leg])
        {
            accepted_sum.x += candidate[leg].x;  // 채택 후보 X를 누적한다.
            accepted_sum.y += candidate[leg].y;  // 채택 후보 Y를 누적한다.
            accepted_sum.z += candidate[leg].z;  // 채택 후보 Z를 누적한다.
            output.valid_leg_count++;             // 채택 다리 수를 늘린다.
        }

        if (handle->slip_latched[leg])
        {
            output.slip_leg_mask |= (uint8_t)(1U << leg);  // Slip 다리를 비트로 표시한다.
        }
    }

    if (output.valid_leg_count > 0U)
    {
        const float divisor = (float)output.valid_leg_count;  // 평균 분모를 계산한다.
        handle->body_position_world.x = accepted_sum.x / divisor;  // 채택 후보 X 평균을 계산한다.
        handle->body_position_world.y = accepted_sum.y / divisor;  // 채택 후보 Y 평균을 계산한다.
        handle->body_position_world.z = accepted_sum.z / divisor;  // 채택 후보 Z 평균을 계산한다.
    }

    output.position_world = handle->body_position_world;  // 최신 몸체 위치를 반환한다.
    return output;
}
