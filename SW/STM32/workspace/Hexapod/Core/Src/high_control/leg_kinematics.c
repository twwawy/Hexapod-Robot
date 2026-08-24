#include "high_control/leg_kinematics.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define LEG_ROOT_DISTANCE_M 0.1845f
#define LEG_IK_TOLERANCE    1.0e-6f
#define LEG_SMALL_VALUE     1.0e-9f

static const float leg_angle_rad[ROBOT_LEG_COUNT] =
{
    -45.0f * ROBOT_DEG_TO_RAD_F,   // 1번 다리 장착각을 저장한다.
    -90.0f * ROBOT_DEG_TO_RAD_F,   // 2번 다리 장착각을 저장한다.
    -135.0f * ROBOT_DEG_TO_RAD_F,  // 3번 다리 장착각을 저장한다.
    45.0f * ROBOT_DEG_TO_RAD_F,    // 4번 다리 장착각을 저장한다.
    90.0f * ROBOT_DEG_TO_RAD_F,    // 5번 다리 장착각을 저장한다.
    135.0f * ROBOT_DEG_TO_RAD_F    // 6번 다리 장착각을 저장한다.
};

/* 한 다리의 Body 기준 관절축 원점을 계산한다. */
static bool LegKinematics_GetRoot(uint8_t leg, RobotVec3_t *root)
{
    const float diagonal = LEG_ROOT_DISTANCE_M / sqrtf(2.0f);  // 대각 원점 거리를 계산한다.

    if ((leg >= ROBOT_LEG_COUNT) || (root == NULL))
    {
        return false;
    }

    root->z = 0.0f;  // 모든 다리 원점의 Z를 맞춘다.

    switch (leg)
    {
        case 0U: root->x = diagonal;  root->y = -diagonal; break;  // 1번 다리 원점을 선택한다.
        case 1U: root->x = 0.0f;      root->y = -LEG_ROOT_DISTANCE_M; break;  // 2번 다리 원점을 선택한다.
        case 2U: root->x = -diagonal; root->y = -diagonal; break;  // 3번 다리 원점을 선택한다.
        case 3U: root->x = diagonal;  root->y = diagonal; break;   // 4번 다리 원점을 선택한다.
        case 4U: root->x = 0.0f;      root->y = LEG_ROOT_DISTANCE_M; break;   // 5번 다리 원점을 선택한다.
        default: root->x = -diagonal; root->y = diagonal; break;  // 6번 다리 원점을 선택한다.
    }

    return true;
}

/* 다리 좌표의 3DOF IK 후보를 계산한다. */
static bool LegKinematics_SolveLocal(const RobotVec3_t *local,
                                     float angle_rad[ROBOT_JOINTS_PER_LEG])
{
    float radial;        // Coxa 축의 수평 거리를 저장한다.
    float planar;        // Femur 원점의 평면 거리를 저장한다.
    float cosine_knee;   // 무릎각의 Cosine을 저장한다.
    float sine_knee;     // 무릎각의 Sine을 저장한다.

    if ((local == NULL) || (angle_rad == NULL) ||
        !isfinite(local->x) || !isfinite(local->y) || !isfinite(local->z))
    {
        return false;
    }

    radial = sqrtf(local->x * local->x + local->y * local->y);             // Coxa 수평 거리를 계산한다.
    planar = radial - ROBOT_LINK_1_M;                                       // 2-link 평면 거리를 계산한다.
    cosine_knee = (planar * planar + local->z * local->z -
                   ROBOT_LINK_2_M * ROBOT_LINK_2_M -
                   ROBOT_LINK_3_M * ROBOT_LINK_3_M) /
                  (2.0f * ROBOT_LINK_2_M * ROBOT_LINK_3_M);                // Cosine 법칙을 적용한다.

    if ((cosine_knee < (-1.0f - LEG_IK_TOLERANCE)) ||
        (cosine_knee > (1.0f + LEG_IK_TOLERANCE)))
    {
        return false;
    }

    cosine_knee = fminf(fmaxf(cosine_knee, -1.0f), 1.0f);                  // 수치 오차를 제한한다.
    sine_knee = sqrtf(fmaxf(0.0f, 1.0f - cosine_knee * cosine_knee));       // 무릎 양의 해를 선택한다.
    angle_rad[0] = atan2f(local->y, local->x);                              // Coxa 각을 계산한다.
    angle_rad[2] = atan2f(sine_knee, cosine_knee);                          // Tibia 각을 계산한다.
    angle_rad[1] = atan2f(-local->z, planar) -
                   atan2f(ROBOT_LINK_3_M * sine_knee,
                          ROBOT_LINK_2_M + ROBOT_LINK_3_M * cosine_knee);   // Femur 각을 계산한다.

    return isfinite(angle_rad[0]) && isfinite(angle_rad[1]) && isfinite(angle_rad[2]) &&
           (fabsf(angle_rad[0]) <= ROBOT_JOINT_MAX_RAD) &&
           (fabsf(angle_rad[1]) <= ROBOT_JOINT_MAX_RAD) &&
           (fabsf(angle_rad[2]) <= ROBOT_JOINT_MAX_RAD);  // 관절 제한을 함께 확인한다.
}

/* 각 다리의 마지막 정상 IK 해를 영점으로 초기화한다. */
void LegKinematics_Init(LegKinematics_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));  // IK 유지값을 제거한다.
    }
}

/* 서기 자세의 여섯 기본 발 위치를 Body 좌표로 반환한다. */
void LegKinematics_GetBaseFeet(RobotVec3_t foot_body[ROBOT_LEG_COUNT])
{
    uint32_t leg;  // 계산할 다리 번호를 저장한다.

    if (foot_body == NULL)
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t root;  // 현재 다리 원점을 저장한다.
        (void)LegKinematics_GetRoot((uint8_t)leg, &root);                  // 다리 원점을 계산한다.
        foot_body[leg].x = root.x + ROBOT_BASE_FOOT_RADIUS_M * cosf(leg_angle_rad[leg]);  // 기본 X를 계산한다.
        foot_body[leg].y = root.y + ROBOT_BASE_FOOT_RADIUS_M * sinf(leg_angle_rad[leg]);  // 기본 Y를 계산한다.
        foot_body[leg].z = ROBOT_BASE_FOOT_Z_M;                            // 기본 Z를 설정한다.
    }
}

/* Body 발 위치를 선택한 다리 좌표로 변환한다. */
bool LegKinematics_BodyToLeg(uint8_t leg,
                             const RobotVec3_t *body,
                             RobotVec3_t *local)
{
    RobotVec3_t root;  // 선택한 다리 원점을 저장한다.
    float dx;          // 원점 기준 X를 저장한다.
    float dy;          // 원점 기준 Y를 저장한다.
    float cosine;      // 장착각 Cosine을 저장한다.
    float sine;        // 장착각 Sine을 저장한다.

    if ((body == NULL) || (local == NULL) || !LegKinematics_GetRoot(leg, &root))
    {
        return false;
    }

    dx = body->x - root.x;             // 다리 원점 기준 X를 계산한다.
    dy = body->y - root.y;             // 다리 원점 기준 Y를 계산한다.
    cosine = cosf(leg_angle_rad[leg]);  // 장착각 Cosine을 계산한다.
    sine = sinf(leg_angle_rad[leg]);    // 장착각 Sine을 계산한다.
    local->x = dx * cosine + dy * sine; // 다리 X로 회전한다.
    local->y = -dx * sine + dy * cosine;// 다리 Y로 회전한다.
    local->z = body->z;                 // Z축을 그대로 전달한다.
    return true;
}

/* 선택한 다리 위치를 Body 좌표로 변환한다. */
bool LegKinematics_LegToBody(uint8_t leg,
                             const RobotVec3_t *local,
                             RobotVec3_t *body)
{
    RobotVec3_t root;  // 선택한 다리 원점을 저장한다.
    float cosine;      // 장착각 Cosine을 저장한다.
    float sine;        // 장착각 Sine을 저장한다.

    if ((local == NULL) || (body == NULL) || !LegKinematics_GetRoot(leg, &root))
    {
        return false;
    }

    cosine = cosf(leg_angle_rad[leg]);                            // 장착각 Cosine을 계산한다.
    sine = sinf(leg_angle_rad[leg]);                              // 장착각 Sine을 계산한다.
    body->x = root.x + local->x * cosine - local->y * sine;       // Body X로 회전한다.
    body->y = root.y + local->x * sine + local->y * cosine;       // Body Y로 회전한다.
    body->z = local->z;                                           // Z축을 그대로 전달한다.
    return true;
}

/* 한 다리의 관절각으로 Body 발 위치를 계산한다. */
bool LegKinematics_Forward(uint8_t leg,
                           const float angle_rad[ROBOT_JOINTS_PER_LEG],
                           RobotVec3_t *foot_body)
{
    RobotVec3_t local;  // 계산한 다리 좌표 발 위치를 저장한다.
    float radial;       // 다리 수평 반경을 저장한다.

    if ((leg >= ROBOT_LEG_COUNT) || (angle_rad == NULL) || (foot_body == NULL))
    {
        return false;
    }

    radial = ROBOT_LINK_1_M + ROBOT_LINK_2_M * cosf(angle_rad[1]) +
             ROBOT_LINK_3_M * cosf(angle_rad[1] + angle_rad[2]);   // 다리 수평 반경을 계산한다.
    local.x = radial * cosf(angle_rad[0]);                          // 다리 X를 계산한다.
    local.y = radial * sinf(angle_rad[0]);                          // 다리 Y를 계산한다.
    local.z = -ROBOT_LINK_2_M * sinf(angle_rad[1]) -
              ROBOT_LINK_3_M * sinf(angle_rad[1] + angle_rad[2]);  // 다리 Z를 계산한다.
    return LegKinematics_LegToBody(leg, &local, foot_body);         // Body 좌표로 변환한다.
}

/* 한 Body 발 위치의 IK를 계산하고 실패 시 직전 해를 유지한다. */
bool LegKinematics_Inverse(LegKinematics_Handle_t *handle,
                           uint8_t leg,
                           const RobotVec3_t *foot_body,
                           float angle_rad[ROBOT_JOINTS_PER_LEG])
{
    RobotVec3_t local;  // 다리 좌표 발 위치를 저장한다.
    float candidate[ROBOT_JOINTS_PER_LEG];  // 새 IK 후보를 저장한다.
    uint32_t joint;     // 복사할 관절 번호를 저장한다.

    if ((handle == NULL) || (leg >= ROBOT_LEG_COUNT) ||
        (foot_body == NULL) || (angle_rad == NULL))
    {
        return false;
    }

    if (!LegKinematics_BodyToLeg(leg, foot_body, &local) ||
        !LegKinematics_SolveLocal(&local, candidate))
    {
        for (joint = 0U; joint < ROBOT_JOINTS_PER_LEG; ++joint)
        {
            angle_rad[joint] = handle->last_angle_rad[leg * ROBOT_JOINTS_PER_LEG + joint];  // 직전 정상 해를 유지한다.
        }
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINTS_PER_LEG; ++joint)
    {
        handle->last_angle_rad[leg * ROBOT_JOINTS_PER_LEG + joint] = candidate[joint];  // 새 정상 해를 저장한다.
        angle_rad[joint] = candidate[joint];                                            // 새 정상 해를 반환한다.
    }

    return true;
}

/* 한 Body 발 위치가 관절 제한 안에서 도달 가능한지 확인한다. */
bool LegKinematics_IsReachable(uint8_t leg,
                               const RobotVec3_t *foot_body)
{
    RobotVec3_t local;  // 다리 좌표 발 위치를 저장한다.
    float angle[ROBOT_JOINTS_PER_LEG];  // 임시 IK 결과를 저장한다.

    return (foot_body != NULL) &&
           LegKinematics_BodyToLeg(leg, foot_body, &local) &&
           LegKinematics_SolveLocal(&local, angle);  // 상태 변경 없이 IK 가능 여부만 반환한다.
}

/* 한 발을 2-link 작업공간 안으로 최소 수정한다. */
bool LegKinematics_LimitFoot(uint8_t leg,
                             const RobotVec3_t *input_body,
                             RobotVec3_t *output_body,
                             bool *limited)
{
    RobotVec3_t local;          // 다리 좌표 입력을 저장한다.
    float radial;               // Coxa 수평 거리를 저장한다.
    float planar;               // Femur 기준 평면 거리를 저장한다.
    float reach;                // 2-link 원점 거리를 저장한다.
    float limited_reach;        // 제한된 원점 거리를 저장한다.
    float scale;                // 방사 방향 축척을 저장한다.

    if ((input_body == NULL) || (output_body == NULL) || (limited == NULL) ||
        !LegKinematics_BodyToLeg(leg, input_body, &local) ||
        !isfinite(local.x) || !isfinite(local.y) || !isfinite(local.z))
    {
        return false;
    }

    radial = sqrtf(local.x * local.x + local.y * local.y);                  // Coxa 수평 거리를 계산한다.
    planar = radial - ROBOT_LINK_1_M;                                        // 2-link 평면 좌표를 계산한다.
    reach = sqrtf(planar * planar + local.z * local.z);                      // 2-link 원점 거리를 계산한다.
    limited_reach = fminf(fmaxf(reach,
                                 fabsf(ROBOT_LINK_2_M - ROBOT_LINK_3_M) + ROBOT_WORKSPACE_MARGIN_M),
                           ROBOT_LINK_2_M + ROBOT_LINK_3_M - ROBOT_WORKSPACE_MARGIN_M);  // 여유를 둔 작업공간으로 제한한다.
    *limited = fabsf(limited_reach - reach) > LEG_SMALL_VALUE;               // 위치 수정 여부를 기록한다.

    if (*limited)
    {
        scale = (reach > LEG_SMALL_VALUE) ? (limited_reach / reach) : 1.0f;  // 방사 방향 축척을 계산한다.
        planar *= scale;                                                      // 평면 거리를 제한한다.
        local.z *= scale;                                                      // 수직 거리를 제한한다.

        if (radial > LEG_SMALL_VALUE)
        {
            const float xy_scale = (ROBOT_LINK_1_M + planar) / radial;  // Coxa 이후 반경의 축척을 계산한다.
            local.x *= xy_scale;                                        // 다리 X를 제한한다.
            local.y *= xy_scale;                                        // 다리 Y를 제한한다.
        }
    }

    return LegKinematics_LegToBody(leg, &local, output_body);  // 제한 결과를 Body 좌표로 반환한다.
}
