#include "test/kinematics_test.h"

#include "high_control/leg_kinematics.h"

#include <math.h>

/* 기본 발 위치의 Body↔Leg와 IK→FK 왕복 오차를 검사한다. */
bool KinematicsTest_Run(float tolerance_m, float tolerance_rad)
{
    LegKinematics_Handle_t kinematics;           // 시험용 IK 상태를 저장한다.
    RobotVec3_t base[ROBOT_LEG_COUNT];            // 기본 발 위치를 저장한다.
    uint32_t leg;                                 // 시험할 다리 번호를 저장한다.

    if ((tolerance_m <= 0.0f) || (tolerance_rad <= 0.0f))
    {
        return false;
    }

    LegKinematics_Init(&kinematics);  // IK 유지값을 초기화한다.
    LegKinematics_GetBaseFeet(base);  // 여섯 기본 발 위치를 읽는다.
    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t local;                                  // 다리 좌표 위치를 저장한다.
        RobotVec3_t body_roundtrip;                         // Body 왕복 위치를 저장한다.
        RobotVec3_t fk;                                     // FK 위치를 저장한다.
        float angle[ROBOT_JOINTS_PER_LEG];                  // IK 관절각을 저장한다.
        const bool transform_ok = LegKinematics_BodyToLeg(
            (uint8_t)leg, &base[leg], &local) &&
            LegKinematics_LegToBody((uint8_t)leg, &local, &body_roundtrip);  // 좌표변환 왕복을 수행한다.

        if (!transform_ok ||
            (fabsf(body_roundtrip.x - base[leg].x) > tolerance_m) ||
            (fabsf(body_roundtrip.y - base[leg].y) > tolerance_m) ||
            (fabsf(body_roundtrip.z - base[leg].z) > tolerance_m))
        {
            return false;
        }

        if (!LegKinematics_Inverse(&kinematics, (uint8_t)leg,
                                   &base[leg], angle) ||
            !LegKinematics_Forward((uint8_t)leg, angle, &fk))
        {
            return false;
        }
        if ((fabsf(fk.x - base[leg].x) > tolerance_m) ||
            (fabsf(fk.y - base[leg].y) > tolerance_m) ||
            (fabsf(fk.z - base[leg].z) > tolerance_m))
        {
            return false;
        }
        if ((angle[0] < ROBOT_JOINT_MIN_RAD - tolerance_rad) ||
            (angle[0] > ROBOT_JOINT_MAX_RAD + tolerance_rad) ||
            (angle[1] < ROBOT_JOINT_MIN_RAD - tolerance_rad) ||
            (angle[1] > ROBOT_JOINT_MAX_RAD + tolerance_rad) ||
            (angle[2] < ROBOT_JOINT_MIN_RAD - tolerance_rad) ||
            (angle[2] > ROBOT_JOINT_MAX_RAD + tolerance_rad))
        {
            return false;
        }
    }

    return true;
}
