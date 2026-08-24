#ifndef LEG_KINEMATICS_H
#define LEG_KINEMATICS_H

#include "common/robot_types.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct
{
    float last_angle_rad[ROBOT_JOINT_COUNT];  // 다리별 마지막 정상 IK 해를 저장한다.
} LegKinematics_Handle_t;

void LegKinematics_Init(LegKinematics_Handle_t *handle);  // IK 유지값을 영점으로 초기화한다.

void LegKinematics_GetBaseFeet(RobotVec3_t foot_body[ROBOT_LEG_COUNT]);  // 기본 발 위치를 반환한다.

bool LegKinematics_BodyToLeg(uint8_t leg,
                             const RobotVec3_t *body,
                             RobotVec3_t *local);  // Body 위치를 다리 좌표로 변환한다.

bool LegKinematics_LegToBody(uint8_t leg,
                             const RobotVec3_t *local,
                             RobotVec3_t *body);  // 다리 위치를 Body 좌표로 변환한다.

bool LegKinematics_Forward(uint8_t leg,
                           const float angle_rad[ROBOT_JOINTS_PER_LEG],
                           RobotVec3_t *foot_body);  // 관절각으로 Body 발 위치를 계산한다.

bool LegKinematics_Inverse(LegKinematics_Handle_t *handle,
                           uint8_t leg,
                           const RobotVec3_t *foot_body,
                           float angle_rad[ROBOT_JOINTS_PER_LEG]);  // Body 발 위치의 IK를 계산한다.

bool LegKinematics_IsReachable(uint8_t leg,
                               const RobotVec3_t *foot_body);  // 관절 제한을 포함한 IK 가능 여부를 반환한다.

bool LegKinematics_LimitFoot(uint8_t leg,
                             const RobotVec3_t *input_body,
                             RobotVec3_t *output_body,
                             bool *limited);  // 최종 발 위치를 0.0001 m 여유로 제한한다.

#endif
