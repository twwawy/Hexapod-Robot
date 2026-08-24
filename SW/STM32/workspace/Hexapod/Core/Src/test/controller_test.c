#include "test/controller_test.h"

#include "high_control/body_posture_controller.h"
#include "high_control/foot_trajectory.h"
#include "high_control/gait_manager.h"
#include "high_control/leg_kinematics.h"

#include <math.h>
#include <string.h>

/* 명시적인 READY 입력으로 발 궤적과 자세·IK 계산을 연결해 검사한다. */
bool ControllerTest_Run(void)
{
    GaitManager_Handle_t gait_manager;                 // Tripod 상태를 저장한다.
    FootTrajectory_Handle_t trajectory;                // 발 궤적 기억을 저장한다.
    BodyPostureController_Handle_t posture_controller; // 자세 제어 상태를 저장한다.
    LegKinematics_Handle_t kinematics;                 // IK 상태를 저장한다.
    RobotDroneOutput_t drone;                          // 명시적인 READY 제어 입력을 저장한다.
    RobotBodyTwist_t twist;                            // 명시적인 정지 Twist를 저장한다.
    RobotGaitPhase_t gait;                             // Tripod 결과를 저장한다.
    RobotFootTargets_t feet;                           // 발 궤적 결과를 저장한다.
    BodyPostureController_Output_t posture;            // 자세 적용 결과를 저장한다.
    RobotEuler_t measured;                             // 명시적인 수평 IMU 자세를 저장한다.
    bool contact[ROBOT_LEG_COUNT];                     // 명시적인 전체 접촉을 저장한다.
    uint32_t leg;                                      // IK를 검사할 다리 번호를 저장한다.

    memset(&drone, 0, sizeof(drone));      // READY 입력을 0으로 준비한다.
    memset(&twist, 0, sizeof(twist));      // 정지 Twist를 준비한다.
    memset(&measured, 0, sizeof(measured));// 수평 자세를 준비한다.
    memset(contact, 1, sizeof(contact));   // 모든 발 접촉을 명시한다.
    GaitManager_Init(&gait_manager);       // Tripod 상태를 초기화한다.
    FootTrajectory_Init(&trajectory);      // 발 기억을 초기화한다.
    BodyPostureController_Init(&posture_controller);  // 자세 명령을 초기화한다.
    LegKinematics_Init(&kinematics);       // IK 유지값을 초기화한다.

    gait = GaitManager_Step(&gait_manager, false,
                            ROBOT_TRIPOD_NORMAL, 0.0f, contact);  // 정지 Stance를 만든다.
    feet = FootTrajectory_Step(&trajectory, &twist, &drone,
                               &gait, &measured);                 // 기본 발 위치를 만든다.
    posture = BodyPostureController_Step(&posture_controller,
                                          feet.foot, &drone,
                                          &measured, true);       // 자세 Reset을 적용한다.
    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        float angle[ROBOT_JOINTS_PER_LEG];  // 한 다리 IK 결과를 저장한다.
        if (!LegKinematics_Inverse(&kinematics, (uint8_t)leg,
                                   &posture.targets.foot[leg], angle) ||
            !isfinite(angle[0]) || !isfinite(angle[1]) || !isfinite(angle[2]))
        {
            return false;
        }
    }

    return posture.targets.command_accepted;  // 기본 자세 후보가 채택됐는지 확인한다.
}
