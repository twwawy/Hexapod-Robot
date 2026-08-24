#include "common/robot_types.h"
#include "high_control/body_posture_controller.h"
#include "high_control/drone_controller.h"
#include "high_control/foot_trajectory.h"
#include "high_control/gait_manager.h"
#include "high_control/gait_pose_controller.h"
#include "high_control/leg_kinematics.h"
#include "high_control/workspace_limiter.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    float target_vx_mps;
    float target_wz_radps;
    float target_roll_rad;
    float target_pitch_rad;
    float body_position_world[3];
    float attitude_rad[3];
    uint8_t foot_contact[ROBOT_LEG_COUNT];
} FirmwareControllerInput;

typedef struct
{
    float joint_angle_rad[ROBOT_JOINT_COUNT];
    float foot_target_body[ROBOT_LEG_COUNT * 3U];
    float applied_twist[4];
    float gait_progress[ROBOT_LEG_COUNT];
    uint8_t gait_state[ROBOT_LEG_COUNT];
    uint8_t ik_valid[ROBOT_LEG_COUNT];
    uint8_t gait_enabled;
    uint8_t gait_accepted;
    uint8_t posture_accepted;
} FirmwareControllerOutput;

typedef struct
{
    DroneController_Handle_t drone;
    GaitPoseController_Handle_t gait_pose;
    WorkspaceLimiter_Handle_t workspace;
    GaitManager_Handle_t gait;
    FootTrajectory_Handle_t trajectory;
    BodyPostureController_Handle_t posture;
    LegKinematics_Handle_t kinematics;
    RobotGaitPhase_t gait_phase;
    float previous_joint_rad[ROBOT_JOINT_COUNT];
    bool first_step;
} FirmwareController;

static float Bridge_Clamp(float value, float minimum, float maximum)
{
    return fminf(fmaxf(value, minimum), maximum);
}

static int16_t Bridge_CommandToRaw(float value, float maximum, int16_t deadband)
{
    float normalized;
    float raw;

    if ((maximum <= 0.0f) || (fabsf(value) < 1.0e-6f))
    {
        return 0;
    }

    normalized = Bridge_Clamp(value / maximum, -1.0f, 1.0f);
    raw = copysignf((float)deadband + fabsf(normalized) * (1000.0f - (float)deadband),
                    normalized);
    return (int16_t)lroundf(raw);
}

static bool Bridge_SeedBaseJoints(FirmwareController *controller)
{
    RobotVec3_t base[ROBOT_LEG_COUNT];
    uint32_t leg;
    bool valid = true;

    LegKinematics_GetBaseFeet(base);
    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const uint32_t first = leg * ROBOT_JOINTS_PER_LEG;
        valid = LegKinematics_Inverse(&controller->kinematics,
                                      (uint8_t)leg,
                                      &base[leg],
                                      &controller->previous_joint_rad[first]) && valid;
    }
    return valid;
}

void *FirmwareController_Create(void)
{
    FirmwareController *controller = calloc(1U, sizeof(*controller));

    if (controller == NULL)
    {
        return NULL;
    }

    DroneController_Init(&controller->drone);
    GaitPoseController_Init(&controller->gait_pose);
    WorkspaceLimiter_Init(&controller->workspace);
    GaitManager_Init(&controller->gait);
    FootTrajectory_Init(&controller->trajectory);
    BodyPostureController_Init(&controller->posture);
    LegKinematics_Init(&controller->kinematics);
    controller->first_step = true;

    if (!Bridge_SeedBaseJoints(controller))
    {
        free(controller);
        return NULL;
    }
    return controller;
}

void FirmwareController_Destroy(void *opaque)
{
    free(opaque);
}

int FirmwareController_Step(void *opaque,
                            const FirmwareControllerInput *input,
                            FirmwareControllerOutput *output)
{
    FirmwareController *controller = (FirmwareController *)opaque;
    RobotPriorityOutput_t priority;
    RobotDroneOutput_t drone;
    GaitPoseController_Output_t gait_pose;
    RobotBodyTwist_t twist;
    RobotFootTargets_t feet;
    RobotVec3_t feet_for_posture[ROBOT_LEG_COUNT];
    BodyPostureController_Output_t posture;
    RobotVec3_t position;
    RobotEuler_t attitude;
    bool contact[ROBOT_LEG_COUNT];
    bool gait_accepted = false;
    uint32_t leg;

    if ((controller == NULL) || (input == NULL) || (output == NULL))
    {
        return 0;
    }

    memset(output, 0, sizeof(*output));
    memset(&priority, 0, sizeof(priority));
    priority.active_mode = ROBOT_MODE_MANUAL;
    priority.throttle = Bridge_CommandToRaw(input->target_vx_mps,
                                             ROBOT_MAX_LINEAR_SPEED_MPS,
                                             ROBOT_THROTTLE_DEADBAND);
    priority.yaw = Bridge_CommandToRaw(input->target_wz_radps,
                                       ROBOT_MAX_YAW_RATE_RADPS,
                                       ROBOT_STICK_DEADBAND);
    priority.roll = Bridge_CommandToRaw(input->target_roll_rad,
                                        ROBOT_MAX_ROLL_RAD,
                                        ROBOT_STICK_DEADBAND);
    priority.pitch = Bridge_CommandToRaw(input->target_pitch_rad,
                                         ROBOT_MAX_PITCH_RAD,
                                         ROBOT_STICK_DEADBAND);
    priority.reset_command = controller->first_step;

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        contact[leg] = input->foot_contact[leg] != 0U;
    }

    position.x = input->body_position_world[0];
    position.y = input->body_position_world[1];
    position.z = input->body_position_world[2];
    attitude.roll = input->attitude_rad[0];
    attitude.pitch = input->attitude_rad[1];
    attitude.yaw = input->attitude_rad[2];

    drone = DroneController_Step(&controller->drone,
                                 &priority,
                                 contact,
                                 attitude.yaw);
    gait_pose = GaitPoseController_Step(&controller->gait_pose,
                                        drone.reset_command,
                                        &drone,
                                        &position,
                                        ROBOT_LEG_COUNT,
                                        attitude.yaw);
    twist = WorkspaceLimiter_Gait(&controller->workspace,
                                  &gait_pose.twist,
                                  drone.manual_enable,
                                  &controller->posture.command_rad,
                                  drone.reset_command,
                                  &gait_accepted);
    controller->gait_phase = GaitManager_Step(&controller->gait,
                                               drone.tripod_enable,
                                               drone.tripod_mode,
                                               drone.recovery_progress,
                                               contact);
    feet = FootTrajectory_Step(&controller->trajectory,
                               &twist,
                               &drone,
                               &controller->gait_phase,
                               &controller->posture.command_rad);
    memcpy(feet_for_posture, feet.foot, sizeof(feet_for_posture));
    posture = BodyPostureController_Step(&controller->posture,
                                         feet_for_posture,
                                         &drone,
                                         &attitude,
                                         drone.reset_command);

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const uint32_t first = leg * ROBOT_JOINTS_PER_LEG;
        RobotVec3_t limited = posture.targets.foot[leg];
        float candidate[ROBOT_JOINTS_PER_LEG];
        bool was_limited = false;
        const bool limit_valid = LegKinematics_LimitFoot((uint8_t)leg,
                                                         &posture.targets.foot[leg],
                                                         &limited,
                                                         &was_limited);
        const bool ik_valid = limit_valid &&
            LegKinematics_Inverse(&controller->kinematics,
                                  (uint8_t)leg,
                                  &limited,
                                  candidate);
        uint32_t joint;

        output->ik_valid[leg] = ik_valid ? 1U : 0U;
        output->gait_state[leg] = (uint8_t)controller->gait_phase.state[leg];
        output->gait_progress[leg] = controller->gait_phase.progress[leg];
        output->foot_target_body[first] = limited.x;
        output->foot_target_body[first + 1U] = limited.y;
        output->foot_target_body[first + 2U] = limited.z;

        for (joint = 0U; joint < ROBOT_JOINTS_PER_LEG; ++joint)
        {
            const uint32_t index = first + joint;
            const float desired = ik_valid ? candidate[joint]
                                           : controller->previous_joint_rad[index];
            const float delta = Bridge_Clamp(desired - controller->previous_joint_rad[index],
                                              -ROBOT_JOINT_STEP_RAD,
                                              ROBOT_JOINT_STEP_RAD);
            controller->previous_joint_rad[index] = Bridge_Clamp(
                controller->previous_joint_rad[index] + delta,
                ROBOT_JOINT_MIN_RAD,
                ROBOT_JOINT_MAX_RAD);
            output->joint_angle_rad[index] = controller->previous_joint_rad[index];
        }
    }

    output->applied_twist[0] = twist.vx;
    output->applied_twist[1] = twist.vy;
    output->applied_twist[2] = twist.vz;
    output->applied_twist[3] = twist.wz;
    output->gait_enabled = controller->gait_phase.enabled_internal ? 1U : 0U;
    output->gait_accepted = gait_accepted ? 1U : 0U;
    output->posture_accepted = posture.accepted ? 1U : 0U;
    controller->first_step = false;
    return 1;
}
