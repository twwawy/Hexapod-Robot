#include "common/robot_calibration.h"

#include "common/robot_config.h"

#include <math.h>
#include <stddef.h>

#define DEFAULT_SERVO_PULSE_PER_RAD (2000.0f / (270.0f * ROBOT_DEG_TO_RAD_F))
#define SERVO_CALIBRATION(angle_deg, direction_sign) \
    {1500U, 500U, 2500U, (angle_deg) * ROBOT_DEG_TO_RAD_F, \
     DEFAULT_SERVO_PULSE_PER_RAD, (direction_sign), true}
#define JOINT_CALIBRATION(raw_min_value, raw_zero_value, raw_max_value, direction_sign) \
    {(raw_min_value), (raw_zero_value), (raw_max_value), \
     (-20.0f * ROBOT_DEG_TO_RAD_F), 0.0f, (20.0f * ROBOT_DEG_TO_RAD_F), \
     (direction_sign), true}

const RobotCalibration_t g_robot_calibration =
{
    .imu =
    {
        .acceleration_sign = {1, 1, 1},        // WT931에서 설정한 X·Y·Z 방향을 그대로 사용한다.
        .angular_velocity_sign = {1, 1, 1},    // WT931에서 설정한 회전 방향을 그대로 사용한다.
        .euler_angle_sign = {1, 1, 1},         // 확인한 Roll·Pitch·Yaw 부호를 그대로 사용한다.
        .euler_offset_rad = {0.0f, 0.0f, 0.0f} // WT931에서 설정한 자세 영점을 그대로 사용한다.
    },
    .imu_calibrated = true,                   // WT931 축·부호·영점 확인 완료를 표시한다.

    .adc =
    {
        {{0U, 0U}, {0U, 1U}, {0U, 2U}, {0U, 3U}},  // 실측한 다리 1의 J1·J2·J3·압력 채널을 저장한다.
        {{0U, 4U}, {0U, 5U}, {0U, 6U}, {0U, 7U}},  // 실측한 다리 2의 J1·J2·J3·압력 채널을 저장한다.
        {{1U, 0U}, {1U, 1U}, {1U, 2U}, {1U, 3U}},  // 실측한 다리 3의 J1·J2·J3·압력 채널을 저장한다.
        {{1U, 4U}, {1U, 5U}, {1U, 6U}, {1U, 7U}},  // 실측한 다리 4의 J1·J2·J3·압력 채널을 저장한다.
        {{2U, 0U}, {2U, 1U}, {2U, 2U}, {2U, 3U}},  // 실측한 다리 5의 J1·J2·J3·압력 채널을 저장한다.
        {{2U, 4U}, {2U, 5U}, {2U, 6U}, {2U, 7U}}   // 실측한 다리 6의 J1·J2·J3·압력 채널을 저장한다.
    },
    .adc_mapping_calibrated = true,  // ADC 24채널 배선 확인 완료를 표시한다.

    .joint =
    {
        JOINT_CALIBRATION( 91U, 104U, 118U, -1),  // L1 J1의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 98U, 112U, 126U,  1),  // L1 J2의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION(106U, 121U, 133U, -1),  // L1 J3의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION(100U, 113U, 127U, -1),  // L2 J1의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 96U, 109U, 123U,  1),  // L2 J2의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION(101U, 115U, 129U, -1),  // L2 J3의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION(101U, 114U, 129U, -1),  // L3 J1의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION(100U, 113U, 127U,  1),  // L3 J2의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 93U, 106U, 119U, -1),  // L3 J3의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 99U, 113U, 126U, -1),  // L4 J1의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 97U, 111U, 125U, -1),  // L4 J2의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 94U, 108U, 122U,  1),  // L4 J3의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 92U, 106U, 119U, -1),  // L5 J1의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION(103U, 117U, 131U, -1),  // L5 J2의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 99U, 112U, 125U,  1),  // L5 J3의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 91U, 105U, 118U, -1),  // L6 J1의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION(105U, 118U, 133U, -1),  // L6 J2의 ADC 보정값과 방향을 저장한다.
        JOINT_CALIBRATION( 97U, 111U, 124U,  1)   // L6 J3의 ADC 보정값과 방향을 저장한다.
    },

    .pressure =
    {
        { 46U, 15U, true, true},  // 다리 1의 30% 접촉·10% 해제값을 저장한다.
        {120U, 40U, true, true},  // 다리 2의 30% 접촉·10% 해제값을 저장한다.
        { 44U, 14U, true, true},  // 다리 3의 30% 접촉·10% 해제값을 저장한다.
        { 31U, 11U, true, true},  // 다리 4의 30% 접촉·10% 해제값을 저장한다.
        { 51U, 17U, true, true},  // 다리 5의 30% 접촉·10% 해제값을 저장한다.
        { 29U,  9U, true, true}   // 다리 6의 30% 접촉·10% 해제값을 저장한다.
    },

    .servo =
    {
        SERVO_CALIBRATION(-12.0f,  1),  // L1 J1의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  7.0f, -1),  // L1 J2의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  8.0f,  1),  // L1 J3의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  6.0f,  1),  // L2 J1의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  9.0f, -1),  // L2 J2의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  1.0f,  1),  // L2 J3의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION( -2.0f,  1),  // L3 J1의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  4.0f, -1),  // L3 J2의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION( -7.0f,  1),  // L3 J3의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  0.0f,  1),  // L4 J1의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION( -9.0f,  1),  // L4 J2의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION( 11.0f, -1),  // L4 J3의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(-12.0f,  1),  // L5 J1의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  1.0f,  1),  // L5 J2의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION( -3.0f, -1),  // L5 J3의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(-10.0f,  1),  // L6 J1의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  4.0f,  1),  // L6 J2의 중립각과 방향을 저장한다.
        SERVO_CALIBRATION(  5.0f, -1)   // L6 J3의 중립각과 방향을 저장한다.
    },

    .relay_for_leg =
    {
        RELAY_INB1,  // 실측한 1번 다리 릴레이를 저장한다.
        RELAY_INC1,  // 실측한 2번 다리 릴레이를 저장한다.
        RELAY_INA1,  // 실측한 3번 다리 릴레이를 저장한다.
        RELAY_INB2,  // 실측한 4번 다리 릴레이를 저장한다.
        RELAY_INA2,  // 실측한 5번 다리 릴레이를 저장한다.
        RELAY_INC2   // 실측한 6번 다리 릴레이를 저장한다.
    },
    .relay_mapped = {true, true, true, true, true, true},  // 여섯 다리 대응 확인 완료를 표시한다.

    .crsf =
    {
        {174U, 992U, 1811U, 1, {0U, 1U, 2U}, true},  // CH1 Roll의 실측 범위와 방향을 저장한다.
        {174U, 992U, 1811U, 1, {0U, 1U, 2U}, true},  // CH2 Pitch의 실측 범위와 방향을 저장한다.
        {174U, 987U, 1811U, 1, {0U, 1U, 2U}, true},  // CH3 Throttle의 실측 범위와 방향을 저장한다.
        {174U, 992U, 1811U, 1, {0U, 1U, 2U}, true},  // CH4 Yaw의 실측 범위와 방향을 저장한다.
        {191U, 991U, 1792U, 1, {0U, 1U, 1U}, true},  // CH5 SA의 실측 범위와 OFF·ON 대응을 저장한다.
        {191U, 991U, 1792U, 1, {0U, 1U, 2U}, true},  // CH6 SB의 실측 범위와 세 위치 대응을 저장한다.
        {191U, 991U, 1792U, 1, {0U, 1U, 2U}, true},  // CH7 SC의 실측 범위와 세 위치 대응을 저장한다.
        {191U, 991U, 1792U, 1, {0U, 1U, 1U}, true},  // CH8 SD의 실측 범위와 OFF·ON 대응을 저장한다.
        {191U, 991U, 1792U, 1, {0U, 1U, 1U}, true},  // CH9 SE의 실측 범위와 해제·누름 대응을 저장한다.
        {191U, 991U, 1792U, 1, {0U, 0U, 1U}, true}   // CH10 S1의 왼쪽·중앙·오른쪽 이동 방식을 저장한다.
    }
};

/* 한 부호 값이 유효한지 확인한다. */
static bool RobotCalibration_IsSignValid(int8_t sign)
{
    return (sign == 1) || (sign == -1);  // 두 방향만 허용한다.
}

/* 중앙 설정 테이블의 모든 실측 완료와 값 범위를 검사한다. */
bool RobotCalibration_IsComplete(const RobotCalibration_t *calibration)
{
    bool adc_used[MCP3008_DEVICE_COUNT][MCP3008_CHANNEL_COUNT] = {{false}};  // ADC 채널 중복을 검사한다.
    uint32_t axis;     // IMU 축 번호를 저장한다.
    uint32_t leg;      // 다리 번호를 저장한다.
    uint32_t input;    // 다리 입력 번호를 저장한다.
    uint32_t joint;    // 관절 번호를 저장한다.
    uint32_t channel;  // CRSF 채널 번호를 저장한다.

    if ((calibration == NULL) || !calibration->imu_calibrated ||
        !calibration->adc_mapping_calibrated)
    {
        return false;
    }

    for (axis = 0U; axis < 3U; ++axis)
    {
        if (!RobotCalibration_IsSignValid(calibration->imu.acceleration_sign[axis]) ||
            !RobotCalibration_IsSignValid(calibration->imu.angular_velocity_sign[axis]) ||
            !RobotCalibration_IsSignValid(calibration->imu.euler_angle_sign[axis]) ||
            !isfinite(calibration->imu.euler_offset_rad[axis]))
        {
            return false;
        }
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (!calibration->relay_mapped[leg] ||
            ((uint32_t)calibration->relay_for_leg[leg] >= (uint32_t)RELAY_CHANNEL_COUNT))
        {
            return false;
        }

        for (input = 0U; input < MCP3008_LEG_INPUT_COUNT; ++input)
        {
            const MCP3008_InputMapping_t *mapping = &calibration->adc[leg][input];  // 검사할 ADC 매핑을 선택한다.

            if ((mapping->device >= MCP3008_DEVICE_COUNT) ||
                (mapping->channel >= MCP3008_CHANNEL_COUNT) ||
                adc_used[mapping->device][mapping->channel])
            {
                return false;
            }
            adc_used[mapping->device][mapping->channel] = true;  // 사용한 채널을 표시한다.
        }
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        const JointFeedback_Calibration_t *feedback = &calibration->joint[joint];  // 관절센서 값을 선택한다.
        const ServoPwm_Calibration_t *servo = &calibration->servo[joint];           // 서보 값을 선택한다.

        if (!feedback->calibrated ||
            (feedback->raw_min >= feedback->raw_zero) ||
            (feedback->raw_zero >= feedback->raw_max) ||
            (feedback->angle_min_rad >= feedback->angle_zero_rad) ||
            (feedback->angle_zero_rad >= feedback->angle_max_rad) ||
            !RobotCalibration_IsSignValid(feedback->direction) ||
            !servo->calibrated ||
            (servo->minimum_us >= servo->neutral_us) ||
            (servo->neutral_us >= servo->maximum_us) ||
            (servo->pulse_per_rad <= 0.0f) ||
            !RobotCalibration_IsSignValid(servo->direction))
        {
            return false;
        }
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        const FootPressure_Calibration_t *pressure = &calibration->pressure[leg];  // 압력센서 값을 선택한다.

        if (!pressure->calibrated ||
            (pressure->active_high &&
             (pressure->release_threshold >= pressure->contact_threshold)) ||
            (!pressure->active_high &&
             (pressure->release_threshold <= pressure->contact_threshold)))
        {
            return false;
        }
    }

    for (channel = 0U; channel < USER_COMMAND_USED_CHANNELS; ++channel)
    {
        const UserCommand_ChannelCalibration_t *crsf = &calibration->crsf[channel];  // CRSF 값을 선택한다.

        if (!crsf->calibrated ||
            (crsf->raw_min >= crsf->raw_center) ||
            (crsf->raw_center >= crsf->raw_max) ||
            !RobotCalibration_IsSignValid(crsf->direction))
        {
            return false;
        }
    }

    return true;
}

/* 완료된 중앙 설정값을 장치별 실행 테이블에 적용한다. */
bool RobotCalibration_Apply(const RobotCalibration_t *calibration,
                            IMU_Handle_t *imu,
                            MCP3008_Handle_t *adc,
                            JointFeedback_Handle_t *joint,
                            FootPressure_Handle_t *pressure,
                            ServoPwm_Handle_t *servo,
                            UserCommand_Handle_t *user_command)
{
    uint32_t leg;      // 적용할 다리 번호를 저장한다.
    uint32_t input;    // 적용할 ADC 입력 번호를 저장한다.
    uint32_t index;    // 적용할 관절 또는 채널 번호를 저장한다.
    bool complete;     // 전체 실측 완료 여부를 저장한다.

    if ((calibration == NULL) || (imu == NULL) || (adc == NULL) ||
        (joint == NULL) || (pressure == NULL) || (servo == NULL) ||
        (user_command == NULL))
    {
        return false;
    }

    complete = RobotCalibration_IsComplete(calibration);  // 적용과 별도로 전체 완료를 확인한다.

    if (calibration->imu_calibrated)
    {
        IMU_SetCalibration(imu, &calibration->imu);  // 완료된 IMU 보정을 적용한다.
    }

    if (calibration->adc_mapping_calibrated)
    {
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            for (input = 0U; input < MCP3008_LEG_INPUT_COUNT; ++input)
            {
                (void)MCP3008_SetInputMapping(adc, (uint8_t)leg,
                                              (MCP3008_LegInput_t)input,
                                              &calibration->adc[leg][input]);  // 완료된 ADC 매핑을 적용한다.
            }
        }
    }

    for (index = 0U; index < ROBOT_JOINT_COUNT; ++index)
    {
        if (calibration->joint[index].calibrated)
        {
            (void)JointFeedback_SetCalibration(joint, (uint8_t)index,
                                               &calibration->joint[index]);  // 완료된 관절센서 값을 적용한다.
        }
        if (calibration->servo[index].calibrated)
        {
            (void)ServoPwm_SetCalibration(servo, (uint8_t)index,
                                         &calibration->servo[index]);       // 완료된 서보 값을 적용한다.
        }
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        if (calibration->pressure[leg].calibrated)
        {
            (void)FootPressure_SetCalibration(pressure, (uint8_t)leg,
                                              &calibration->pressure[leg]);  // 완료된 압력값을 적용한다.
        }
    }

    for (index = 0U; index < USER_COMMAND_USED_CHANNELS; ++index)
    {
        if (calibration->crsf[index].calibrated)
        {
            (void)UserCommand_SetCalibration(user_command, (uint8_t)index,
                                             &calibration->crsf[index]);  // 완료된 CRSF 값을 적용한다.
        }
    }

    return complete;
}
