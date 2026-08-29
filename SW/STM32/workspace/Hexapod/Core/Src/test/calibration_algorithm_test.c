#include "test/calibration_algorithm_test.h"

#include "common/robot_calibration.h"

#include <math.h>
#include <string.h>

#define CALIBRATION_TEST_ANGLE_TOLERANCE_RAD 0.0001f

/* 중앙 관절센서·압력·서보·IMU·ADC 값의 변환을 검사한다. */
bool CalibrationAlgorithmTest_Run(void)
{
    JointFeedback_Handle_t joint_feedback;                    // 시험용 관절센서 변환 상태를 저장한다.
    FootPressure_Handle_t pressure;                           // 시험용 접촉 변환 상태를 저장한다.
    IMU_Handle_t imu;                                         // 시험용 IMU 보정 상태를 저장한다.
    IMU_Calibration_t imu_result;                             // 다시 읽은 IMU 보정값을 저장한다.
    MCP3008_Handle_t adc;                                     // 시험용 ADC 매핑 상태를 저장한다.
    uint16_t joint_raw[ROBOT_JOINT_COUNT];                    // 관절센서 입력을 저장한다.
    float joint_angle[ROBOT_JOINT_COUNT];                     // 변환한 관절각을 저장한다.
    uint16_t pressure_raw[ROBOT_PRESSURE_COUNT];              // 압력센서 입력을 저장한다.
    bool contact[ROBOT_PRESSURE_COUNT];                       // 변환한 접촉 상태를 저장한다.
    uint32_t axis;                                            // IMU 축 번호를 저장한다.
    uint32_t leg;                                             // 다리 번호를 저장한다.
    uint32_t input;                                           // ADC 입력 번호를 저장한다.
    uint32_t joint;                                           // 관절 번호를 저장한다.
    uint32_t sample;                                          // 접촉 확인 표본을 저장한다.

    if (!RobotCalibration_IsComplete(&g_robot_calibration))
    {
        return false;
    }

    memset(&imu, 0, sizeof(imu));                    // HAL 없이 보정 저장만 시험한다.
    memset(&adc, 0, sizeof(adc));                    // HAL 없이 매핑 저장만 시험한다.
    JointFeedback_Init(&joint_feedback);             // 관절 변환 기본 상태를 준비한다.
    FootPressure_Init(&pressure);                    // 접촉 변환 기본 상태를 준비한다.
    IMU_SetCalibration(&imu, &g_robot_calibration.imu);  // 중앙 IMU 값을 적용한다.
    IMU_GetCalibration(&imu, &imu_result);               // 적용된 값을 다시 읽는다.

    for (axis = 0U; axis < 3U; ++axis)
    {
        if ((imu_result.acceleration_sign[axis] != g_robot_calibration.imu.acceleration_sign[axis]) ||
            (imu_result.angular_velocity_sign[axis] != g_robot_calibration.imu.angular_velocity_sign[axis]) ||
            (imu_result.euler_angle_sign[axis] != g_robot_calibration.imu.euler_angle_sign[axis]) ||
            (fabsf(imu_result.euler_offset_rad[axis] -
                   g_robot_calibration.imu.euler_offset_rad[axis]) > CALIBRATION_TEST_ANGLE_TOLERANCE_RAD))
        {
            return false;
        }
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        for (input = 0U; input < MCP3008_LEG_INPUT_COUNT; ++input)
        {
            if (!MCP3008_SetInputMapping(&adc, (uint8_t)leg,
                                         (MCP3008_LegInput_t)input,
                                         &g_robot_calibration.adc[leg][input]))
            {
                return false;
            }
            if (memcmp(&adc.mapping[leg][input], &g_robot_calibration.adc[leg][input],
                       sizeof(adc.mapping[leg][input])) != 0)
            {
                return false;
            }
        }
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        uint16_t pulse_us;  // 영점에서 계산한 서보 Pulse를 저장한다.

        if (!JointFeedback_SetCalibration(&joint_feedback, (uint8_t)joint,
                                          &g_robot_calibration.joint[joint]) ||
            !ServoPwm_CalculatePulse(&g_robot_calibration.servo[joint],
                                     g_robot_calibration.servo[joint].zero_angle_rad,
                                     &pulse_us) ||
            (pulse_us != g_robot_calibration.servo[joint].neutral_us))
        {
            return false;
        }
        joint_raw[joint] = g_robot_calibration.joint[joint].raw_zero;  // 실측 영점 ADC를 입력한다.
    }

    if (!JointFeedback_Convert(&joint_feedback, joint_raw, joint_angle))
    {
        return false;
    }
    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        const float expected = (g_robot_calibration.joint[joint].direction < 0) ?
            -g_robot_calibration.joint[joint].angle_zero_rad :
             g_robot_calibration.joint[joint].angle_zero_rad;  // 센서 방향을 적용한 영점각을 계산한다.

        if (fabsf(joint_angle[joint] - expected) > CALIBRATION_TEST_ANGLE_TOLERANCE_RAD)
        {
            return false;
        }
    }

    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        if (!FootPressure_SetCalibration(&pressure, (uint8_t)leg,
                                         &g_robot_calibration.pressure[leg]))
        {
            return false;
        }
        pressure_raw[leg] = g_robot_calibration.pressure[leg].contact_threshold;  // 접촉 진입값을 입력한다.
    }
    FootPressure_Update(&pressure, pressure_raw, contact);  // 1 ms 접촉 후보만 입력한다.
    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        if (!pressure.raw_contact[leg] || contact[leg])
        {
            return false;  // 순간 접촉이 확정 상태로 올라오면 실패한다.
        }
        pressure_raw[leg] = g_robot_calibration.pressure[leg].release_threshold;  // 후보 취소값을 입력한다.
    }
    FootPressure_Update(&pressure, pressure_raw, contact);  // 1 ms 접촉 후보를 취소한다.
    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        if (pressure.raw_contact[leg] || contact[leg])
        {
            return false;
        }
        pressure_raw[leg] = g_robot_calibration.pressure[leg].contact_threshold;  // 연속 접촉값을 다시 입력한다.
    }
    for (sample = 0U; sample < ROBOT_PRESSURE_CONTACT_CONFIRM_SAMPLES; ++sample)
    {
        FootPressure_Update(&pressure, pressure_raw, contact);  // 5 ms 연속 접촉을 입력한다.
    }
    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        if (!contact[leg])
        {
            return false;
        }
        pressure_raw[leg] = g_robot_calibration.pressure[leg].release_threshold;  // 접촉 해제값을 입력한다.
    }
    for (sample = 0U; sample < ROBOT_PRESSURE_RELEASE_CONFIRM_SAMPLES; ++sample)
    {
        FootPressure_Update(&pressure, pressure_raw, contact);  // 10 ms 연속 해제를 입력한다.
    }
    for (leg = 0U; leg < ROBOT_PRESSURE_COUNT; ++leg)
    {
        if (contact[leg])
        {
            return false;
        }
    }

    return true;
}
