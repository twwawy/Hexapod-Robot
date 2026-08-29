#include "app/hexapod_app.h"

#include "common/robot_calibration.h"
#include "high_control/stand_landing.h"
#include "low_control/relay.h"
#include "test/rc_command_generator.h"

#include <stddef.h>
#include <string.h>

extern SPI_HandleTypeDef hspi1;   // MCP3008 SPI Handle을 연결한다.
extern SPI_HandleTypeDef hspi2;   // Jetson SPI2 Slave Handle을 연결한다.
extern TIM_HandleTypeDef htim1;   // 1·6번 일부 서보 Timer를 연결한다.
extern TIM_HandleTypeDef htim2;   // 2번 서보 Timer를 연결한다.
extern TIM_HandleTypeDef htim3;   // 3·6번 일부 서보 Timer를 연결한다.
extern TIM_HandleTypeDef htim4;   // 4·6번 일부 서보 Timer를 연결한다.
extern TIM_HandleTypeDef htim5;   // 5번 일부 서보 Timer를 연결한다.
extern TIM_HandleTypeDef htim6;   // 1 ms 기준 Timer를 연결한다.
extern TIM_HandleTypeDef htim8;   // 5번 일부 서보 Timer를 연결한다.
extern UART_HandleTypeDef huart2; // GPS UART Handle을 연결한다.
extern UART_HandleTypeDef huart3; // WT931 UART Handle을 연결한다.
extern UART_HandleTypeDef huart6; // CRSF UART Handle을 연결한다.

HexapodApp_Handle_t g_hexapod_app;              // 최종 앱의 전체 실행 상태를 저장한다.
volatile bool g_robot_bringup_emergency_stop;  // 임시 조종기 시험 긴급정지를 저장한다.

static RcCommandGenerator_t simulated_rc;  // 2단계 중립 조종기 입력을 저장한다.
static uint32_t simulated_rc_start_ms;      // 임시 조종기 시작 시각을 저장한다.

/* 실수 값을 지정한 절댓값 범위로 제한한다. */
static float HexapodApp_ClampMagnitude(float value, float limit)
{
    if (value > limit)
    {
        return limit;
    }
    if (value < -limit)
    {
        return -limit;
    }
    return value;
}

/* 선택한 실기 시험 단계의 허가 범위를 준비한다. */
static void HexapodApp_InitializeBringup(HexapodApp_Handle_t *handle)
{
    handle->bringup.stage = ROBOT_BRINGUP_STAGE;                                      // 컴파일된 시험 단계를 저장한다.
    handle->bringup.neutral_output_active = (ROBOT_BRINGUP_STAGE == 2U);               // 2단계에서 0도를 고정한다.
    handle->bringup.stand_landing_allowed = (ROBOT_BRINGUP_STAGE >= 3U);               // 3단계부터 서기·착지를 허가한다.
    handle->bringup.correction_allowed = (ROBOT_BRINGUP_STAGE >= 4U);                  // 4단계부터 보정을 허가한다.
    handle->bringup.walking_allowed = (ROBOT_BRINGUP_STAGE >= 5U);                     // 5단계부터 보행을 허가한다.
    handle->bringup.simulated_rc_active = (ROBOT_BRINGUP_STAGE == 2U) ||
                                          (ROBOT_BRINGUP_STAGE == 3U);                  // 2·3단계에서 임시 조종기를 사용한다.
    handle->bringup.linear_limit_mps = (ROBOT_BRINGUP_STAGE == 5U) ?
                                       ROBOT_BRINGUP_LOW_SPEED_MPS :
                                       ROBOT_MAX_LINEAR_SPEED_MPS;                    // 5단계만 저속으로 제한한다.
    handle->bringup.yaw_limit_radps = (ROBOT_BRINGUP_STAGE == 5U) ?
                                      ROBOT_BRINGUP_LOW_YAW_RATE_RADPS :
                                      ROBOT_MAX_YAW_RATE_RADPS;                       // 5단계만 저속 회전으로 제한한다.
    handle->bringup.relay_enabled = false;                                             // 실제 릴레이 상태를 OFF로 시작한다.

    g_robot_bringup_emergency_stop = false;                                            // 임시 긴급정지를 해제한다.
    if (handle->bringup.simulated_rc_active)
    {
        RcCommandGenerator_Init(&simulated_rc);                                        // 연결된 중립 조종기를 만든다.
        simulated_rc_start_ms = HAL_GetTick();                                         // 1초 대기 시작 시각을 저장한다.
        if ((ROBOT_BRINGUP_STAGE == 3U) &&
            (ROBOT_BRINGUP_PRESSURE_CALIBRATION != 0U))
        {
            PressureLoadCalibration_Init(&handle->pressure_calibration,
                                         simulated_rc_start_ms);                       // 3단계 압력 자동 보정을 준비한다.
        }
    }
}

/* 2·3단계용 연결된 임시 조종기 명령을 갱신한다. */
static void HexapodApp_UpdateSimulatedRc(HexapodApp_Handle_t *handle, uint32_t now_ms)
{
    uint32_t elapsed_ms;  // 임시 조종기 경과시간을 저장한다.

    if ((handle == NULL) || !handle->bringup.simulated_rc_active)
    {
        return;
    }

    elapsed_ms = now_ms - simulated_rc_start_ms;                         // 자동 시험 경과시간을 계산한다.
    handle->user = RcCommandGenerator_Step(&simulated_rc);               // 모든 짐벌이 중립인 연결 명령을 만든다.
    handle->user.timestamp_ms = now_ms;                                  // 현재 임시 프레임 시각을 기록한다.
    handle->user.sd = g_robot_bringup_emergency_stop ? 1U : 0U;          // 디버거 긴급정지를 SD Kill로 바꾼다.
    handle->user.motion_armed =
        (elapsed_ms >= ROBOT_BRINGUP_FAKE_RC_DELAY_MS) &&
        !g_robot_bringup_emergency_stop;                                 // 1초 후에만 0도 출력을 허가한다.
    handle->bringup.simulated_rc_elapsed_ms = elapsed_ms;                 // Live Expressions에 경과시간을 표시한다.

    if (ROBOT_BRINGUP_STAGE == 2U)
    {
        handle->bringup.simulated_rc_phase = 1U;                          // 2단계 중립 유지 상태를 표시한다.
        return;
    }

    if (elapsed_ms < ROBOT_BRINGUP_STAGE3_STAND_MS)
    {
        handle->user.sb = 0U;                                            // 시작 자세에서 서기 요청을 해제한다.
        handle->bringup.simulated_rc_phase = 0U;                          // 초기 착지 대기를 표시한다.
        return;
    }

    if (handle->bringup.simulated_rc_phase == 0U)
    {
        handle->bringup.simulated_rc_phase = 1U;                          // 자동 서기 구간을 표시한다.
    }
    if ((handle->bringup.simulated_rc_phase == 1U) &&
        (((ROBOT_BRINGUP_PRESSURE_CALIBRATION != 0U) &&
          PressureLoadCalibration_IsFinished()) ||
         ((ROBOT_BRINGUP_PRESSURE_CALIBRATION == 0U) &&
          (elapsed_ms >= ROBOT_BRINGUP_STAGE3_LANDING_MS))))
    {
        handle->bringup.simulated_rc_phase = 2U;                          // 보정 후 자동 착지를 시작한다.
    }

    handle->user.sb = (handle->bringup.simulated_rc_phase == 1U) ?
                      1U : 0U;                                          // 보정 완료 전까지만 서기를 유지한다.
    if ((handle->bringup.simulated_rc_phase == 2U) &&
        handle->drone.landing_done)
    {
        handle->bringup.simulated_rc_phase = 3U;                          // 자동 착지 완료를 표시한다.
    }
}

/* 시험 단계에서 아직 허용하지 않은 조종 명령을 제거한다. */
static RobotUserCommand_t HexapodApp_LimitUserCommand(const HexapodApp_Handle_t *handle)
{
    RobotUserCommand_t limited = handle->user;  // 실제 조종 명령을 복사한다.

    if (!handle->bringup.stand_landing_allowed)
    {
        limited.throttle = 0;  // 이동 명령을 제거한다.
        limited.yaw = 0;       // 회전 명령을 제거한다.
        limited.roll = 0;      // Roll 명령을 제거한다.
        limited.pitch = 0;     // Pitch 명령을 제거한다.
        limited.sb = 0U;       // 자동 서기를 차단한다.
        limited.sc = 0U;       // Reset 명령을 차단한다.
        limited.se = 1U;       // 수동·보정 모드를 차단한다.
    }
    else if (!handle->bringup.correction_allowed)
    {
        limited.throttle = 0;  // 서기 시험 중 이동 명령을 제거한다.
        limited.yaw = 0;       // 서기 시험 중 회전 명령을 제거한다.
        limited.roll = 0;      // 서기 시험 중 Roll 명령을 제거한다.
        limited.pitch = 0;     // 서기 시험 중 Pitch 명령을 제거한다.
        limited.sc = 0U;       // Reset 명령을 차단한다.
        limited.se = 1U;       // 수동·보정 모드를 차단한다.
    }
    else if (!handle->bringup.walking_allowed && (limited.sc != 2U))
    {
        limited.se = 1U;       // 4단계에서 수동 모드를 READY로 바꾼다.
        limited.throttle = 0;  // READY의 잔류 이동 명령을 제거한다.
        limited.yaw = 0;       // READY의 잔류 회전 명령을 제거한다.
        limited.roll = 0;      // READY의 잔류 Roll 명령을 제거한다.
        limited.pitch = 0;     // READY의 잔류 Pitch 명령을 제거한다.
    }

    return limited;
}

/* 저속 보행 단계의 선속도와 회전속도를 제한한다. */
static void HexapodApp_LimitDroneCommand(const HexapodApp_Handle_t *handle,
                                         RobotDroneOutput_t *drone)
{
    if ((handle == NULL) || (drone == NULL) || !handle->bringup.walking_allowed)
    {
        return;
    }

    drone->vx_user_mps = HexapodApp_ClampMagnitude(
        drone->vx_user_mps, handle->bringup.linear_limit_mps);  // 전후 속도를 현재 단계로 제한한다.
    drone->vy_user_mps = HexapodApp_ClampMagnitude(
        drone->vy_user_mps, handle->bringup.linear_limit_mps);  // 횡이동 속도를 현재 단계로 제한한다.
    drone->wz_user_radps = HexapodApp_ClampMagnitude(
        drone->wz_user_radps, handle->bringup.yaw_limit_radps); // 회전속도를 현재 단계로 제한한다.
}

/* 초기 관절 명령을 기본 발 위치의 정상 IK로 만든다. */
static bool HexapodApp_InitializeJointCommand(HexapodApp_Handle_t *handle)
{
    RobotVec3_t base[ROBOT_LEG_COUNT];  // 기본 발 위치를 저장한다.
    uint32_t leg;                       // IK를 계산할 다리 번호를 저장한다.
    bool all_valid = true;              // 전체 IK 결과를 저장한다.

    LegKinematics_GetBaseFeet(base);  // 기본 발 위치를 읽는다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const uint32_t first = leg * ROBOT_JOINTS_PER_LEG;  // 다리의 첫 관절 위치를 계산한다.
        handle->joints.ik_valid[leg] = LegKinematics_Inverse(
            &handle->kinematics,
            (uint8_t)leg,
            &base[leg],
            &handle->joints.angle_rad[first]);  // 기본 위치의 IK를 계산한다.
        all_valid = all_valid && handle->joints.ik_valid[leg];  // 전체 결과를 누적한다.
    }

    return all_valid;
}

/* 최초 제어기 Fault의 다리와 발 목표를 영구 기록한다. */
static void HexapodApp_RecordControllerFault(
    HexapodApp_Handle_t *handle,
    RobotControllerFaultReason_t reason,
    uint8_t leg,
    const RobotVec3_t *target_body,
    const RobotVec3_t *limited_body,
    bool was_limited)
{
    if ((handle == NULL) || handle->controller_fault.valid)
    {
        return;
    }

    memset(&handle->controller_fault, 0,
           sizeof(handle->controller_fault));                        // 최초 Fault 기록을 준비한다.
    handle->controller_fault.reason = reason;                        // 최초 실패 원인을 저장한다.
    handle->controller_fault.leg = leg;                              // 최초 실패 다리를 저장한다.
    handle->controller_fault.control_count = handle->control_count;  // 최초 실패 제어 주기를 저장한다.
    handle->controller_fault.was_limited = was_limited;              // 실패 전 제한 여부를 저장한다.
    if (target_body != NULL)
    {
        handle->controller_fault.target_body = *target_body;     // 제한 전 발 목표를 저장한다.
    }
    if (limited_body != NULL)
    {
        handle->controller_fault.limited_body = *limited_body;   // 제한 후 발 목표를 저장한다.
    }
    handle->controller_fault.valid = true;                       // 최초 Fault 기록을 확정한다.
}

/* 현재 관절 명령을 초기 영점 정렬 속도로 0도에 접근시킨다. */
static float HexapodApp_MoveStartupAngleToZero(float angle_rad)
{
    const float maximum_step = ROBOT_STARTUP_ZERO_RATE_RADPS *
                               ROBOT_CONTROL_PERIOD_S;  // 한 주기 최대 이동각을 계산한다.

    if (angle_rad > maximum_step)
    {
        return angle_rad - maximum_step;
    }
    if (angle_rad < -maximum_step)
    {
        return angle_rad + maximum_step;
    }
    return 0.0f;
}

/* PWM을 끈 채 초기 관절각을 측정하고 영점 정렬을 준비한다. */
static void HexapodApp_PrepareStartupZero(HexapodApp_Handle_t *handle,
                                          RobotUserCommand_t *limited_user,
                                          bool sensor_updated)
{
    bool start_requested;  // 안전한 서기 요청 여부를 저장한다.
    uint32_t joint;        // 측정할 관절 번호를 저장한다.

    if ((handle == NULL) || (limited_user == NULL) ||
        (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_COMPLETE))
    {
        return;
    }

    start_requested = limited_user->connected &&
                      limited_user->motion_armed &&
                      (limited_user->sb == 1U) &&
                      (limited_user->sd == 0U) &&
                      !handle->safety.rollover_fault &&
                      !handle->safety.controller_fault;  // 연결·서기·안전 조건을 함께 검사한다.

    if (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_FAULT)
    {
        limited_user->sb = 0U;  // 서기 요청을 차단한다.
        limited_user->sd = 1U;  // PWM 시작 실패를 Kill로 유지한다.
        return;
    }

    if (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_WAIT)
    {
        if (start_requested)
        {
            memset(handle->startup_angle_sum_rad,
                   0,
                   sizeof(handle->startup_angle_sum_rad));             // 초기각 합계를 초기화한다.
            handle->startup_sensor_settle_time_s = 0.0f;                // 전원 안정 시간을 초기화한다.
            handle->startup_sensor_sample_count = 0U;                   // 초기각 측정 횟수를 초기화한다.
            handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_CAPTURE; // 센서 전원과 측정을 시작한다.
        }
        limited_user->sb = 0U;  // 초기각 측정 전 기존 서기 진입을 막는다.
    }
    else if (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_CAPTURE)
    {
        if (!start_requested)
        {
            ServoPwm_Stop(&handle->servo_pwm);                       // 취소 시 PWM을 정지한다.
            handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_WAIT;  // 전원 차단 대기로 돌아간다.
        }
        else if (handle->startup_sensor_settle_time_s < ROBOT_STARTUP_SENSOR_SETTLE_S)
        {
            handle->startup_sensor_settle_time_s += ROBOT_CONTROL_PERIOD_S;  // 센서 전원 안정 시간을 누적한다.
        }
        else if (sensor_updated)
        {
            for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
            {
                handle->startup_angle_sum_rad[joint] +=
                    handle->sensor_snapshot.joint_angle_rad[joint];  // 실측 관절각을 누적한다.
            }
            handle->startup_sensor_sample_count++;  // 유효한 ADC 측정을 기록한다.

            if (handle->startup_sensor_sample_count >= ROBOT_STARTUP_SENSOR_SAMPLES)
            {
                for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
                {
                    handle->startup_initial_angle_rad[joint] = HexapodApp_ClampMagnitude(
                        handle->startup_angle_sum_rad[joint] /
                            (float)handle->startup_sensor_sample_count,
                        ROBOT_JOINT_MAX_RAD);  // 평균 초기각을 관절 범위로 제한한다.
                    handle->startup_command_angle_rad[joint] =
                        handle->startup_initial_angle_rad[joint];  // 실측각을 첫 PWM 명령으로 사용한다.
                }

                if (ServoPwm_StartAngles(&handle->servo_pwm,
                                         handle->startup_command_angle_rad) == HAL_OK)
                {
                    handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_MOVE;  // 실측각에서 영점 정렬을 시작한다.
                }
                else
                {
                    ServoPwm_Stop(&handle->servo_pwm);                        // 부분 PWM 출력을 정리한다.
                    handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_FAULT; // 재시작 실패를 유지한다.
                    limited_user->sd = 1U;                                  // 릴레이 차단을 요청한다.
                }
            }
        }
        limited_user->sb = 0U;  // PWM 시작과 영점 정렬 전 기존 서기 진입을 막는다.
    }
    else if (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_MOVE)
    {
        if (!start_requested)
        {
            ServoPwm_Stop(&handle->servo_pwm);                       // 릴레이 차단 전 PWM을 정지한다.
            handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_WAIT;  // 요청 해제 시 다시 측정한다.
        }
        limited_user->sb = 0U;  // 영점 이동 중 기존 서기 진입을 막는다.
    }
    else if ((handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_READY) &&
             !start_requested)
    {
        ServoPwm_Stop(&handle->servo_pwm);                       // 릴레이 차단 전 PWM을 정지한다.
        handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_WAIT;  // 서기 취소 시 초기각 대기로 돌아간다.
        limited_user->sb = 0U;                                  // 취소된 서기 요청을 제거한다.
    }
}

/* 초기 영점 상태에 맞는 관절 명령을 만들고 완료를 판정한다. */
static void HexapodApp_ApplyStartupZero(HexapodApp_Handle_t *handle)
{
    bool all_zero = true;  // 전체 관절 영점 도달 여부를 저장한다.
    uint32_t joint;        // 갱신할 관절 번호를 저장한다.

    if ((handle == NULL) ||
        (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_COMPLETE))
    {
        return;
    }

    if (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_MOVE)
    {
        for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
        {
            handle->startup_command_angle_rad[joint] =
                HexapodApp_MoveStartupAngleToZero(
                    handle->startup_command_angle_rad[joint]);  // 관절별 명령을 0도에 접근시킨다.
            all_zero = all_zero &&
                       (handle->startup_command_angle_rad[joint] == 0.0f);  // 전체 영점 도달을 누적한다.
        }

        if (all_zero)
        {
            handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_READY;  // 다음 주기에 기존 서기를 허가한다.
        }
    }

    memcpy(handle->joints.angle_rad,
           handle->startup_command_angle_rad,
           sizeof(handle->joints.angle_rad));  // 대기·이동·완료 직전 명령을 최종 PWM에 적용한다.
}

/* 완전 착지 후 PWM을 끈 다음 초기각 측정을 재무장한다. */
static void HexapodApp_RearmStartupZero(HexapodApp_Handle_t *handle)
{
    const bool landing_complete =
        (handle != NULL) &&
        (handle->priority.active_mode == ROBOT_MODE_LANDING) &&
        handle->drone.landing_done;  // 정상 착지 완료를 검사한다.

    if ((handle == NULL) || (ROBOT_BRINGUP_STAGE < 3U) ||
        (handle->startup_zero_state != HEXAPOD_STARTUP_ZERO_COMPLETE) ||
        !landing_complete)
    {
        return;
    }

    ServoPwm_Stop(&handle->servo_pwm);                       // 릴레이 차단 전 PWM을 정지한다.
    handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_WAIT;  // 다음 서기의 초기각 측정을 기다린다.
}

/* 실제 장치 드라이버와 모든 제어 상태를 초기화한다. */
HAL_StatusTypeDef HexapodApp_Init(HexapodApp_Handle_t *handle,
                                  const HexapodApp_Hardware_t *hardware)
{
    HAL_StatusTypeDef status;  // 장치 초기화 결과를 저장한다.
    float neutral_angle_rad[ROBOT_JOINT_COUNT] = {0.0f};  // 2단계 Rate Limit 시작각을 저장한다.

    if ((handle == NULL) || (hardware == NULL) ||
        (hardware->gps_uart == NULL) || (hardware->imu_uart == NULL) ||
        (hardware->crsf_uart == NULL) || (hardware->adc_spi == NULL) ||
        (hardware->control_timer == NULL))
    {
        return HAL_ERROR;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 실행 상태를 제거한다.
    handle->hardware = *hardware;        // CubeMX Handle 연결을 저장한다.
    HexapodApp_InitializeBringup(handle); // 단계별 출력 허가를 준비한다.
    handle->startup_zero_state = (ROBOT_BRINGUP_STAGE >= 3U) ?
                                 HEXAPOD_STARTUP_ZERO_WAIT :
                                 HEXAPOD_STARTUP_ZERO_COMPLETE;  // 각 서기 전 초기 영점 정렬을 요구한다.
    ControlTimingDebug_Init();           // 5 ms 제어 시간 측정을 준비한다.

    GPS_Init(&handle->gps, hardware->gps_uart);             // GPS 드라이버를 준비한다.
    IMU_Init(&handle->imu, hardware->imu_uart);             // WT931 드라이버를 준비한다.
    status = MCP3008_Init(&handle->adc, hardware->adc_spi); // ADC 드라이버를 준비한다.
    if (status != HAL_OK)
    {
        return status;
    }

    SensorManager_Init(&handle->sensors, &handle->gps,
                       &handle->imu, &handle->adc);          // 실제 센서를 한 스냅샷으로 연결한다.
    CRSF_Receiver_Init(&handle->crsf_receiver,
                       hardware->crsf_uart);                 // CRSF UART를 준비한다.
    CRSF_Protocol_Init(&handle->crsf_protocol);              // CRSF 프레임 해석기를 준비한다.
    UserCommand_Init(&handle->user_command);                 // 조종기 기본 보정표를 준비한다.

    ControlPriority_Init(&handle->priority_control);         // 운용 상태를 착지로 초기화한다.
    DroneController_Init(&handle->drone_control);            // 조종 제어 상태를 초기화한다.
    BodyPositionEstimator_Init(&handle->position_estimator); // FK 추정 상태를 초기화한다.
    GaitPoseController_Init(&handle->gait_pose_control);     // 보행 PI 상태를 초기화한다.
    WorkspaceLimiter_Init(&handle->workspace_limiter);       // 작업공간 기억을 초기화한다.
    GaitManager_Init(&handle->gait_manager);                 // Tripod 위상을 초기화한다.
    FootTrajectory_Init(&handle->foot_trajectory);           // 발 궤적 기억을 초기화한다.
    BodyPostureController_Init(&handle->posture_control);    // 자세 PI 상태를 초기화한다.
    LegKinematics_Init(&handle->kinematics);                 // IK 유지값을 초기화한다.
    Safety_Init(&handle->safety_control);                    // Fault Latch를 초기화한다.

    if (!HexapodApp_InitializeJointCommand(handle))
    {
        return HAL_ERROR;
    }

    ServoPwm_Init(&handle->servo_pwm, &hardware->servo_timers);  // 18개 서보 채널을 배치한다.
    if (!RobotCalibration_Apply(&g_robot_calibration,
                                &handle->imu,
                                &handle->adc,
                                &handle->sensors.joints,
                                &handle->sensors.pressure,
                                &handle->servo_pwm,
                                &handle->user_command))           // 중앙 실측값을 모든 장치에 적용한다.
    {
        return HAL_ERROR;
    }
    Relay_Init();  // 서보 전원을 꺼진 상태로 준비한다.

    if (ROBOT_BRINGUP_STAGE < 3U)
    {
        status = ServoPwm_Start(&handle->servo_pwm);  // 1·2단계는 중립 PWM을 즉시 시작한다.
        if (status != HAL_OK)
        {
            return status;
        }
        ServoPwm_SeedAngles(&handle->servo_pwm,
                            handle->bringup.neutral_output_active ?
                            neutral_angle_rad : handle->joints.angle_rad);  // 시험 단계의 Rate Limit 시작점을 맞춘다.
    }

    if (hardware->lora_uart != NULL)
    {
        LoRa_Init(&handle->lora, hardware->lora_uart);  // 115200 baud LoRa UART를 연결한다.
        status = LoRa_Start(&handle->lora);             // LoRa 인터럽트 수신을 시작한다.
        if (status != HAL_OK)
        {
            return status;
        }
        if (hardware->configure_lora)
        {
            HAL_Delay(500U);  // RYLR998 기동이 끝날 때까지 기다린다.
            status = LoRa_Configure(&handle->lora,
                                    hardware->lora_local_address,
                                    hardware->lora_network_id);  // 주소와 Network를 모듈 Flash에 저장한다.
            if (status != HAL_OK)
            {
                return status;
            }
        }
    }
    RobotTelemetry_Init(&handle->telemetry);  // 관제 패킷 주기를 초기화한다.

    if (hardware->jetson_spi != NULL)
    {
        JetsonSpi_Init(&handle->jetson, hardware->jetson_spi);  // 32바이트 SPI 센서 프로토콜을 준비한다.
    }

    status = GPS_Start(&handle->gps);  // GPS 인터럽트 수신을 시작한다.
    if (status != HAL_OK)
    {
        return status;
    }
    status = IMU_Start(&handle->imu);  // WT931 인터럽트 수신을 시작한다.
    if (status != HAL_OK)
    {
        return status;
    }
    status = CRSF_Receiver_Start(&handle->crsf_receiver);  // USART6 CRSF 수신을 시작한다.
    if (status != HAL_OK)
    {
        return status;
    }
    status = HAL_TIM_Base_Start_IT(hardware->control_timer);  // 1 ms 압력·제어 Timer를 시작한다.
    if (status != HAL_OK)
    {
        return status;
    }

    handle->initialized = true;  // 전체 초기화 완료를 표시한다.
    return HAL_OK;
}

/* CRSF 프레임을 명령으로 변환하고 수신 통신과 관제 전송을 처리한다. */
void HexapodApp_Process(HexapodApp_Handle_t *handle)
{
    char telemetry_text[ROBOT_TELEMETRY_MAX_TEXT + 1U];  // LoRa 관제 문자열을 저장한다.
    const uint32_t now_ms = HAL_GetTick();               // 현재 HAL 시각을 저장한다.
    uint32_t process_start_cycle;                        // 통신 처리 시작 Cycle을 저장한다.
    uint32_t process_end_cycle;                          // 통신 처리 종료 Cycle을 저장한다.
    uint32_t crsf_age_ms;                                // 최신 조종기 명령 나이를 저장한다.
    uint16_t crsf_buffer_used;                           // 현재 CRSF 버퍼 사용량을 저장한다.
    uint32_t frame_count;                                // 새 CRSF 프레임 수를 저장한다.

    if ((handle == NULL) || !handle->initialized)
    {
        return;
    }

    process_start_cycle = ControlTimingDebug_ReadCycle();  // 통신 처리 시작 시각을 읽는다.

    (void)GPS_Process(&handle->gps);                 // GPS 수신 바이트를 해석한다.
    (void)IMU_Process(&handle->imu);                 // WT931 수신 바이트를 해석한다.
    (void)LoRa_Process(&handle->lora);               // LoRa 응답과 수신 메시지를 해석한다.
    frame_count = CRSF_Protocol_ProcessReceiver(
        &handle->crsf_protocol,
        &handle->crsf_receiver);                     // 대기 중인 CRSF 프레임을 해석한다.

    if (frame_count > 0U)
    {
        UserCommand_UpdateChannels(&handle->user_command,
                                   handle->crsf_protocol.channel,
                                   now_ms);           // 최신 채널을 사용자 명령으로 변환한다.
    }
    UserCommand_UpdateTimeout(&handle->user_command, now_ms);  // 연결 끊김과 중립 재허가를 갱신한다.
    (void)UserCommand_Get(&handle->user_command, &handle->user);  // 안전한 현재 명령을 복사한다.
    HexapodApp_UpdateSimulatedRc(handle, now_ms);                  // 2·3단계이면 실제 CRSF 대신 시험 명령을 넣는다.

    if ((handle->hardware.lora_uart != NULL) &&
        RobotTelemetry_BuildNext(&handle->telemetry,
                                 now_ms,
                                 handle->priority.active_mode,
                                 &handle->safety,
                                 &handle->sensor_snapshot,
                                 Relay_GetStateMask(),
                                 telemetry_text,
                                 sizeof(telemetry_text)))
    {
        (void)LoRa_SendText(&handle->lora,
                            handle->hardware.lora_destination,
                            telemetry_text);  // 주기가 된 관제 패킷 하나를 전송한다.
    }

    if ((handle->hardware.jetson_spi != NULL) &&
        JetsonSpi_PrepareSensorFrame(&handle->jetson,
                                     &handle->sensor_snapshot,
                                     now_ms))
    {
        (void)JetsonSpi_Process(&handle->jetson);  // 준비된 센서 패킷을 Jetson과 교환한다.
    }

    crsf_buffer_used = (uint16_t)((handle->crsf_receiver.head -
                                   handle->crsf_receiver.tail) &
                                  (CRSF_RECEIVER_BUFFER_SIZE - 1U));  // 남은 CRSF 바이트 수를 계산한다.
    crsf_age_ms = handle->user.connected ?
                  (HAL_GetTick() - handle->user.timestamp_ms) : 0U;   // 연결 중인 안전 명령 나이만 계산한다.
    process_end_cycle = ControlTimingDebug_ReadCycle();               // 통신 처리 종료 시각을 읽는다.
    ControlTimingDebug_RecordProcess(process_start_cycle,
                                     process_end_cycle,
                                     crsf_age_ms,
                                     crsf_buffer_used,
                                     handle->crsf_receiver.overflow_count,
                                     handle->crsf_receiver.uart_error_count,
                                     handle->crsf_protocol.crc_error_count);  // 통신 시간과 CRSF 오류를 기록한다.
}

/* 센서에서 서보와 릴레이까지 제어 체인을 한 주기 실행한다. */
/* 새 접촉 다리를 현재 PWM 명령각의 발 위치에 즉시 고정한다. */
static void HexapodApp_ProcessTouchdownLatch(HexapodApp_Handle_t *handle,
                                             uint32_t now_ms)
{
    const uint8_t contact_mask = SensorManager_TakeContactLatch(&handle->sensors);  // 새 접촉 비트를 꺼낸다.
    uint32_t leg;                                                                    // 처리할 다리 번호를 저장한다.

    handle->touchdown_control_mask |= contact_mask;  // 다음 5 ms 제어까지 접촉을 유지한다.
    if (!handle->drone.manual_enable ||
        (handle->drone.tripod_mode != ROBOT_TRIPOD_NORMAL))
    {
        return;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const uint32_t first = leg * ROBOT_JOINTS_PER_LEG;  // 첫 관절 위치를 계산한다.
        const float *command_angle =
            &handle->servo_pwm.previous_angle_rad[first];   // 현재 PWM 명령각을 선택한다.
        RobotVec3_t commanded_foot;                          // 현재 PWM 명령의 FK를 저장한다.

        if (((contact_mask & (uint8_t)(1U << leg)) == 0U) ||
            ((handle->gait.state[leg] != ROBOT_LEG_SWING) &&
             (handle->gait.state[leg] != ROBOT_LEG_LATE_LANDING)))
        {
            continue;
        }

        if (!LegKinematics_Forward((uint8_t)leg,
                                   command_angle,
                                   &commanded_foot) ||
            !FootTrajectory_LatchTouchdown(&handle->foot_trajectory,
                                           (uint8_t)leg,
                                           &commanded_foot,
                                           &handle->posture_control.command_rad,
                                           now_ms))
        {
            continue;
        }
    }
}

/* 1 ms 요청에서 압력 6채널과 접촉 Latch를 갱신한다. */
static bool HexapodApp_RunPressureIfDue(HexapodApp_Handle_t *handle)
{
    if ((handle == NULL) || !handle->initialized || !handle->pressure_due)
    {
        return false;
    }

    handle->pressure_due = false;  // 이번 1 ms 요청을 소비한다.
    if (!SensorManager_UpdatePressure(&handle->sensors))
    {
        return false;
    }

    HexapodApp_ProcessTouchdownLatch(handle, HAL_GetTick());  // 새 접촉을 즉시 처리한다.
    return true;
}

/* 한 번의 5 ms 전체 제어를 실행한다. */
static void HexapodApp_ControlStep(HexapodApp_Handle_t *handle)
{
    BodyPositionEstimator_Output_t position;  // FK 몸체 위치 추정값을 저장한다.
    GaitPoseController_Output_t gait_pose;    // 제한 전 보행 명령을 저장한다.
    BodyPostureController_Output_t posture;   // 자세가 적용된 발 위치를 저장한다.
    RobotFootTargets_t feet;                  // 자세 적용 전 발 위치를 저장한다.
    RobotBodyTwist_t twist;                   // 작업공간이 허용한 보행 명령을 저장한다.
    RobotVec3_t stand_delta[ROBOT_LEG_COUNT]; // 서기·착지 발 변화량을 저장한다.
    bool gait_accepted;                       // 보행 명령 채택 여부를 저장한다.
    bool relay_enable;                        // 모터 전원 허가를 저장한다.
    bool sensor_updated;                      // 관절 초기각 갱신 성공 여부를 저장한다.
    uint32_t leg;                             // 다리 계산 번호를 저장한다.
    uint32_t now_ms;                          // 압력 보정 시각을 저장한다.
    uint32_t sensor_start_cycle;              // 센서 구간 시작 Cycle을 저장한다.
    uint32_t sensor_end_cycle;                // 센서 구간 종료 Cycle을 저장한다.
    uint32_t algorithm_end_cycle;             // 알고리즘 구간 종료 Cycle을 저장한다.
    uint32_t output_end_cycle;                // 출력 구간 종료 Cycle을 저장한다.
    RobotUserCommand_t limited_user;          // 시험 단계가 허용한 명령을 저장한다.

    now_ms = HAL_GetTick();                                       // 현재 제어 시각을 읽는다.
    sensor_start_cycle = ControlTimingDebug_ReadCycle();          // 센서 구간 시작 시각을 읽는다.
    sensor_updated = SensorManager_Update(&handle->sensors);      // 실제 센서 스냅샷을 갱신한다.
    HexapodApp_ProcessTouchdownLatch(handle, now_ms);             // 전체 읽기에서 생긴 접촉도 처리한다.
    (void)SensorManager_GetSnapshot(&handle->sensors,
                                    &handle->sensor_snapshot);     // 같은 주기의 센서값을 복사한다.
    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if ((handle->touchdown_control_mask & (uint8_t)(1U << leg)) != 0U)
        {
            handle->sensor_snapshot.foot_contact[leg] = true;  // 1 ms 사이 접촉을 이번 제어에 전달한다.
        }
    }
    sensor_end_cycle = ControlTimingDebug_ReadCycle();            // 센서 구간 종료 시각을 읽는다.
    handle->safety = Safety_EvaluateImu(
        &handle->safety_control,
        &handle->sensor_snapshot.imu);                            // 현재 자세 Fault를 먼저 평가한다.
    limited_user = HexapodApp_LimitUserCommand(handle);            // 시험 단계 밖의 조종 명령을 제거한다.
    HexapodApp_PrepareStartupZero(handle,
                                  &limited_user,
                                  sensor_updated);                 // 서기 전에 실측각 기반 영점 정렬을 준비한다.
    handle->priority = ControlPriority_Step(&handle->priority_control,
                                            &limited_user,
                                            handle->drone.stand_done,
                                            handle->drone.landing_done,
                                            &handle->safety);       // 안전 상태를 포함해 운용 모드를 결정한다.
    if ((handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_READY) &&
        (handle->priority.active_mode == ROBOT_MODE_STAND))
    {
        handle->startup_zero_state = HEXAPOD_STARTUP_ZERO_COMPLETE;  // 릴레이를 유지한 채 기존 서기로 넘긴다.
    }
    handle->drone = DroneController_Step(&handle->drone_control,
                                         &handle->priority,
                                         handle->sensor_snapshot.foot_contact,
                                         handle->sensor_snapshot.imu.attitude_rad.yaw);  // 조종 입력을 제어 명령으로 바꾼다.
    if ((ROBOT_BRINGUP_STAGE == 3U) &&
        (ROBOT_BRINGUP_PRESSURE_CALIBRATION != 0U))
    {
        PressureLoadCalibration_Update(
            &handle->pressure_calibration,
            handle->sensor_snapshot.pressure_raw,
            (handle->priority.active_mode == ROBOT_MODE_READY) &&
            handle->drone.stand_done,
            now_ms,
            &handle->sensors.pressure);  // 시작 자세와 완전 기립 자세로 압력 임계값을 만든다.
    }
    HexapodApp_LimitDroneCommand(handle, &handle->drone);                          // 현재 단계의 최대 속도를 적용한다.
    position = BodyPositionEstimator_Step(
        &handle->position_estimator,
        handle->servo_pwm.previous_angle_rad,
        &handle->gait,
        handle->sensor_snapshot.foot_contact,
        &handle->sensor_snapshot.imu.attitude_rad);                 // 직전 서보 명령각으로 Stance FK를 계산한다.
    gait_pose = GaitPoseController_Step(
        &handle->gait_pose_control,
        handle->drone.reset_command,
        &handle->drone,
        &position.position_world,
        position.valid_leg_count,
        handle->sensor_snapshot.imu.attitude_rad.yaw);              // 사용자와 PI 보행 명령을 결합한다.
    handle->gait = GaitManager_Step(&handle->gait_manager,
                                    handle->drone.manual_enable,
                                    handle->drone.tripod_enable,
                                    handle->workspace_limiter.phase_result_valid,
                                    handle->workspace_limiter.phase_result_accepted,
                                    handle->drone.tripod_mode,
                                    handle->drone.recovery_progress,
                                    handle->sensor_snapshot.foot_contact);  // Tripod 상태와 진행률을 계산한다.
    FootTrajectory_UpdateCommandedLanding(
        &handle->foot_trajectory,
        handle->servo_pwm.previous_angle_rad,
        &handle->gait,
        &handle->posture_control.command_rad,
        (ROBOT_COMMON_Z_RECOVERY_ENABLE != 0U) &&
        handle->drone.manual_enable &&
        (handle->drone.tripod_mode == ROBOT_TRIPOD_NORMAL),
        now_ms);  // 활성 시 접촉 20 ms 후 PWM FK의 공통 Z 오차를 수집한다.
    handle->touchdown_control_mask = 0U;  // Gait가 소비한 접촉 Latch를 비운다.
    twist = WorkspaceLimiter_Gait(&handle->workspace_limiter,
                                  &gait_pose.twist,
                                  handle->drone.manual_enable,
                                  &handle->gait,
                                  &handle->posture_control.command_rad,
                                  handle->drone.reset_command,
                                  &gait_accepted);                   // 미래 검증을 마친 보행 속도를 적용한다.
    feet = FootTrajectory_Step(&handle->foot_trajectory,
                               &twist,
                               &handle->drone,
                               &handle->gait,
                               &handle->posture_control.command_rad);      // 연속 발 궤적을 계산한다.
    StandLanding_Calculate(handle->drone.stand_enable,
                           handle->drone.landing_enable,
                           handle->drone.posture_progress,
                           stand_delta);                                  // 서기·착지 변화량을 계산한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        feet.foot[leg].x += stand_delta[leg].x;  // 서기·착지 X 변화를 더한다.
        feet.foot[leg].y += stand_delta[leg].y;  // 서기·착지 Y 변화를 더한다.
        feet.foot[leg].z += stand_delta[leg].z;  // 서기·착지 Z 변화를 더한다.
    }

    posture = BodyPostureController_Step(
        &handle->posture_control,
        feet.foot,
        &handle->drone,
        &handle->sensor_snapshot.imu.attitude_rad,
        handle->drone.reset_command);  // 동적 자세 제한과 발 역회전을 적용한다.

    algorithm_end_cycle = ControlTimingDebug_ReadCycle();  // 제어 알고리즘 종료 시각을 읽는다.
    ControlTimingDebug_RecordSignals(&gait_pose.twist, &twist);  // PI 후보와 작업공간 채택값을 기록한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t limited = {0.0f, 0.0f, 0.0f};            // 최종 제한 발 위치를 저장한다.
        bool was_limited = false;                            // 발 위치 제한 여부를 저장한다.
        bool inverse_ok = false;                             // 최종 IK 계산 성공 여부를 저장한다.
        const uint32_t first = leg * ROBOT_JOINTS_PER_LEG;  // 다리 첫 관절 위치를 계산한다.
        const bool limit_ok = LegKinematics_LimitFoot(
            (uint8_t)leg,
            &posture.targets.foot[leg],
            &limited,
            &was_limited);                                  // 관절 여유를 포함한 최종 제한을 적용한다.

        if (limit_ok)
        {
            inverse_ok = LegKinematics_Inverse(
                &handle->kinematics,
                (uint8_t)leg,
                &limited,
                &handle->joints.angle_rad[first]);  // 제한된 위치의 IK를 계산한다.
        }
        handle->joints.ik_valid[leg] = limit_ok && inverse_ok;  // 최종 제한과 IK 결과를 함께 저장한다.

        if (!handle->joints.ik_valid[leg])
        {
            HexapodApp_RecordControllerFault(
                handle,
                limit_ok ? ROBOT_CONTROLLER_FAULT_IK_SOLVE
                         : ROBOT_CONTROLLER_FAULT_IK_INPUT,
                (uint8_t)leg,
                &posture.targets.foot[leg],
                limit_ok ? &limited : NULL,
                was_limited);  // 최초 실패 좌표와 단계를 보존한다.
        }
    }

    handle->safety = Safety_Evaluate(&handle->safety_control,
                                     &handle->sensor_snapshot.imu,
                                     handle->joints.ik_valid);        // 현재 IK 실패의 연속 횟수를 평가한다.

    if (handle->bringup.neutral_output_active)
    {
        memset(handle->joints.angle_rad, 0,
               sizeof(handle->joints.angle_rad));                    // 2단계에서 18개 관절을 0도로 고정한다.
    }
    HexapodApp_ApplyStartupZero(handle);                              // 초기각에서 0도로 저속 이동시킨다.
    (void)ServoPwm_WriteAngles(&handle->servo_pwm,
                               handle->joints.angle_rad);             // 속도 제한 후 관절 PWM을 출력한다.
    HexapodApp_RearmStartupZero(handle);                              // 착지 완료 후 다음 서기를 준비한다.

    relay_enable = (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_CAPTURE) ||
                   (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_MOVE) ||
                   (handle->startup_zero_state == HEXAPOD_STARTUP_ZERO_READY) ||
                   ((ROBOT_BRINGUP_STAGE == 2U) &&
                    handle->user.connected &&
                    handle->user.motion_armed) ||
                   ((ROBOT_BRINGUP_STAGE >= 3U) &&
                    ((handle->priority.active_mode == ROBOT_MODE_STAND) ||
                     (handle->priority.active_mode == ROBOT_MODE_READY) ||
                     (handle->priority.active_mode == ROBOT_MODE_MANUAL) ||
                     (handle->priority.active_mode == ROBOT_MODE_CORRECTION) ||
                     ((handle->priority.active_mode == ROBOT_MODE_LANDING) &&
                      !handle->drone.landing_done)));                  // 연결 확인 후 현재 단계가 허용한 동작에서만 전원을 요청한다.
    relay_enable = relay_enable &&
                   !handle->safety.rollover_fault &&
                   !handle->safety.controller_fault &&
                   !handle->drone.kill_enable;                         // Kill과 Fault에서 전원 요청을 제거한다.
    Relay_ApplySafety(relay_enable,
                      handle->drone.kill_enable ||
                      handle->safety.rollover_fault ||
                      handle->safety.controller_fault);                // Kill과 Fault에서 릴레이를 즉시 끈다.
    handle->bringup.relay_enabled = (Relay_GetStateMask() != 0U);       // 실제 릴레이 출력 여부를 기록한다.
    handle->control_count++;                                           // 완료 제어 주기를 기록한다.
    output_end_cycle = ControlTimingDebug_ReadCycle();                  // IK와 출력 종료 시각을 읽는다.
    ControlTimingDebug_RecordBreakdown(sensor_end_cycle - sensor_start_cycle,
                                       algorithm_end_cycle - sensor_end_cycle,
                                       output_end_cycle - algorithm_end_cycle);  // 제어 구간별 시간을 기록한다.
}

/* Timer 요청이 있을 때만 제어 체인을 한 번 실행한다. */
bool HexapodApp_RunControlIfDue(HexapodApp_Handle_t *handle)
{
    uint32_t control_start_cycle;  // 전체 제어 시작 Cycle을 저장한다.
    uint32_t control_end_cycle;    // 전체 제어 종료 Cycle을 저장한다.

    if ((handle == NULL) || !handle->initialized || !handle->control_due)
    {
        return false;
    }

    control_start_cycle = ControlTimingDebug_ReadCycle();  // 실제 제어 시작 시각을 읽는다.
    ControlTimingDebug_BeginControl(control_start_cycle);   // 직전 제어와의 시작 간격을 기록한다.
    handle->control_due = false;                            // 이번 Timer 요청을 소비한다.
    HexapodApp_ControlStep(handle);                          // 한 번의 5 ms 제어를 실행한다.
    control_end_cycle = ControlTimingDebug_ReadCycle();     // 실제 제어 종료 시각을 읽는다.
    ControlTimingDebug_EndControl(control_start_cycle,
                                  control_end_cycle,
                                  HAL_GetTick(),
                                  handle->control_count,
                                  handle->missed_control_count,
                                  handle->priority.active_mode,
                                  &handle->user,
                                  &handle->drone,
                                  handle->gait_manager.phase_index,
                                  handle->gait_manager.phase_time_s);  // 실행 시간과 이상 순간을 기록한다.
    return true;
}

/* TIM6 1 ms Tick을 압력 요청과 5분주 제어 요청으로 나눈다. */
void HexapodApp_TimerCallback(HexapodApp_Handle_t *handle,
                              TIM_HandleTypeDef *timer)
{
    if ((handle == NULL) || (timer != handle->hardware.control_timer))
    {
        return;
    }

    handle->pressure_due = true;  // 매 Tick마다 압력센서 읽기를 요청한다.
    handle->control_tick_divider++;  // 5 ms 전체 제어 분주를 진행한다.
    if (handle->control_tick_divider < ROBOT_CONTROL_PERIOD_MS)
    {
        return;
    }

    handle->control_tick_divider = 0U;  // 다음 5 ms 구간을 시작한다.
    if (handle->control_due)
    {
        handle->missed_control_count++;  // 이전 제어 미처리 상태를 기록한다.
        ControlTimingDebug_RecordTimerMissed(handle->missed_control_count);  // 디버그 통계에 누락을 반영한다.
    }
    handle->control_due = true;           // 다음 Main Loop 제어 실행을 요청한다.
}

/* UART 수신 완료를 네 개 실제 통신 드라이버에 전달한다. */
void HexapodApp_UartRxCallback(HexapodApp_Handle_t *handle,
                               UART_HandleTypeDef *uart)
{
    if (handle == NULL)
    {
        return;
    }

    GPS_RxCpltCallback(&handle->gps, uart);                    // GPS UART이면 다음 수신을 건다.
    IMU_RxCpltCallback(&handle->imu, uart);                    // WT931 UART이면 다음 수신을 건다.
    LoRa_RxCpltCallback(&handle->lora, uart);                  // LoRa UART이면 다음 수신을 건다.
    CRSF_Receiver_RxCpltCallback(&handle->crsf_receiver, uart);// CRSF UART이면 다음 수신을 건다.
}

/* UART 오류를 네 개 실제 통신 드라이버에 전달한다. */
void HexapodApp_UartErrorCallback(HexapodApp_Handle_t *handle,
                                  UART_HandleTypeDef *uart)
{
    if (handle == NULL)
    {
        return;
    }

    GPS_ErrorCallback(&handle->gps, uart);                     // GPS 오류를 복구한다.
    IMU_ErrorCallback(&handle->imu, uart);                     // WT931 오류를 복구한다.
    LoRa_ErrorCallback(&handle->lora, uart);                   // LoRa 오류를 복구한다.
    CRSF_Receiver_ErrorCallback(&handle->crsf_receiver, uart); // CRSF 오류를 복구한다.
}

/* 현재 CubeMX Handle을 최종 앱 장치에 연결한다. */
HAL_StatusTypeDef HexapodApp_BoardInit(void)
{
    HexapodApp_Hardware_t hardware = {0};  // 현재 보드의 장치 연결을 준비한다.

    hardware.gps_uart = &huart2;           // GPS를 USART2에 연결한다.
    hardware.imu_uart = &huart3;           // WT931을 USART3에 연결한다.
    hardware.lora_uart = NULL;             // 실기 검증 전 LoRa를 비활성화한다.
    hardware.crsf_uart = &huart6;          // CRSF를 USART6에 연결한다.
    hardware.adc_spi = &hspi1;             // MCP3008을 SPI1에 연결한다.
    hardware.jetson_spi = &hspi2;          // Jetson 32바이트 SPI2 Slave 통신을 활성화한다.
    hardware.control_timer = &htim6;       // 1 ms 기준 주기를 TIM6에 연결한다.
    hardware.servo_timers.tim1 = &htim1;   // TIM1 서보 채널을 연결한다.
    hardware.servo_timers.tim2 = &htim2;   // TIM2 서보 채널을 연결한다.
    hardware.servo_timers.tim3 = &htim3;   // TIM3 서보 채널을 연결한다.
    hardware.servo_timers.tim4 = &htim4;   // TIM4 서보 채널을 연결한다.
    hardware.servo_timers.tim5 = &htim5;   // TIM5 서보 채널을 연결한다.
    hardware.servo_timers.tim8 = &htim8;   // TIM8 서보 채널을 연결한다.

    return HexapodApp_Init(&g_hexapod_app, &hardware);  // 전체 최종 앱을 시작한다.
}

/* Main Loop에서 압력·통신·5 ms 제어 요청을 처리한다. */
void HexapodApp_BoardProcess(void)
{
    (void)HexapodApp_RunPressureIfDue(&g_hexapod_app);  // 1 ms 압력 요청을 먼저 처리한다.
    HexapodApp_Process(&g_hexapod_app);                 // 대기 중인 통신을 해석한다.
    (void)HexapodApp_RunControlIfDue(&g_hexapod_app);   // 5분주 제어 요청이 있으면 실행한다.
}

/* 보드 Timer 완료를 최종 앱으로 전달한다. */
void HexapodApp_BoardTimerCallback(TIM_HandleTypeDef *timer)
{
    HexapodApp_TimerCallback(&g_hexapod_app, timer);  // TIM6 요청을 압력·제어 플래그로 바꾼다.
}

/* 보드 UART 수신 완료를 최종 앱으로 전달한다. */
void HexapodApp_BoardUartRxCallback(UART_HandleTypeDef *uart)
{
    HexapodApp_UartRxCallback(&g_hexapod_app, uart);  // 해당 UART 드라이버의 다음 수신을 건다.
}

/* 보드 UART 오류를 최종 앱으로 전달한다. */
void HexapodApp_BoardUartErrorCallback(UART_HandleTypeDef *uart)
{
    HexapodApp_UartErrorCallback(&g_hexapod_app, uart);  // 해당 UART 드라이버의 수신을 복구한다.
}
