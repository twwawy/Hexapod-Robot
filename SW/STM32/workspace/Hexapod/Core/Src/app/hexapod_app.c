#include "app/hexapod_app.h"

#include "common/robot_calibration.h"
#include "high_control/stand_landing.h"
#include "low_control/relay.h"

#include <stddef.h>
#include <string.h>

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

/* 실제 장치 드라이버와 모든 제어 상태를 초기화한다. */
HAL_StatusTypeDef HexapodApp_Init(HexapodApp_Handle_t *handle,
                                  const HexapodApp_Hardware_t *hardware)
{
    HAL_StatusTypeDef status;  // 장치 초기화 결과를 저장한다.

    if ((handle == NULL) || (hardware == NULL) ||
        (hardware->gps_uart == NULL) || (hardware->imu_uart == NULL) ||
        (hardware->crsf_uart == NULL) || (hardware->adc_spi == NULL) ||
        (hardware->control_timer == NULL))
    {
        return HAL_ERROR;
    }

    memset(handle, 0, sizeof(*handle));  // 이전 실행 상태를 제거한다.
    handle->hardware = *hardware;        // CubeMX Handle 연결을 저장한다.

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
    (void)RobotCalibration_Apply(&g_robot_calibration,
                                 &handle->imu,
                                 &handle->adc,
                                 &handle->sensors.joints,
                                 &handle->sensors.pressure,
                                 &handle->servo_pwm,
                                 &handle->user_command);          // 중앙 실측값을 모든 장치에 적용한다.
    status = ServoPwm_Start(&handle->servo_pwm);                  // 보정된 중립 PWM을 시작한다.
    if (status != HAL_OK)
    {
        return status;
    }
    ServoPwm_SeedAngles(&handle->servo_pwm,
                        handle->joints.angle_rad);                // 정상 IK를 Rate Limit 시작점으로 둔다.
    Relay_Init();                                                 // 모터 전원을 꺼진 상태로 준비한다.

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
    status = HAL_TIM_Base_Start_IT(hardware->control_timer);  // 5 ms 제어 Timer를 시작한다.
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
    uint32_t frame_count;                                // 새 CRSF 프레임 수를 저장한다.

    if ((handle == NULL) || !handle->initialized)
    {
        return;
    }

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
}

/* 센서에서 서보와 릴레이까지 제어 체인을 한 주기 실행한다. */
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
    uint32_t leg;                             // 다리 계산 번호를 저장한다.

    (void)SensorManager_Update(&handle->sensors);                  // 실제 센서 스냅샷을 갱신한다.
    (void)SensorManager_GetSnapshot(&handle->sensors,
                                    &handle->sensor_snapshot);     // 같은 주기의 센서값을 복사한다.
    handle->safety = Safety_Evaluate(&handle->safety_control,
                                     &handle->sensor_snapshot.imu,
                                     handle->joints.ik_valid);     // 이전 IK와 현재 자세 Fault를 먼저 평가한다.
    handle->priority = ControlPriority_Step(&handle->priority_control,
                                            &handle->user,
                                            handle->drone.stand_done,
                                            handle->drone.landing_done,
                                            &handle->safety);       // 안전 상태를 포함해 운용 모드를 결정한다.
    handle->drone = DroneController_Step(&handle->drone_control,
                                         &handle->priority,
                                         handle->sensor_snapshot.foot_contact,
                                         handle->sensor_snapshot.imu.attitude_rad.yaw);  // 조종 입력을 제어 명령으로 바꾼다.
    position = BodyPositionEstimator_Step(
        &handle->position_estimator,
        handle->sensor_snapshot.joint_angle_rad,
        &handle->gait,
        handle->sensor_snapshot.foot_contact,
        &handle->sensor_snapshot.imu.attitude_rad);                 // Stance FK로 몸체 위치를 추정한다.
    gait_pose = GaitPoseController_Step(
        &handle->gait_pose_control,
        handle->drone.reset_command,
        &handle->drone,
        &position.position_world,
        position.valid_leg_count,
        handle->sensor_snapshot.imu.attitude_rad.yaw);              // 사용자와 PI 보행 명령을 결합한다.
    twist = WorkspaceLimiter_Gait(&handle->workspace_limiter,
                                  &gait_pose.twist,
                                  handle->drone.manual_enable,
                                  &handle->posture_control.command_rad,
                                  handle->drone.reset_command,
                                  &gait_accepted);                   // 전체 위상 IK가 가능한 속도만 채택한다.
    handle->gait = GaitManager_Step(&handle->gait_manager,
                                    handle->drone.tripod_enable,
                                    handle->drone.tripod_mode,
                                    handle->drone.recovery_progress,
                                    handle->sensor_snapshot.foot_contact);  // Tripod 상태와 진행률을 계산한다.
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

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        RobotVec3_t limited;                                      // 최종 제한 발 위치를 저장한다.
        bool was_limited;                                         // 발 위치 제한 여부를 저장한다.
        const uint32_t first = leg * ROBOT_JOINTS_PER_LEG;        // 다리 첫 관절 위치를 계산한다.
        const bool limit_ok = LegKinematics_LimitFoot(
            (uint8_t)leg,
            &posture.targets.foot[leg],
            &limited,
            &was_limited);                                        // IK 앞 최종 작업공간 여유를 적용한다.

        handle->joints.ik_valid[leg] = limit_ok &&
            LegKinematics_Inverse(&handle->kinematics,
                                  (uint8_t)leg,
                                  &limited,
                                  &handle->joints.angle_rad[first]);  // 제한된 위치의 IK를 계산한다.
    }

    handle->safety = Safety_Evaluate(&handle->safety_control,
                                     &handle->sensor_snapshot.imu,
                                     handle->joints.ik_valid);        // 현재 IK 실패를 같은 주기에 Latch한다.
    (void)ServoPwm_WriteAngles(&handle->servo_pwm,
                               handle->joints.angle_rad);             // 속도 제한 후 관절 PWM을 출력한다.

    relay_enable = !handle->safety.rollover_fault &&
                   !handle->safety.controller_fault &&
                   !handle->drone.kill_enable &&
                   ((handle->priority.active_mode == ROBOT_MODE_STAND) ||
                    (handle->priority.active_mode == ROBOT_MODE_READY) ||
                    (handle->priority.active_mode == ROBOT_MODE_MANUAL) ||
                    (handle->priority.active_mode == ROBOT_MODE_CORRECTION) ||
                    ((handle->priority.active_mode == ROBOT_MODE_LANDING) &&
                     !handle->drone.landing_done));                    // 동작 중일 때만 모터 전원을 허가한다.
    Relay_ApplySafety(relay_enable,
                      handle->drone.kill_enable ||
                      handle->safety.rollover_fault ||
                      handle->safety.controller_fault);                // Kill과 Fault에서 릴레이를 즉시 끈다.
    handle->control_count++;                                           // 완료 제어 주기를 기록한다.
}

/* Timer 요청이 있을 때만 제어 체인을 한 번 실행한다. */
bool HexapodApp_RunControlIfDue(HexapodApp_Handle_t *handle)
{
    if ((handle == NULL) || !handle->initialized || !handle->control_due)
    {
        return false;
    }

    handle->control_due = false;  // 이번 Timer 요청을 소비한다.
    HexapodApp_ControlStep(handle);// 한 번의 5 ms 제어를 실행한다.
    return true;
}

/* TIM6 완료를 짧은 제어 실행 요청으로 바꾼다. */
void HexapodApp_TimerCallback(HexapodApp_Handle_t *handle,
                              TIM_HandleTypeDef *timer)
{
    if ((handle == NULL) || (timer != handle->hardware.control_timer))
    {
        return;
    }

    if (handle->control_due)
    {
        handle->missed_control_count++;  // 이전 제어 미처리 상태를 기록한다.
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
