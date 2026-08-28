#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#define ROBOT_LEG_COUNT                 6U          // 전체 다리 수를 정의한다.
#define ROBOT_JOINTS_PER_LEG            3U          // 다리별 관절 수를 정의한다.
#define ROBOT_JOINT_COUNT               18U         // 전체 관절 수를 정의한다.
#define ROBOT_PRESSURE_COUNT            6U          // 전체 압력센서 수를 정의한다.
#define ROBOT_CRSF_CHANNEL_COUNT        16U         // CRSF 채널 수를 정의한다.

#define ROBOT_CONTROL_PERIOD_S          0.005f      // 제어 주기를 초 단위로 정의한다.
#define ROBOT_CONTROL_PERIOD_MS         5U          // 제어 주기를 밀리초 단위로 정의한다.
#define ROBOT_PRESSURE_PERIOD_MS        1U          // 압력센서 읽기 주기를 정의한다.
#define ROBOT_TOUCHDOWN_SETTLE_MS       20U         // 접촉 후 FK 안정 시간을 정의한다.
#define ROBOT_PI_F                      3.14159265358979323846f
#define ROBOT_DEG_TO_RAD_F              (ROBOT_PI_F / 180.0f)
#define ROBOT_RAD_TO_DEG_F              (180.0f / ROBOT_PI_F)

#define ROBOT_LINK_1_M                  0.074f      // 첫 번째 링크 길이를 정의한다.
#define ROBOT_LINK_2_M                  0.121f      // 두 번째 링크 길이를 정의한다.
#define ROBOT_LINK_3_M                  0.230f      // 세 번째 링크 길이를 정의한다.
#define ROBOT_BASE_FOOT_RADIUS_M        0.218728f   // 기본 발의 수평 반경을 정의한다.
#define ROBOT_BASE_FOOT_Z_M             (-0.287006f) // 기본 발 높이를 정의한다.
#define ROBOT_WORKSPACE_MARGIN_M        0.001f      // IK 작업공간 여유를 정의한다.

#define ROBOT_GAIT_PHASE_TIME_S         0.5f        // 한 Tripod 위상 시간을 정의한다.
#define ROBOT_GAIT_START_DELAY_MS       100U        // 정지 상태의 첫 보행 입력 안정 시간을 정의한다.
#define ROBOT_GAIT_NEXT_COMMAND_MS      25U         // 다음 위상 명령 확정 시점을 정의한다.
#define ROBOT_GAIT_START_DELAY_CYCLES   \
    (ROBOT_GAIT_START_DELAY_MS / ROBOT_CONTROL_PERIOD_MS)  // 첫 보행 대기 주기 수를 계산한다.
#define ROBOT_GAIT_PREVIEW_SAMPLE_COUNT \
    (ROBOT_GAIT_NEXT_COMMAND_MS / ROBOT_CONTROL_PERIOD_MS)  // 다음 위상 검사 지점 수를 계산한다.
#define ROBOT_STAND_TIME_S              5.6f        // 서기 시간을 정의한다.
#define ROBOT_LANDING_TIME_S            5.6f        // 착지 하강 시간을 정의한다.
#define ROBOT_SETTLING_TIME_S           0.5f        // 자세 안정 시간을 정의한다.
#define ROBOT_RECOVERY_TIME_S           0.5f        // Tripod 복구 시간을 정의한다.

#define ROBOT_MAX_LINEAR_SPEED_MPS      0.28f       // X/Y 최대 이동 속도를 정의한다.
#define ROBOT_MAX_LATERAL_SPEED_MPS     0.20f       // 조종 횡이동 최대 속도를 정의한다.
#define ROBOT_MAX_CORRECTION_SPEED_MPS  0.10f       // 보정 최대 이동 속도를 정의한다.
#define ROBOT_MAX_YAW_RATE_RADPS        (18.0f * ROBOT_DEG_TO_RAD_F)  // 조종 Yaw 최대 속도를 기존의 40%로 제한한다.
#define ROBOT_MAX_ROLL_RAD              (45.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_MAX_PITCH_RAD             (45.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_MAX_CORRECTION_YAW_RAD    (30.0f * ROBOT_DEG_TO_RAD_F)

#define ROBOT_GAIT_LINEAR_THRESHOLD_MPS 0.005f      // 보행 시작 선속도 기준을 정의한다.
#define ROBOT_GAIT_YAW_THRESHOLD_RADPS  (1.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_SWING_HEIGHT_M            0.20f       // 기본 Swing 높이를 정의한다.
#define ROBOT_SWING_HEIGHT_MIN_M        0.05f       // 최소 Swing 높이를 정의한다.
#define ROBOT_SWING_HEIGHT_MAX_M        0.25f       // 최대 Swing 높이를 정의한다.
#define ROBOT_SWING_RADIAL_OFFSET_M     0.07f       // Swing 방사 오프셋을 정의한다.
#define ROBOT_EARLY_LANDING_PROGRESS    0.50f       // Early Landing 시작점을 정의한다.
#define ROBOT_LATE_LANDING_SPEED_MPS    0.20f       // Late Landing 하강 속도를 정의한다.
#define ROBOT_LATE_INWARD_SPEED_MPS     0.16f       // Late Landing 안쪽 속도를 정의한다.
#define ROBOT_LATE_LANDING_MAX_DISTANCE_M 0.10f     // Late Landing 최대 하강 거리를 정의한다.
#define ROBOT_LATE_LANDING_MAX_TIME_S   \
    (ROBOT_LATE_LANDING_MAX_DISTANCE_M / ROBOT_LATE_LANDING_SPEED_MPS)  // 최대 탐색 시간을 계산한다.

#define ROBOT_COMMON_Z_RECOVERY_ENABLE      1U        // PWM 명령 FK 기반 공통 Z 복구를 활성화한다.
#define ROBOT_COMMON_Z_RECOVERY_DEADBAND_M  0.0005f   // 0.5 mm 이하 착지 오차를 무시한다.
#define ROBOT_COMMON_Z_RECOVERY_GAIN        1.00f     // 데드밴드 초과 오차의 100%를 반영한다.
#define ROBOT_COMMON_Z_RECOVERY_MAX_M       0.1000f   // 착지당 복구량을 최대 100 mm로 제한한다.
#define ROBOT_COMMON_Z_RECOVERY_TIME_S      0.25f     // S-curve 복구 시간을 정의한다.

#define ROBOT_SLIP_DISTANCE_M           0.05f       // Stance Foot Slip 거리를 정의한다.
#define ROBOT_SLIP_CONFIRM_SAMPLES      5U          // Slip 확정 연속 횟수를 정의한다.

#define ROBOT_ROLLOVER_LIMIT_RAD        (80.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_IK_FAULT_CONFIRM_SAMPLES  3U          // IK Fault 확정 연속 횟수를 정의한다.
#define ROBOT_JOINT_MIN_RAD             (-135.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_JOINT_MAX_RAD             (135.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_STARTUP_SENSOR_SETTLE_S    0.20f       // 서보 전원 안정 시간을 정의한다.
#define ROBOT_STARTUP_SENSOR_SAMPLES     20U         // 초기 관절각 평균 횟수를 정의한다.
#define ROBOT_STARTUP_ZERO_RATE_RADPS    (30.0f * ROBOT_DEG_TO_RAD_F)  // 초기 영점 정렬 속도를 정의한다.
#define ROBOT_JOINT_RATE_RADPS          (315.8f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_JOINT_STEP_RAD            (ROBOT_JOINT_RATE_RADPS * ROBOT_CONTROL_PERIOD_S)

#define ROBOT_SERVO_MIN_US              500U        // 서보 최소 Pulse를 정의한다.
#define ROBOT_SERVO_NEUTRAL_US          1500U       // 서보 기본 중립 Pulse를 정의한다.
#define ROBOT_SERVO_MAX_US              2500U       // 서보 최대 Pulse를 정의한다.

#define ROBOT_CRSF_TIMEOUT_MS           100U        // CRSF 연결 끊김 시간을 정의한다.
#define ROBOT_CRSF_REARM_MS             200U        // 재연결 중립 유지 시간을 정의한다.
#define ROBOT_THROTTLE_DEADBAND         20          // Throttle Dead Zone을 정의한다.
#define ROBOT_STICK_DEADBAND            50          // 일반 짐벌 Dead Zone을 정의한다.
#define ROBOT_STICK_FILTER_HZ           5.0f        // 짐벌 LPF 차단 주파수를 정의한다.
#define ROBOT_TEST_GIMBAL_RATE_RAWPS    5000.0f     // 시험 짐벌 변화율을 정의한다.

#endif
