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
#define ROBOT_PRESSURE_CONTACT_CONFIRM_MS 10U       // 접촉 확정 연속 시간을 정의한다.
#define ROBOT_PRESSURE_RELEASE_CONFIRM_MS 10U       // 접촉 해제 확정 연속 시간을 정의한다.
#define ROBOT_PRESSURE_CONTACT_CONFIRM_SAMPLES \
    (ROBOT_PRESSURE_CONTACT_CONFIRM_MS / ROBOT_PRESSURE_PERIOD_MS)  // 접촉 확정 표본 수를 계산한다.
#define ROBOT_PRESSURE_CONTACT_ACCEPT_SAMPLES 8U        // 10 ms 검사에서 필요한 접촉 표본 수를 정의한다.
#define ROBOT_PRESSURE_RELEASE_CONFIRM_SAMPLES \
    (ROBOT_PRESSURE_RELEASE_CONFIRM_MS / ROBOT_PRESSURE_PERIOD_MS)  // 접촉 해제 표본 수를 계산한다.
#define ROBOT_PI_F                      3.14159265358979323846f
#define ROBOT_DEG_TO_RAD_F              (ROBOT_PI_F / 180.0f)
#define ROBOT_RAD_TO_DEG_F              (180.0f / ROBOT_PI_F)

#define ROBOT_IMU_AUTO_LEVEL_ENABLE          0U                              // 저장한 Roll·Pitch 영점 보정값을 사용한다.
#define ROBOT_IMU_LEVEL_SETTLE_MS            1000U                           // IMU 출력 안정 대기 시간을 정의한다.
#define ROBOT_IMU_LEVEL_CAPTURE_MS           2000U                           // 정지 자세 측정 시간을 정의한다.
#define ROBOT_IMU_LEVEL_TIMEOUT_MS           10000U                          // 자동 영점 보정 제한 시간을 정의한다.
#define ROBOT_IMU_LEVEL_MIN_SAMPLES          100U                            // 영점 계산에 필요한 최소 자세 표본 수를 정의한다.
#define ROBOT_IMU_LEVEL_MAX_GYRO_RADPS       (3.0f * ROBOT_DEG_TO_RAD_F)      // 정지 판정 각속도를 정의한다.
#define ROBOT_IMU_LEVEL_MAX_DEVIATION_RAD    (1.0f * ROBOT_DEG_TO_RAD_F)      // 표본 자세 편차를 정의한다.
#define ROBOT_IMU_LEVEL_MAX_OFFSET_RAD       (10.0f * ROBOT_DEG_TO_RAD_F)     // 허용할 최대 영점 오차를 정의한다.

#define ROBOT_LINK_1_M                  0.074f      // 첫 번째 링크 길이를 정의한다.
#define ROBOT_LINK_2_M                  0.121f      // 두 번째 링크 길이를 정의한다.
#define ROBOT_LINK_3_M                  0.230f      // 세 번째 링크 길이를 정의한다.
#define ROBOT_BASE_FOOT_RADIUS_M        0.218728f   // 기본 발의 수평 반경을 정의한다.
#define ROBOT_BASE_FOOT_Z_M             (-0.287006f) // 기본 발 높이를 정의한다.
#define ROBOT_WORKSPACE_MARGIN_M        0.001f      // IK 작업공간 여유를 정의한다.

#define ROBOT_GAIT_PHASE_TIME_S         1.0f        // 한 Tripod 위상 시간을 정의한다.
#define ROBOT_WAVE_STANCE_PHASES        5U          // 개별 다리가 지지하는 위상 수를 정의한다.
#define ROBOT_WAVE_SPEED_SCALE          0.2f        // 다섯 지지 위상에서도 기존 보폭 범위를 유지한다.
#define ROBOT_GAIT_START_DELAY_MS       100U        // 정지 상태의 첫 보행 입력 안정 시간을 정의한다.
#define ROBOT_GAIT_START_DELAY_CYCLES   \
    (ROBOT_GAIT_START_DELAY_MS / ROBOT_CONTROL_PERIOD_MS)  // 첫 보행 대기 주기 수를 계산한다.
#define ROBOT_GAIT_PREVIEW_SAMPLE_COUNT 3U          // 시작·중앙·끝 검사 지점 수를 정의한다.
#define ROBOT_GAIT_YAW_FEEDBACK_ENABLE  0U          // IMU Heading 자동 보정을 비활성화한다.
#define ROBOT_STAND_TIME_S              5.6f        // 서기 시간을 정의한다.
#define ROBOT_LANDING_TIME_S            5.6f        // 착지 하강 시간을 정의한다.
#define ROBOT_SETTLING_TIME_S           0.5f        // 자세 안정 시간을 정의한다.
#define ROBOT_RECOVERY_TIME_S           0.5f        // Tripod 복구 시간을 정의한다.

#define ROBOT_MAX_LINEAR_SPEED_MPS      0.10f       // X축 최대 이동 속도를 정의한다.
#define ROBOT_MAX_LATERAL_SPEED_MPS     0.07f       // Y축 최대 횡이동 속도를 정의한다.
#define ROBOT_MAX_CORRECTION_SPEED_MPS  0.10f       // 보정 최대 이동 속도를 정의한다.
#define ROBOT_MAX_YAW_RATE_RADPS        (18.0f * ROBOT_DEG_TO_RAD_F)  // 조종 Yaw 최대 속도를 기존의 40%로 제한한다.
#define ROBOT_MAX_ROLL_RAD              (45.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_MAX_PITCH_RAD             (45.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_MAX_CORRECTION_YAW_RAD    (30.0f * ROBOT_DEG_TO_RAD_F)

#define ROBOT_RL_ACTION_TIMEOUT_MS           100U                          // 새 정책 명령의 최대 대기 시간을 정의한다.
#define ROBOT_RL_OBSERVATION_MAX_AGE_MS       60U                           // 추론 관측의 최대 나이를 정의한다.
#define ROBOT_RL_OBSERVATION_HISTORY_COUNT   32U                           // 최근 관측의 검증 이력 수를 정의한다.
#define ROBOT_RL_MAX_ROLL_RAD                (15.0f * ROBOT_DEG_TO_RAD_F)  // 정책 Roll 목표의 허용 크기를 정의한다.
#define ROBOT_RL_MAX_PITCH_RAD               (15.0f * ROBOT_DEG_TO_RAD_F)  // 정책 Pitch 목표의 허용 크기를 정의한다.
#define ROBOT_RL_MAX_DX_M                    0.03f                         // 착지 X 잔차의 허용 크기를 정의한다.
#define ROBOT_RL_MAX_DY_M                    0.03f                         // 착지 Y 잔차의 허용 크기를 정의한다.
#define ROBOT_RL_MAX_DZ_M                    0.03f                         // 착지 Z 잔차의 허용 크기를 정의한다.
#define ROBOT_RL_MAX_DH_M                    0.03f                         // Swing 높이 잔차의 허용 크기를 정의한다.

#define ROBOT_GAIT_LINEAR_THRESHOLD_MPS 0.005f      // 보행 시작 선속도 기준을 정의한다.
#define ROBOT_GAIT_YAW_THRESHOLD_RADPS  (1.0f * ROBOT_DEG_TO_RAD_F)
#define ROBOT_SWING_HEIGHT_M            0.20f       // 기본 Swing 높이를 정의한다.
#define ROBOT_SWING_HEIGHT_MIN_M        0.05f       // 최소 Swing 높이를 정의한다.
#define ROBOT_SWING_HEIGHT_MAX_M        0.25f       // 최대 Swing 높이를 정의한다.
#define ROBOT_SWING_RADIAL_OFFSET_M     0.07f       // Swing 방사 오프셋을 정의한다.
#define ROBOT_EARLY_LANDING_PROGRESS    0.50f       // Early Landing 시작점을 정의한다.
#define ROBOT_SWING_LANDING_APPROACH_M  0.03f       // 착지 감속을 시작할 지면 접근 거리를 정의한다.
#define ROBOT_SWING_LANDING_SPEED_MPS   0.12f       // 지면 접근 구간의 최대 하강 속도를 정의한다.
#define ROBOT_LATE_LANDING_SPEED_MPS    0.12f       // Late Landing 하강 속도를 정의한다.
#define ROBOT_LATE_INWARD_SPEED_MPS     0.096f      // Late Landing 안쪽 속도를 두 배로 높인다.
#define ROBOT_LATE_LANDING_MAX_DISTANCE_M 0.10f     // Late Landing 최대 하강 거리를 정의한다.
#define ROBOT_LATE_LANDING_MAX_TIME_S   \
    (ROBOT_LATE_LANDING_MAX_DISTANCE_M / ROBOT_LATE_LANDING_SPEED_MPS)  // 최대 탐색 시간을 계산한다.

#define ROBOT_COMMON_Z_RECOVERY_ENABLE      1U        // 추정 관절각 FK 기반 공통 Z 복구를 활성화한다.
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
#define ROBOT_JOINT_ADC_LPF_ALPHA        0.15f       // Median 출력의 저역통과 반영률을 정의한다.
#define ROBOT_JOINT_ADC_CORRECTION_GAIN  0.05f       // ADC 절대각의 상보 보정률을 정의한다.
#define ROBOT_JOINT_PWM_PREDICTION_RATE_RADPS \
    (300.0f * ROBOT_DEG_TO_RAD_F)                    // PWM 예측의 최대 관절 속도를 정의한다.
#define ROBOT_STARTUP_SENSOR_SETTLE_S    0.20f       // 서보 전원 안정 시간을 정의한다.
#define ROBOT_STARTUP_SENSOR_SAMPLES     20U         // 초기 관절각 평균 횟수를 정의한다.
#define ROBOT_STARTUP_ZERO_RATE_RADPS    (30.0f * ROBOT_DEG_TO_RAD_F)  // 초기 영점 정렬 속도를 정의한다.

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
