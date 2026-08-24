#ifndef ROBOT_CALIBRATION_H
#define ROBOT_CALIBRATION_H

#include "low_control/relay.h"
#include "low_control/servo_pwm.h"
#include "sensor/foot_pressure.h"
#include "sensor/imu.h"
#include "sensor/joint_feedback.h"
#include "sensor/mcp3008.h"
#include "user_command/user_command.h"

#include <stdbool.h>

typedef struct
{
    IMU_Calibration_t imu;                                                        // WT931 축과 Offset을 저장한다.
    bool imu_calibrated;                                                          // WT931 실측 완료 여부를 저장한다.
    MCP3008_InputMapping_t adc[MCP3008_LEG_COUNT][MCP3008_LEG_INPUT_COUNT];        // ADC 입력 배치를 저장한다.
    bool adc_mapping_calibrated;                                                  // ADC 배치 확인 여부를 저장한다.
    JointFeedback_Calibration_t joint[ROBOT_JOINT_COUNT];                         // 관절센서 보정값을 저장한다.
    FootPressure_Calibration_t pressure[ROBOT_PRESSURE_COUNT];                    // 압력센서 임계값을 저장한다.
    ServoPwm_Calibration_t servo[ROBOT_JOINT_COUNT];                              // 서보 출력 보정값을 저장한다.
    Relay_Channel_t relay_for_leg[ROBOT_LEG_COUNT];                               // 다리별 릴레이를 저장한다.
    bool relay_mapped[ROBOT_LEG_COUNT];                                           // 릴레이 대응 확인 여부를 저장한다.
    UserCommand_ChannelCalibration_t crsf[USER_COMMAND_USED_CHANNELS];            // CRSF 채널 보정값을 저장한다.
} RobotCalibration_t;

extern const RobotCalibration_t g_robot_calibration;  // 실측 후 직접 채울 중앙 설정 테이블을 공개한다.

bool RobotCalibration_IsComplete(const RobotCalibration_t *calibration);  // 모든 실측값의 완료 여부를 검사한다.

bool RobotCalibration_Apply(const RobotCalibration_t *calibration,
                            IMU_Handle_t *imu,
                            MCP3008_Handle_t *adc,
                            JointFeedback_Handle_t *joint,
                            FootPressure_Handle_t *pressure,
                            ServoPwm_Handle_t *servo,
                            UserCommand_Handle_t *user_command);  // 완료된 중앙 설정값을 각 모듈에 적용한다.

#endif
