#include "measurement/measurement_stage5.h"

#include "common/robot_calibration.h"
#include "low_control/relay.h"
#include "low_control/servo_pwm.h"
#include "measurement/joint_calibration_measurement.h"
#include "measurement/measurement_debug.h"
#include "measurement/measurement_runner.h"
#include "sensor/mcp3008.h"

#include <stddef.h>
#include <string.h>

#define MEASUREMENT_STAGE5_CONTROL_PERIOD_MS        5U
#define MEASUREMENT_STAGE5_ADC_PERIOD_MS           20U
#define MEASUREMENT_STAGE5_POSITION_TIME_MS      3000U
#define MEASUREMENT_STAGE5_SAMPLE_START_MS       2000U
#define MEASUREMENT_STAGE5_RETURN_TIME_MS        1500U
#define MEASUREMENT_STAGE5_RELAY_OFF_TIME_MS      500U
#define MEASUREMENT_STAGE5_MINIMUM_RAW_SPAN        20U
#define MEASUREMENT_STAGE5_TEST_ANGLE_DEG         20.0f

typedef enum
{
    MEASUREMENT_STAGE5_NEGATIVE = 0,  // -20도 측정 단계를 나타낸다.
    MEASUREMENT_STAGE5_ZERO,          // 0도 측정 단계를 나타낸다.
    MEASUREMENT_STAGE5_POSITIVE,      // +20도 측정 단계를 나타낸다.
    MEASUREMENT_STAGE5_RETURN_ZERO,   // 다음 관절 전 0도 복귀를 나타낸다.
    MEASUREMENT_STAGE5_RELAY_OFF,     // 다리 전원 차단 구간을 나타낸다.
    MEASUREMENT_STAGE5_COMPLETE       // 전체 측정 완료를 나타낸다.
} MeasurementStage5_Phase_t;

static MCP3008_Handle_t measurement_adc;                            // MCP3008 통신 상태를 저장한다.
static MCP3008_Data_t measurement_adc_data;                         // 최근 ADC 24채널 값을 저장한다.
static ServoPwm_Handle_t measurement_servo;                         // 18개 PWM 출력 상태를 저장한다.
static JointCalibrationMeasurement_t measurement_joint;            // 관절별 세 위치 ADC를 저장한다.
static JointFeedback_Calibration_t measurement_table[ROBOT_JOINT_COUNT]; // 완성한 관절 보정표를 저장한다.
static MeasurementRunner_t measurement_runner;                     // 현재 실측 단계를 저장한다.
static float measurement_target_rad[ROBOT_JOINT_COUNT];             // 현재 18개 관절 목표각을 저장한다.
static float measurement_minimum_angle_rad[ROBOT_JOINT_COUNT];      // -20도 보정각을 저장한다.
static float measurement_zero_angle_rad[ROBOT_JOINT_COUNT];         // 0도 보정각을 저장한다.
static float measurement_maximum_angle_rad[ROBOT_JOINT_COUNT];      // +20도 보정각을 저장한다.
static int8_t measurement_direction[ROBOT_JOINT_COUNT];             // ADC 증가 방향을 저장한다.
static uint32_t measurement_last_control_ms;                        // 최근 PWM 갱신 시각을 저장한다.
static uint32_t measurement_last_adc_ms;                            // 최근 ADC 실행 시각을 저장한다.
static uint32_t measurement_phase_start_ms;                         // 현재 측정 단계 시작 시각을 저장한다.
static uint32_t measurement_sample_sum;                             // 현재 위치 ADC 합계를 저장한다.
static uint16_t measurement_sample_count;                           // 현재 위치 ADC 표본 수를 저장한다.
static uint8_t measurement_joint_index;                             // 현재 전체 관절 번호 0~17을 저장한다.
static MeasurementStage5_Phase_t measurement_phase;                // 현재 관절 측정 단계를 저장한다.
static bool measurement_initialized;                               // 5단계 실행 가능 여부를 저장한다.

/* 모든 다리 전원과 PWM을 정지한다. */
static void MeasurementStage5_StopOutputs(void)
{
    Relay_AllOff();                              // 서보 전원을 먼저 차단한다.
    ServoPwm_Stop(&measurement_servo);           // 18개 PWM 출력을 정지한다.
    g_measurement_debug.relay_state_mask = 0U;   // 전체 릴레이 OFF를 표시한다.
    g_measurement_debug.servo_test_output_active = false;  // 전체 출력 정지를 표시한다.
}

/* 현재 관절에서 오류가 발생하면 출력을 차단한다. */
static void MeasurementStage5_Fail(void)
{
    MeasurementStage5_StopOutputs();  // 오류 즉시 모든 서보 출력을 정지한다.
    measurement_initialized = false;  // 오류 후 자동 재실행을 막는다.
    g_measurement_debug.joint_calibration_failed = true;  // 보정 실패를 표시한다.
    g_measurement_debug.joint_calibration_error_joint =
        (uint8_t)(measurement_joint_index + 1U);  // 실패한 전체 관절 번호를 표시한다.
}

/* 현재 단계의 목표각과 디버그 상태를 갱신한다. */
static void MeasurementStage5_SetPhase(MeasurementStage5_Phase_t phase,
                                       uint32_t now_ms)
{
    float target_deg = 0.0f;  // 새 단계의 목표각을 저장한다.

    measurement_phase = phase;           // 새 측정 단계를 저장한다.
    measurement_phase_start_ms = now_ms;  // 새 단계 시작 시각을 저장한다.
    measurement_sample_sum = 0U;          // 이전 ADC 합계를 제거한다.
    measurement_sample_count = 0U;        // 이전 ADC 표본 수를 제거한다.

    if (phase == MEASUREMENT_STAGE5_NEGATIVE)
    {
        target_deg = -MEASUREMENT_STAGE5_TEST_ANGLE_DEG;  // 음의 보정각을 선택한다.
    }
    else if (phase == MEASUREMENT_STAGE5_POSITIVE)
    {
        target_deg = MEASUREMENT_STAGE5_TEST_ANGLE_DEG;   // 양의 보정각을 선택한다.
    }

    measurement_target_rad[measurement_joint_index] =
        target_deg * ROBOT_DEG_TO_RAD_F;  // 현재 관절 목표를 rad로 저장한다.

    if (phase == MEASUREMENT_STAGE5_RELAY_OFF)
    {
        Relay_AllOff();  // 다음 관절 전 다리 전원을 차단한다.
    }

    g_measurement_debug.joint_calibration_phase = (uint8_t)phase;  // 현재 단계를 표시한다.
    g_measurement_debug.joint_calibration_target_deg = (int16_t)target_deg;  // 현재 목표각을 표시한다.
    g_measurement_debug.joint_calibration_phase_elapsed_seconds = 0U;  // 단계 경과 시간을 초기화한다.
    g_measurement_debug.joint_calibration_sample_count = 0U;  // 평균 표본 수를 초기화한다.
    g_measurement_debug.relay_state_mask = Relay_GetStateMask();  // 현재 릴레이 상태를 표시한다.
}

/* 현재 관절이 속한 다리만 켜고 -20도 측정을 시작한다. */
static bool MeasurementStage5_StartJoint(uint32_t now_ms)
{
    const uint8_t leg = (uint8_t)(measurement_joint_index / ROBOT_JOINTS_PER_LEG);  // 현재 다리를 계산한다.

    if (!g_robot_calibration.relay_mapped[leg])
    {
        return false;
    }

    Relay_AllOff();                                             // 이전 다리 전원을 먼저 차단한다.
    Relay_On(g_robot_calibration.relay_for_leg[leg]);           // 현재 다리만 전원을 공급한다.
    g_measurement_debug.joint_calibration_active_leg =
        (uint8_t)(leg + 1U);                                    // 현재 다리를 1~6으로 표시한다.
    g_measurement_debug.joint_calibration_active_joint =
        (uint8_t)((measurement_joint_index % ROBOT_JOINTS_PER_LEG) + 1U);  // 현재 관절을 1~3으로 표시한다.
    MeasurementStage5_SetPhase(MEASUREMENT_STAGE5_NEGATIVE, now_ms);  // -20도 단계부터 시작한다.
    return true;
}

/* 현재 위치의 ADC 평균값을 해당 배열에 기록한다. */
static bool MeasurementStage5_CapturePosition(void)
{
    uint16_t average;  // 현재 위치의 ADC 평균값을 저장한다.

    if (measurement_sample_count == 0U)
    {
        return false;
    }

    average = (uint16_t)(measurement_sample_sum / measurement_sample_count);  // 표본 평균을 계산한다.
    if (measurement_phase == MEASUREMENT_STAGE5_NEGATIVE)
    {
        measurement_joint.minimum[measurement_joint_index] = average;          // -20도 ADC를 저장한다.
        g_measurement_debug.joint_minimum_raw[measurement_joint_index] = average;  // 디버거에 -20도 ADC를 표시한다.
    }
    else if (measurement_phase == MEASUREMENT_STAGE5_ZERO)
    {
        measurement_joint.zero[measurement_joint_index] = average;             // 0도 ADC를 저장한다.
        g_measurement_debug.joint_zero_raw[measurement_joint_index] = average;  // 디버거에 0도 ADC를 표시한다.
    }
    else if (measurement_phase == MEASUREMENT_STAGE5_POSITIVE)
    {
        measurement_joint.maximum[measurement_joint_index] = average;          // +20도 ADC를 저장한다.
        g_measurement_debug.joint_maximum_raw[measurement_joint_index] = average;  // 디버거에 +20도 ADC를 표시한다.
    }
    else
    {
        return false;
    }

    return true;
}

/* 현재 관절의 세 ADC 값으로 방향과 유효성을 확인한다. */
static bool MeasurementStage5_ValidateJoint(void)
{
    const uint16_t negative = measurement_joint.minimum[measurement_joint_index];  // -20도 ADC를 읽는다.
    const uint16_t zero = measurement_joint.zero[measurement_joint_index];          // 0도 ADC를 읽는다.
    const uint16_t positive = measurement_joint.maximum[measurement_joint_index];  // +20도 ADC를 읽는다.
    const uint16_t raw_min = (negative < positive) ? negative : positive;           // 작은 끝점 ADC를 선택한다.
    const uint16_t raw_max = (negative > positive) ? negative : positive;           // 큰 끝점 ADC를 선택한다.

    if (((uint16_t)(raw_max - raw_min) < MEASUREMENT_STAGE5_MINIMUM_RAW_SPAN) ||
        (zero <= raw_min) || (zero >= raw_max))
    {
        return false;
    }

    measurement_direction[measurement_joint_index] =
        (positive > negative) ? 1 : -1;  // 기구학 각도 증가에 따른 ADC 방향을 저장한다.
    g_measurement_debug.joint_calibration_direction[measurement_joint_index] =
        measurement_direction[measurement_joint_index];  // 디버거에 방향을 표시한다.
    return true;
}

/* 18개 관절의 세 위치 결과로 최종 보정표를 만든다. */
static bool MeasurementStage5_BuildTable(void)
{
    uint32_t joint;  // 보정각을 준비할 관절 번호를 저장한다.

    measurement_joint.minimum_captured = true;  // 전체 -20도 측정 완료를 표시한다.
    measurement_joint.zero_captured = true;     // 전체 0도 측정 완료를 표시한다.
    measurement_joint.maximum_captured = true;  // 전체 +20도 측정 완료를 표시한다.

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        measurement_minimum_angle_rad[joint] =
            -MEASUREMENT_STAGE5_TEST_ANGLE_DEG * ROBOT_DEG_TO_RAD_F;  // 음의 보정각을 넣는다.
        measurement_zero_angle_rad[joint] = 0.0f;                     // 영점 보정각을 넣는다.
        measurement_maximum_angle_rad[joint] =
             MEASUREMENT_STAGE5_TEST_ANGLE_DEG * ROBOT_DEG_TO_RAD_F;  // 양의 보정각을 넣는다.
    }

    return JointCalibrationMeasurement_Build(&measurement_joint,
                                              measurement_minimum_angle_rad,
                                              measurement_zero_angle_rad,
                                              measurement_maximum_angle_rad,
                                              measurement_direction,
                                              measurement_table);  // 18개 관절 보정표를 생성한다.
}

/* ADC 24채널을 읽고 현재 관절 표본을 누적한다. */
static void MeasurementStage5_UpdateAdc(uint32_t now_ms)
{
    const uint8_t leg = (uint8_t)(measurement_joint_index / ROBOT_JOINTS_PER_LEG);    // 현재 다리를 계산한다.
    const uint8_t joint = (uint8_t)(measurement_joint_index % ROBOT_JOINTS_PER_LEG);  // 현재 다리 관절을 계산한다.
    const MCP3008_InputMapping_t *mapping = &g_robot_calibration.adc[leg][joint];      // 현재 관절 ADC 채널을 선택한다.
    uint32_t device;   // 복사할 MCP3008 번호를 저장한다.
    uint32_t channel;  // 복사할 ADC 채널을 저장한다.
    uint16_t active_raw;  // 현재 관절 ADC를 저장한다.

    if ((now_ms - measurement_last_adc_ms) < MEASUREMENT_STAGE5_ADC_PERIOD_MS)
    {
        return;
    }
    measurement_last_adc_ms = now_ms;  // 이번 ADC 실행 시각을 저장한다.

    if (MCP3008_ReadAll(&measurement_adc, &measurement_adc_data) != HAL_OK)
    {
        g_measurement_debug.adc_driver_error_count = measurement_adc_data.error_count;        // ADC 오류 수를 갱신한다.
        g_measurement_debug.adc_last_error_device = measurement_adc_data.last_error_device;   // 실패 장치를 갱신한다.
        g_measurement_debug.adc_last_error_channel = measurement_adc_data.last_error_channel; // 실패 채널을 갱신한다.
        MeasurementStage5_Fail();  // ADC 오류 시 시험을 중단한다.
        return;
    }

    for (device = 0U; device < MCP3008_DEVICE_COUNT; ++device)
    {
        for (channel = 0U; channel < MCP3008_CHANNEL_COUNT; ++channel)
        {
            g_measurement_debug.adc_raw[device][channel] =
                measurement_adc_data.raw[device][channel];  // 최근 ADC Raw를 디버거에 복사한다.
        }
    }

    active_raw = measurement_adc_data.raw[mapping->device][mapping->channel];  // 현재 관절 ADC를 읽는다.
    g_measurement_debug.joint_calibration_active_raw = active_raw;             // 현재 ADC를 표시한다.
    g_measurement_debug.last_sample_ms = measurement_adc_data.mcu_time_ms;     // 최근 ADC 시각을 표시한다.
    g_measurement_debug.adc_update_count = measurement_adc_data.update_counter;  // ADC 갱신 횟수를 표시한다.

    if (((measurement_phase == MEASUREMENT_STAGE5_NEGATIVE) ||
         (measurement_phase == MEASUREMENT_STAGE5_ZERO) ||
         (measurement_phase == MEASUREMENT_STAGE5_POSITIVE)) &&
        ((now_ms - measurement_phase_start_ms) >= MEASUREMENT_STAGE5_SAMPLE_START_MS))
    {
        measurement_sample_sum += active_raw;   // 안정 구간 ADC를 누적한다.
        ++measurement_sample_count;             // 평균 표본 수를 증가시킨다.
        g_measurement_debug.joint_calibration_sample_count =
            measurement_sample_count;           // 현재 표본 수를 표시한다.
    }
}

/* SPI1 ADC와 18개 보정 PWM의 자동 관절센서 측정을 준비한다. */
bool MeasurementStage5_Init(SPI_HandleTypeDef *adc_spi,
                            TIM_HandleTypeDef *tim1,
                            TIM_HandleTypeDef *tim2,
                            TIM_HandleTypeDef *tim3,
                            TIM_HandleTypeDef *tim4,
                            TIM_HandleTypeDef *tim5,
                            TIM_HandleTypeDef *tim8)
{
    const ServoPwm_TimerBank_t timers = {tim1, tim2, tim3, tim4, tim5, tim8};  // CubeMX 타이머를 묶는다.
    uint32_t joint;  // 초기화할 관절 번호를 저장한다.
    uint32_t now_ms; // 초기화 완료 시각을 저장한다.

    MeasurementRunner_Init(&measurement_runner);                   // 전역 디버그값을 초기화한다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 0단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 1단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 2단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 3단계를 건너뛴다.
    (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 완료한 4단계를 건너뛴다.
    g_measurement_debug.calibration = g_robot_calibration;         // 앞 단계의 중앙 설정값을 유지한다.
    JointCalibrationMeasurement_Init(&measurement_joint);         // 관절 ADC 측정 상태를 준비한다.
    memset(&measurement_adc_data, 0, sizeof(measurement_adc_data)); // 이전 ADC 값을 제거한다.
    memset(measurement_table, 0, sizeof(measurement_table));       // 이전 보정표를 제거한다.
    memset(measurement_target_rad, 0, sizeof(measurement_target_rad));  // 전체 목표를 0도로 준비한다.
    memset(measurement_direction, 0, sizeof(measurement_direction));    // 이전 ADC 방향을 제거한다.
    Relay_Init();                                                  // 모든 다리 전원을 차단한다.
    ServoPwm_Init(&measurement_servo, &timers);                    // 18개 PWM 채널을 준비한다.
    measurement_initialized = false;                              // 초기화 완료 전 실행을 막는다.
    measurement_joint_index = 0U;                                 // 1번 다리 J1부터 시작한다.
    g_measurement_debug.joint_calibration_completed_count = 0U;   // 완료 관절 수를 초기화한다.
    g_measurement_debug.joint_calibration_error_joint = 0U;       // 오류 관절을 초기화한다.
    g_measurement_debug.joint_calibration_complete = false;       // 전체 완료 전으로 표시한다.
    g_measurement_debug.joint_calibration_failed = false;         // 오류 없음으로 시작한다.
    g_measurement_debug.servo_test_output_active = false;         // PWM 비활성 상태로 시작한다.
    g_measurement_debug.relay_state_mask = 0U;                    // 전체 릴레이 OFF를 표시한다.

    if ((adc_spi == NULL) || (tim1 == NULL) || (tim2 == NULL) ||
        (tim3 == NULL) || (tim4 == NULL) || (tim5 == NULL) || (tim8 == NULL))
    {
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 잘못된 CubeMX Handle을 기록한다.
        return false;
    }
    if (MCP3008_Init(&measurement_adc, adc_spi) != HAL_OK)
    {
        g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_ADC;  // ADC 초기화 실패를 기록한다.
        return false;
    }

    for (joint = 0U; joint < ROBOT_JOINT_COUNT; ++joint)
    {
        if (!ServoPwm_SetCalibration(&measurement_servo,
                                     (uint8_t)joint,
                                     &g_robot_calibration.servo[joint]))
        {
            MeasurementStage5_StopOutputs();  // 잘못된 서보 보정값이면 출력을 막는다.
            g_measurement_debug.initialization_error =
                MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 서보 보정값 오류를 기록한다.
            return false;
        }
    }

    if (ServoPwm_Start(&measurement_servo) != HAL_OK)
    {
        MeasurementStage5_StopOutputs();  // PWM 시작 실패 시 출력을 정리한다.
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // PWM 시작 실패를 기록한다.
        return false;
    }
    if (!ServoPwm_WriteAngles(&measurement_servo, measurement_target_rad))
    {
        MeasurementStage5_StopOutputs();  // 보정 0도 계산 실패 시 출력을 정리한다.
        g_measurement_debug.initialization_error =
            MEASUREMENT_DEBUG_INIT_INVALID_HANDLE;  // 보정 0도 실패를 기록한다.
        return false;
    }

    now_ms = HAL_GetTick();                       // 첫 관절 시작 시각을 읽는다.
    measurement_last_control_ms = now_ms;         // PWM 갱신 기준 시각을 저장한다.
    measurement_last_adc_ms = now_ms;             // ADC 실행 기준 시각을 저장한다.
    measurement_initialized = true;               // 주기 실행을 허용한다.
    g_measurement_debug.servo_test_output_active = true;  // PWM 실행 상태를 표시한다.
    if (!MeasurementStage5_StartJoint(now_ms))
    {
        MeasurementStage5_Fail();  // 릴레이 매핑 오류 시 시험을 중단한다.
        return false;
    }

    g_measurement_debug.initialization_error = MEASUREMENT_DEBUG_INIT_OK;  // 초기화 성공을 기록한다.
    g_measurement_debug.initialization_complete = true;                   // 디버거에 준비 완료를 표시한다.
    return true;
}

/* 한 관절씩 -20·0·+20도 ADC를 평균하여 보정표를 만든다. */
void MeasurementStage5_Process(void)
{
    const uint32_t now_ms = HAL_GetTick();  // 현재 실측 시각을 저장한다.
    const uint32_t elapsed_ms = now_ms - measurement_phase_start_ms;  // 현재 단계 경과 시간을 계산한다.

    if (!measurement_initialized)
    {
        return;
    }

    if ((now_ms - measurement_last_control_ms) >= MEASUREMENT_STAGE5_CONTROL_PERIOD_MS)
    {
        measurement_last_control_ms = now_ms;  // 이번 PWM 실행 시각을 저장한다.
        if (!ServoPwm_WriteAngles(&measurement_servo, measurement_target_rad))
        {
            MeasurementStage5_Fail();  // PWM 변환 실패 시 시험을 중단한다.
            return;
        }
    }

    MeasurementStage5_UpdateAdc(now_ms);  // ADC와 현재 위치 평균을 갱신한다.
    if (!measurement_initialized)
    {
        return;
    }

    g_measurement_debug.joint_calibration_phase_elapsed_seconds =
        (uint8_t)(elapsed_ms / 1000U);  // 현재 단계 경과 초를 표시한다.

    if ((measurement_phase == MEASUREMENT_STAGE5_NEGATIVE) ||
        (measurement_phase == MEASUREMENT_STAGE5_ZERO) ||
        (measurement_phase == MEASUREMENT_STAGE5_POSITIVE))
    {
        if (elapsed_ms < MEASUREMENT_STAGE5_POSITION_TIME_MS)
        {
            return;
        }
        if (!MeasurementStage5_CapturePosition())
        {
            MeasurementStage5_Fail();  // 평균 표본이 없으면 시험을 중단한다.
            return;
        }

        if (measurement_phase == MEASUREMENT_STAGE5_NEGATIVE)
        {
            MeasurementStage5_SetPhase(MEASUREMENT_STAGE5_ZERO, now_ms);  // 0도 측정으로 이동한다.
        }
        else if (measurement_phase == MEASUREMENT_STAGE5_ZERO)
        {
            MeasurementStage5_SetPhase(MEASUREMENT_STAGE5_POSITIVE, now_ms);  // +20도 측정으로 이동한다.
        }
        else
        {
            if (!MeasurementStage5_ValidateJoint())
            {
                MeasurementStage5_Fail();  // ADC 방향이나 변화 폭 오류 시 중단한다.
                return;
            }
            ++g_measurement_debug.joint_calibration_completed_count;  // 현재 관절 완료를 기록한다.
            MeasurementStage5_SetPhase(MEASUREMENT_STAGE5_RETURN_ZERO, now_ms);  // 0도로 복귀한다.
        }
        return;
    }

    if (measurement_phase == MEASUREMENT_STAGE5_RETURN_ZERO)
    {
        if (elapsed_ms >= MEASUREMENT_STAGE5_RETURN_TIME_MS)
        {
            MeasurementStage5_SetPhase(MEASUREMENT_STAGE5_RELAY_OFF, now_ms);  // 다리 전원을 잠시 끈다.
        }
        return;
    }

    if ((measurement_phase == MEASUREMENT_STAGE5_RELAY_OFF) &&
        (elapsed_ms >= MEASUREMENT_STAGE5_RELAY_OFF_TIME_MS))
    {
        ++measurement_joint_index;  // 다음 전체 관절 번호로 이동한다.
        if (measurement_joint_index >= ROBOT_JOINT_COUNT)
        {
            if (!MeasurementStage5_BuildTable())
            {
                measurement_joint_index = ROBOT_JOINT_COUNT - 1U;  // 오류 표시 관절을 범위 안에 둔다.
                MeasurementStage5_Fail();  // 최종 보정표 생성 실패를 표시한다.
                return;
            }

            MeasurementStage5_StopOutputs();                         // 완료 후 모든 출력을 정지한다.
            measurement_phase = MEASUREMENT_STAGE5_COMPLETE;         // 전체 완료 단계를 저장한다.
            measurement_initialized = false;                         // 완료 후 주기 동작을 멈춘다.
            g_measurement_debug.joint_minimum_captured = true;       // 전체 -20도 기록 완료를 표시한다.
            g_measurement_debug.joint_zero_captured = true;          // 전체 0도 기록 완료를 표시한다.
            g_measurement_debug.joint_maximum_captured = true;       // 전체 +20도 기록 완료를 표시한다.
            g_measurement_debug.joint_calibration_phase =
                (uint8_t)MEASUREMENT_STAGE5_COMPLETE;                 // 전체 완료 단계를 표시한다.
            g_measurement_debug.joint_calibration_complete = true;   // 최종 보정 완료를 표시한다.
            (void)MeasurementRunner_CompleteCurrent(&measurement_runner);  // 다음 실측 단계로 이동한다.
            return;
        }

        if (!MeasurementStage5_StartJoint(now_ms))
        {
            MeasurementStage5_Fail();  // 다음 릴레이 시작 실패 시 시험을 중단한다.
        }
    }
}
