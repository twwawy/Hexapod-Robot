#include "test/rl_controller_test.h"

#include "high_control/rl_controller.h"

#include <math.h>
#include <string.h>

/* 정상 세션의 첫 관측과 다리 출력 후보를 준비한다. */
static RobotRlAction_t RlControllerTest_Prepare(RlController_Handle_t *handle)
{
    RobotRlAction_t action = {0};  // 사용하지 않는 자세축과 잔차를 0으로 준비한다.

    RlController_Init(handle);                            // 시험 상태를 초기화한다.
    RlController_BeginSession(handle, 31U);               // 명시적인 시험 세션을 시작한다.
    RlController_SetPlan(handle, 41U, 0x15U, true);       // 1·3·5 다리의 다음 계획을 광고한다.
    RlController_RecordObservation(handle, 200U, 1000U);  // 실제 관측 시각을 기록한다.

    action.session_id = 31U;                   // 현재 세션에 맞는 입력을 준비한다.
    action.sequence = 100U;                    // 첫 출력 순번을 준비한다.
    action.observation_sequence = 200U;        // 공개한 원관측을 참조한다.
    action.plan_id = 41U;                      // 공개한 계획을 참조한다.
    action.swing_mask = 0x15U;                 // 공개한 이륙 다리를 참조한다.
    action.leg_plan_valid = true;              // 다리 계획을 포함한 출력을 준비한다.
    action.posture_reference_rad.roll = 0.1f;  // 보존 여부를 확인할 자세 목표를 준비한다.
    action.residual[0].dx = 0.01f;             // 보존 여부를 확인할 다리 잔차를 준비한다.
    return action;
}

/* 비활성 입력과 이전 세션 재전송 및 종료 후 잔류를 검사한다. */
static bool RlControllerTest_CheckSession(void)
{
    RlController_Handle_t handle;  // 세션 수명 시험 상태를 저장한다.
    RobotRlAction_t action = {0};  // 세션이 없는 입력을 준비한다.
    RobotRlAction_t output;        // 수신 출력을 저장한다.

    RlController_Init(&handle);              // 비활성 상태를 준비한다.
    if ((RlController_Submit(&handle, &action, 0U) != RL_SUBMIT_INACTIVE) ||
        RlController_GetAction(&handle, &output, 0U))
    {
        return false;
    }
    RlController_BeginSession(&handle, 0U);  // 예약 번호로 세션 시작을 시도한다.
    if (handle.session_active)
    {
        return false;
    }

    action = RlControllerTest_Prepare(&handle);  // 정상 관측과 출력을 준비한다.
    if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_ACCEPTED)
    {
        return false;
    }
    RlController_EndSession(&handle);            // 수락한 출력이 있는 세션을 종료한다.
    if (handle.has_received_action || handle.action_valid ||
        RlController_GetAction(&handle, &output, 1011U))
    {
        return false;
    }

    RlController_BeginSession(&handle, 32U);  // 다른 번호의 새 운용 세션을 시작한다.
    if (RlController_Submit(&handle, &action, 1020U) != RL_SUBMIT_SESSION_MISMATCH)
    {
        return false;
    }
    action.session_id = 32U;                  // 이전 관측을 새 세션으로 재포장한다.
    return (RlController_Submit(&handle, &action, 1020U) == RL_SUBMIT_OBSERVATION) &&
           (RlController_Submit(&handle, NULL, 1020U) == RL_SUBMIT_INVALID_ARGUMENT) &&
           (RlController_Submit(NULL, &action, 1020U) == RL_SUBMIT_INVALID_ARGUMENT);
}

/* 출력 순번의 래핑과 재전송 차단 및 독립된 수신 만료를 검사한다. */
static bool RlControllerTest_CheckSequenceAndTimeout(void)
{
    RlController_Handle_t handle;                                // 순번과 신선도 시험 상태를 저장한다.
    RobotRlAction_t action = RlControllerTest_Prepare(&handle);  // 정상 후보를 준비한다.
    RobotRlAction_t output;                                      // 만료 경계의 출력을 저장한다.

    action.sequence = UINT16_MAX;  // 래핑 직전의 첫 출력을 준비한다.
    if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_ACCEPTED)
    {
        return false;
    }
    action.sequence = 0U;          // 래핑 직후의 다음 출력을 준비한다.
    if ((RlController_Submit(&handle, &action, 1020U) != RL_SUBMIT_ACCEPTED) ||
        (RlController_Submit(&handle, &action, 1050U) != RL_SUBMIT_SEQUENCE))
    {
        return false;
    }
    action.sequence = UINT16_MAX;  // 직전 출력을 늦게 재전송한다.
    if (RlController_Submit(&handle, &action, 1050U) != RL_SUBMIT_SEQUENCE)
    {
        return false;
    }
    action.sequence = 0x8000U;     // 순서를 구분할 수 없는 반 범위 차이를 준비한다.
    if ((RlController_Submit(&handle, &action, 1050U) != RL_SUBMIT_SEQUENCE) ||
        (handle.last_action_ms != 1020U) || (handle.action_observation_ms != 1000U) ||
        (handle.accepted_count != 2U) || (handle.rejected_count != 3U))
    {
        return false;
    }
    if (!RlController_GetAction(&handle, &output, 1020U + ROBOT_RL_ACTION_TIMEOUT_MS) ||
        (output.sequence != 0U) || (output.residual[0].dx != 0.01f) ||
        RlController_GetAction(&handle, &output, 1021U + ROBOT_RL_ACTION_TIMEOUT_MS))
    {
        return false;
    }
    return handle.has_received_action && handle.action_valid &&
           !output.leg_plan_valid && (output.residual[0].dx == 0.0f);
}

/* 관측 시각 경계와 관측 순번 래핑 및 최근 이력 보존을 검사한다. */
static bool RlControllerTest_CheckObservation(void)
{
    RlController_Handle_t handle;                                // 원관측 검증 상태를 저장한다.
    RobotRlAction_t action = RlControllerTest_Prepare(&handle);  // 정상 후보를 준비한다.
    uint32_t index;                                              // 관측 이력 생성 횟수를 저장한다.

    action.observation_sequence = 201U;                    // 공개하지 않은 미래 관측을 참조한다.
    if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_OBSERVATION)
    {
        return false;
    }
    action.observation_sequence = 200U;                    // 최초 공개한 관측을 다시 참조한다.
    RlController_RecordObservation(&handle, 200U, 1050U);  // 중복 관측으로 시각 갱신을 시도한다.
    RlController_RecordObservation(&handle, 199U, 1050U);  // 역순 관측의 추가를 시도한다.
    if (RlController_Submit(&handle, &action,
                            1000U + ROBOT_RL_OBSERVATION_MAX_AGE_MS) != RL_SUBMIT_ACCEPTED)
    {
        return false;
    }
    action.sequence++;                                     // 같은 오래된 관측으로 새 출력을 준비한다.
    if (RlController_Submit(&handle, &action,
                            1001U + ROBOT_RL_OBSERVATION_MAX_AGE_MS) != RL_SUBMIT_OBSERVATION)
    {
        return false;
    }

    action = RlControllerTest_Prepare(&handle);               // 최근 이력 시험을 다시 준비한다.
    for (index = 1U; index <= 12U; ++index)
    {
        RlController_RecordObservation(&handle, (uint16_t)(200U + index),
                                        1000U + 5U * index);  // 200 Hz의 뒤따른 관측을 기록한다.
    }
    if (RlController_Submit(&handle, &action, 1060U) != RL_SUBMIT_ACCEPTED)
    {
        return false;
    }

    RlController_BeginSession(&handle, 31U);                                // 순번과 타이머 래핑 시험 상태를 준비한다.
    RlController_SetPlan(&handle, 41U, 0x15U, true);                        // 래핑 시험의 다리 계획을 광고한다.
    RlController_RecordObservation(&handle, UINT16_MAX, UINT32_MAX - 20U);  // 타이머 래핑 직전 관측을 기록한다.
    RlController_RecordObservation(&handle, 0U, UINT32_MAX - 15U);          // 순번 래핑 후 관측을 기록한다.
    action.observation_sequence = UINT16_MAX;                               // 래핑 이전 원관측을 참조한다.
    if ((RlController_Submit(&handle, &action, 5U) != RL_SUBMIT_ACCEPTED) ||
        (handle.action_observation_ms != UINT32_MAX - 20U))
    {
        return false;
    }
    action.sequence++;                                                      // 래핑 후 원관측의 새 출력을 준비한다.
    action.observation_sequence = 0U;                                       // 래핑 후 공개한 관측을 참조한다.
    return RlController_Submit(&handle, &action, 10U) == RL_SUBMIT_ACCEPTED;
}

/* 다리 계획의 관측 일치와 자세 전용 출력의 엄격한 분리를 검사한다. */
static bool RlControllerTest_CheckPlan(void)
{
    RlController_Handle_t handle;                                // 계획 일치 시험 상태를 저장한다.
    RobotRlAction_t action = RlControllerTest_Prepare(&handle);  // 정상 후보를 준비한다.
    RobotRlAction_t output;                                      // 계획 전환 뒤 자세 출력을 저장한다.

    action.swing_mask = 0x2AU;                             // 계획과 다른 이륙 다리를 준비한다.
    if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_PLAN)
    {
        return false;
    }
    action.swing_mask = 0x15U;                             // 원래 계획의 다리 비트를 복구한다.
    if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_ACCEPTED)
    {
        return false;
    }
    RlController_SetPlan(&handle, 42U, 0x2AU, true);       // 다음 보행 계획으로 전환한다.
    if (!RlController_GetAction(&handle, &output, 1020U) || output.leg_plan_valid ||
        (output.plan_id != 0U) || (output.swing_mask != 0U) ||
        (output.residual[0].dx != 0.0f) || (output.posture_reference_rad.roll != 0.1f))
    {
        return false;
    }
    action.sequence++;                                     // 다음 계획의 새 출력을 준비한다.
    action.plan_id = 42U;                                  // 현재 계획 번호로 입력을 변경한다.
    action.swing_mask = 0x2AU;                             // 현재 이륙 다리로 입력을 변경한다.
    if (RlController_Submit(&handle, &action, 1020U) != RL_SUBMIT_PLAN)
    {
        return false;
    }
    RlController_RecordObservation(&handle, 201U, 1020U);  // 새 계획이 실제 공개된 관측을 기록한다.
    action.observation_sequence = 201U;                    // 새 계획의 원관측을 참조한다.
    if (RlController_Submit(&handle, &action, 1030U) != RL_SUBMIT_ACCEPTED)
    {
        return false;
    }

    action.sequence++;                                    // 자세 전용 새 출력을 준비한다.
    action.leg_plan_valid = false;                        // 다리 계획 전달을 해제한다.
    if (RlController_Submit(&handle, &action, 1030U) != RL_SUBMIT_PLAN)
    {
        return false;
    }
    action.plan_id = 0U;                                  // 자세 전용 계획 번호를 정리한다.
    action.swing_mask = 0U;                               // 자세 전용 다리 비트를 정리한다.
    if (RlController_Submit(&handle, &action, 1030U) != RL_SUBMIT_VALUE)
    {
        return false;
    }
    memset(action.residual, 0, sizeof(action.residual));  // 자세 전용 출력의 잔차를 제거한다.
    RlController_SetPlan(&handle, 0U, 0U, false);         // 보행 계획이 없는 정지 상태를 준비한다.
    if (RlController_Submit(&handle, &action, 1030U) != RL_SUBMIT_ACCEPTED)
    {
        return false;
    }
    RlController_SetPlan(&handle, 1U, 0x80U, true);       // 존재하지 않는 다리 계획을 입력한다.
    return !handle.plan_valid && (handle.swing_mask == 0U);
}

/* 각 출력축의 NaN·무한대·범위 초과와 경계값을 검사한다. */
static bool RlControllerTest_CheckValues(void)
{
    RlController_Handle_t handle;  // 수치 범위 시험 상태를 저장한다.
    RobotRlAction_t action;        // 현재 축의 출력 후보를 저장한다.
    uint32_t axis;                 // 자세와 다리 출력축 번호를 저장한다.

    for (axis = 0U; axis < 2U + 4U * ROBOT_LEG_COUNT; ++axis)
    {
        float *value;  // 검사할 출력 스칼라를 저장한다.
        float limit;   // 해당 스칼라의 양방향 한계를 저장한다.

        action = RlControllerTest_Prepare(&handle);                                 // 축마다 독립적인 정상 상태를 준비한다.
        if (axis < 2U)
        {
            value = (axis == 0U) ? &action.posture_reference_rad.roll :
                                   &action.posture_reference_rad.pitch;             // 두 자세 출력 중 하나를 선택한다.
            limit = (axis == 0U) ? ROBOT_RL_MAX_ROLL_RAD : ROBOT_RL_MAX_PITCH_RAD;  // 자세축 한계를 선택한다.
        }
        else
        {
            RobotLegResidual_t *residual = &action.residual[(axis - 2U) / 4U];      // 검사할 다리 잔차를 선택한다.

            switch ((axis - 2U) % 4U)
            {
                case 0U: value = &residual->dx; limit = ROBOT_RL_MAX_DX_M; break;  // 착지 X 잔차를 선택한다.
                case 1U: value = &residual->dy; limit = ROBOT_RL_MAX_DY_M; break;  // 착지 Y 잔차를 선택한다.
                case 2U: value = &residual->dz; limit = ROBOT_RL_MAX_DZ_M; break;  // 착지 Z 잔차를 선택한다.
                default: value = &residual->dh; limit = ROBOT_RL_MAX_DH_M; break;  // Swing 높이 잔차를 선택한다.
            }
        }

        *value = NAN;              // 계산 불능 출력을 준비한다.
        if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_VALUE)
        {
            return false;
        }
        *value = INFINITY;         // 무한대 출력을 준비한다.
        if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_VALUE)
        {
            return false;
        }
        *value = limit + 0.001f;   // 양수 범위 초과 출력을 준비한다.
        if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_VALUE)
        {
            return false;
        }
        *value = -limit - 0.001f;  // 음수 범위 초과 출력을 준비한다.
        if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_VALUE)
        {
            return false;
        }
        *value = -limit;           // 허용한 음수 경계값을 준비한다.
        if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_ACCEPTED)
        {
            return false;
        }
        action.sequence++;         // 양수 경계값의 새 출력을 준비한다.
        *value = limit;            // 허용한 양수 경계값을 준비한다.
        if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_ACCEPTED)
        {
            return false;
        }
    }

    action = RlControllerTest_Prepare(&handle);  // 미사용 Yaw축의 검증 상태를 준비한다.
    action.posture_reference_rad.yaw = 0.001f;   // 허용하지 않은 Yaw 자세 명령을 준비한다.
    if (RlController_Submit(&handle, &action, 1010U) != RL_SUBMIT_VALUE)
    {
        return false;
    }
    action.posture_reference_rad.yaw = NAN;      // 미사용 자세축의 NaN을 준비한다.
    return RlController_Submit(&handle, &action, 1010U) == RL_SUBMIT_VALUE;
}

/* 통신 해석 뒤 RL 입력이 제어기에 도달하기 전의 검증을 실행한다. */
bool RlControllerTest_Run(void)
{
    return RlControllerTest_CheckSession() &&
           RlControllerTest_CheckSequenceAndTimeout() &&
           RlControllerTest_CheckObservation() &&
           RlControllerTest_CheckPlan() &&
           RlControllerTest_CheckValues();  // 세션과 계획 및 수치 검증을 함께 확인한다.
}
