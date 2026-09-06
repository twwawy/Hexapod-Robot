#include "high_control/rl_controller.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define RL_LEG_MASK ((uint8_t)((1U << ROBOT_LEG_COUNT) - 1U))  // 존재하는 다리의 비트 범위를 정의한다.

/* 16비트 순번의 래핑을 허용하고 중복과 역순을 구분한다. */
static bool RlController_SequenceNewer(uint16_t next, uint16_t previous)
{
    const uint16_t delta = (uint16_t)(next - previous);  // 래핑을 포함한 순번 차이를 계산한다.

    return (delta != 0U) && (delta < 0x8000U);  // 반 범위 안의 새 순번만 허가한다.
}

/* 마지막 제출 결과와 거부 횟수를 함께 기록한다. */
static RlController_SubmitResult_t RlController_Reject(RlController_Handle_t *handle,
                                                       RlController_SubmitResult_t result)
{
    if (handle != NULL)
    {
        handle->last_result = result;  // 거부 원인을 기록한다.
        handle->rejected_count++;      // 거부한 입력 횟수를 누적한다.
    }
    return result;
}

/* 유한한 스칼라가 허용한 양방향 범위에 있는지 검사한다. */
static bool RlController_ValueValid(float value, float limit)
{
    return isfinite(value) && (fabsf(value) <= limit);  // NaN과 무한대 및 범위 초과를 거부한다.
}

/* 자세와 여섯 다리 잔차의 수치 범위를 검사한다. */
static bool RlController_ActionValuesValid(const RobotRlAction_t *action)
{
    uint32_t leg;  // 검사할 다리 번호를 저장한다.

    if (!RlController_ValueValid(action->posture_reference_rad.roll, ROBOT_RL_MAX_ROLL_RAD) ||
        !RlController_ValueValid(action->posture_reference_rad.pitch, ROBOT_RL_MAX_PITCH_RAD) ||
        !RlController_ValueValid(action->posture_reference_rad.yaw, 0.0f))
    {
        return false;
    }

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        const RobotLegResidual_t *residual = &action->residual[leg];  // 현재 다리 잔차를 선택한다.

        if (!RlController_ValueValid(residual->dx, ROBOT_RL_MAX_DX_M) ||
            !RlController_ValueValid(residual->dy, ROBOT_RL_MAX_DY_M) ||
            !RlController_ValueValid(residual->dz, ROBOT_RL_MAX_DZ_M) ||
            !RlController_ValueValid(residual->dh, ROBOT_RL_MAX_DH_M))
        {
            return false;
        }
        if (!action->leg_plan_valid &&
            ((residual->dx != 0.0f) || (residual->dy != 0.0f) ||
             (residual->dz != 0.0f) || (residual->dh != 0.0f)))
        {
            return false;
        }
    }
    return true;
}

/* 최근 공개한 관측 중 입력이 참조한 순번을 찾는다. */
static const RlController_Observation_t *RlController_FindObservation(
    const RlController_Handle_t *handle, uint16_t sequence, uint32_t now_ms)
{
    uint32_t index;  // 검사할 관측 슬롯을 저장한다.

    for (index = 0U; index < ROBOT_RL_OBSERVATION_HISTORY_COUNT; ++index)
    {
        const RlController_Observation_t *observation = &handle->observation[index];  // 최근 관측 슬롯을 선택한다.

        if (observation->valid && (observation->sequence == sequence) &&
            ((uint32_t)(now_ms - observation->timestamp_ms) <= ROBOT_RL_OBSERVATION_MAX_AGE_MS))
        {
            return observation;
        }
    }
    return NULL;
}

/* 운용 세션과 수신 이력을 모두 초기화한다. */
void RlController_Init(RlController_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));        // 이전 세션의 모든 입력을 제거한다.
        handle->last_result = RL_SUBMIT_INACTIVE;  // 첫 세션 시작 전 비활성 상태를 표시한다.
    }
}

/* 외부에서 발급한 0이 아닌 번호로 새 세션을 시작한다. */
void RlController_BeginSession(RlController_Handle_t *handle, uint32_t session_id)
{
    if (handle == NULL)
    {
        return;
    }

    RlController_Init(handle);                    // 이전 관측과 출력 순번을 폐기한다.
    handle->session_id = session_id;              // 현재 세션 번호를 저장한다.
    handle->session_active = (session_id != 0U);  // 예약 번호 0의 세션을 차단한다.
}

/* 세션을 종료하고 모든 수신 후보를 폐기한다. */
void RlController_EndSession(RlController_Handle_t *handle)
{
    RlController_Init(handle);  // 재진입 전에 새로운 관측을 요구한다.
}

/* 다리 출력이 일치해야 할 현재 보행 계획을 설정한다. */
void RlController_SetPlan(RlController_Handle_t *handle,
                          uint16_t plan_id, uint8_t swing_mask, bool valid)
{
    if ((handle == NULL) || !handle->session_active)
    {
        return;
    }

    handle->plan_valid = valid && (swing_mask != 0U) &&
                         ((swing_mask & (uint8_t)~RL_LEG_MASK) == 0U);  // 존재하는 다리의 계획만 허가한다.
    handle->plan_id = handle->plan_valid ? plan_id : 0U;                // 유효한 계획 번호만 광고한다.
    handle->swing_mask = handle->plan_valid ? swing_mask : 0U;          // 무효 계획의 다리 비트를 제거한다.
}

/* 실제 공개한 새 관측의 시각과 당시 보행 계획을 저장한다. */
void RlController_RecordObservation(RlController_Handle_t *handle,
                                    uint16_t sequence, uint32_t now_ms)
{
    RlController_Observation_t *observation;  // 기록할 최근 관측 슬롯을 저장한다.

    if ((handle == NULL) || !handle->session_active ||
        (handle->has_observation &&
         !RlController_SequenceNewer(sequence, handle->latest_observation_sequence)))
    {
        return;
    }

    observation = &handle->observation[handle->observation_write_index];  // 다음 관측 슬롯을 선택한다.
    observation->sequence = sequence;                                     // 공개한 관측 순번을 저장한다.
    observation->timestamp_ms = now_ms;                                   // 실제 공개한 로컬 시각을 저장한다.
    observation->plan_id = handle->plan_id;                               // 관측에 포함한 계획 번호를 저장한다.
    observation->swing_mask = handle->swing_mask;                         // 관측에 포함한 이륙 다리를 저장한다.
    observation->plan_valid = handle->plan_valid;                         // 관측의 다리 계획 유효성을 저장한다.
    observation->valid = true;                                            // 완성한 관측 슬롯을 활성화한다.

    handle->observation_write_index = (uint8_t)((handle->observation_write_index + 1U) %
                                               ROBOT_RL_OBSERVATION_HISTORY_COUNT);  // 순환 이력의 다음 위치로 이동한다.
    handle->latest_observation_sequence = sequence;                                  // 중복 관측의 시각 갱신을 차단한다.
    handle->has_observation = true;                                                  // 다음 관측부터 순번을 비교한다.
}

/* 새 출력을 검증한 후 마지막 정상 출력으로 한 번에 교체한다. */
RlController_SubmitResult_t RlController_Submit(RlController_Handle_t *handle,
                                               const RobotRlAction_t *action,
                                               uint32_t now_ms)
{
    const RlController_Observation_t *observation;  // 출력이 참조한 원관측을 저장한다.

    if ((handle == NULL) || (action == NULL))
    {
        return RlController_Reject(handle, RL_SUBMIT_INVALID_ARGUMENT);
    }
    if (!handle->session_active)
    {
        return RlController_Reject(handle, RL_SUBMIT_INACTIVE);
    }
    if (action->session_id != handle->session_id)
    {
        return RlController_Reject(handle, RL_SUBMIT_SESSION_MISMATCH);
    }
    if (handle->action_valid &&
        !RlController_SequenceNewer(action->sequence, handle->action.sequence))
    {
        return RlController_Reject(handle, RL_SUBMIT_SEQUENCE);
    }

    observation = RlController_FindObservation(handle, action->observation_sequence, now_ms);  // 최근 실제 공개한 관측을 찾는다.
    if (observation == NULL)
    {
        return RlController_Reject(handle, RL_SUBMIT_OBSERVATION);
    }
    if (action->leg_plan_valid)
    {
        if (!handle->plan_valid || !observation->plan_valid ||
            (action->plan_id != handle->plan_id) || (action->swing_mask != handle->swing_mask) ||
            (action->plan_id != observation->plan_id) || (action->swing_mask != observation->swing_mask))
        {
            return RlController_Reject(handle, RL_SUBMIT_PLAN);
        }
    }
    else if ((action->plan_id != 0U) || (action->swing_mask != 0U))
    {
        return RlController_Reject(handle, RL_SUBMIT_PLAN);
    }
    if (!RlController_ActionValuesValid(action))
    {
        return RlController_Reject(handle, RL_SUBMIT_VALUE);
    }

    handle->action = *action;                                   // 검증을 마친 전체 출력을 교체한다.
    handle->last_action_ms = now_ms;                            // 정상 출력의 수락 시각을 저장한다.
    handle->action_observation_ms = observation->timestamp_ms;  // 고정 다리 계획의 원관측 시각을 전달한다.
    handle->action_valid = true;                                // 정상 출력의 존재를 표시한다.
    handle->has_received_action = true;                         // 첫 출력 대기 상태를 해제한다.
    handle->accepted_count++;                                   // 수락한 출력 횟수를 누적한다.
    handle->last_result = RL_SUBMIT_ACCEPTED;                   // 마지막 제출 성공을 기록한다.
    return RL_SUBMIT_ACCEPTED;
}

/* 신선한 출력을 복사하고 이전 보행 계획의 잔차 전달을 차단한다. */
bool RlController_GetAction(const RlController_Handle_t *handle,
                            RobotRlAction_t *action, uint32_t now_ms)
{
    if (action == NULL)
    {
        return false;
    }

    memset(action, 0, sizeof(*action));  // 시간 초과 시 잔류 출력을 전달하지 않는다.
    if ((handle == NULL) || !handle->session_active || !handle->action_valid ||
        ((uint32_t)(now_ms - handle->last_action_ms) > ROBOT_RL_ACTION_TIMEOUT_MS))
    {
        return false;
    }

    *action = handle->action;                                   // 신선한 자세와 다리 출력을 복사한다.
    if (action->leg_plan_valid &&
        (!handle->plan_valid || (action->plan_id != handle->plan_id) ||
         (action->swing_mask != handle->swing_mask)))
    {
        action->leg_plan_valid = false;                         // 이전 계획의 다리 출력을 차단한다.
        action->plan_id = 0U;                                   // 자세 전용 출력으로 계획을 정리한다.
        action->swing_mask = 0U;                                // 자세 전용 출력의 이륙 다리를 제거한다.
        memset(action->residual, 0, sizeof(action->residual));  // 검증한 자세만 계속 전달한다.
    }
    return true;
}
