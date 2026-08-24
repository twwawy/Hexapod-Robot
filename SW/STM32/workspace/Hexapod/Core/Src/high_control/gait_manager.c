#include "high_control/gait_manager.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* 현재 위상에서 Swing 다리인지 확인한다. */
static bool GaitManager_IsSwingLeg(uint32_t phase_index, uint32_t leg)
{
    const bool group_135 = ((leg % 2U) == 0U);  // 1·3·5 그룹 여부를 계산한다.
    return ((phase_index % 2U) == 0U) ? group_135 : !group_135;  // 위상에 맞는 Swing 그룹을 반환한다.
}

/* Tripod 상태를 정지 상태로 초기화한다. */
void GaitManager_Init(GaitManager_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));  // 이전 위상과 접촉 상태를 제거한다.
    }
}

/* Enable·모드·접촉에 따라 여섯 다리 상태와 진행률을 갱신한다. */
RobotGaitPhase_t GaitManager_Step(GaitManager_Handle_t *handle,
                                  bool tripod_enable,
                                  RobotTripodMode_t tripod_mode,
                                  float recovery_progress,
                                  const bool contact[ROBOT_LEG_COUNT])
{
    RobotGaitPhase_t output;  // 이번 다리 상태 출력을 저장한다.
    uint32_t leg;             // 처리할 다리 번호를 저장한다.

    memset(&output, 0, sizeof(output));  // 기본 STANCE 출력을 준비한다.

    if ((handle == NULL) || (contact == NULL))
    {
        return output;
    }

    if (tripod_mode == ROBOT_TRIPOD_NORMAL)
    {
        if (tripod_enable)
        {
            if (!handle->run_enable)
            {
                handle->run_enable = true;      // 새 정상 보행을 시작한다.
                handle->initialized = false;    // 첫 위상을 다시 초기화한다.
                handle->phase_index = 0U;       // 1·3·5 Swing부터 시작한다.
                handle->phase_time_s = 0.0f;    // 위상 시간을 초기화한다.
                memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));  // 비접촉 이력을 제거한다.
                memset(handle->landed, 0, sizeof(handle->landed));                // 착지 이력을 제거한다.
            }

            handle->stop_pending = false;  // 활성 명령에서 정지 요청을 해제한다.
        }
        else if (handle->run_enable)
        {
            handle->stop_pending = true;   // 현재 Swing 착지 후 정지를 예약한다.
        }
        else
        {
            handle->initialized = false;   // 완전 정지 상태를 초기화한다.
            handle->phase_index = 0U;
            handle->phase_time_s = 0.0f;
            handle->stop_pending = false;
            memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));
            memset(handle->landed, 0, sizeof(handle->landed));
        }
    }
    else
    {
        handle->initialized = false;  // 특수 착지에서 정상 위상을 초기화한다.
        handle->phase_index = 0U;
        handle->phase_time_s = 0.0f;
        handle->run_enable = false;
        handle->stop_pending = false;
        memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));
        memset(handle->landed, 0, sizeof(handle->landed));
    }

    if ((tripod_mode == ROBOT_TRIPOD_NORMAL) && handle->run_enable)
    {
        float progress;             // 현재 위상 진행률을 저장한다.
        bool all_swing_landed = true;  // Swing 그룹 전체 착지를 저장한다.

        if (!handle->initialized)
        {
            handle->initialized = true;  // 정상 위상을 활성화한다.
            handle->phase_index = 0U;    // 첫 위상을 선택한다.
            handle->phase_time_s = 0.0f; // 첫 위상 시간을 초기화한다.
        }
        else
        {
            handle->phase_time_s += ROBOT_CONTROL_PERIOD_S;  // 현재 위상 시간을 누적한다.
        }

        progress = fminf(fmaxf(handle->phase_time_s / ROBOT_GAIT_PHASE_TIME_S,
                               0.0f), 1.0f);  // 위상 진행률을 계산한다.

        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            if (!GaitManager_IsSwingLeg(handle->phase_index, leg))
            {
                continue;
            }

            if (!contact[leg])
            {
                handle->airborne_seen[leg] = true;  // 실제 비접촉을 확인한다.
            }

            if ((handle->airborne_seen[leg] || handle->stop_pending) &&
                contact[leg] && (progress >= ROBOT_EARLY_LANDING_PROGRESS))
            {
                handle->landed[leg] = true;  // 진행률 50% 이후 착지를 확정한다.
            }

            if (!handle->landed[leg])
            {
                all_swing_landed = false;  // 아직 착지하지 않은 Swing을 기록한다.
            }
        }

        if ((progress >= 1.0f) && all_swing_landed)
        {
            if (handle->stop_pending)
            {
                handle->run_enable = false;   // 모든 Swing 착지 후 보행을 정지한다.
                handle->stop_pending = false;
                handle->initialized = false;
                handle->phase_index = 0U;
                handle->phase_time_s = 0.0f;
            }
            else
            {
                handle->phase_index++;        // 다음 Tripod 그룹으로 전환한다.
                handle->phase_time_s = 0.0f;  // 새 위상 시간을 초기화한다.
                progress = 0.0f;              // 새 위상 진행률을 0으로 둔다.
            }

            memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));  // 새 위상의 비접촉 이력을 초기화한다.
            memset(handle->landed, 0, sizeof(handle->landed));                // 새 위상의 착지 이력을 초기화한다.
        }

        if (handle->run_enable)
        {
            for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
            {
                if (!GaitManager_IsSwingLeg(handle->phase_index, leg) || handle->landed[leg])
                {
                    output.state[leg] = ROBOT_LEG_STANCE;  // 지지 또는 조기 착지 상태를 출력한다.
                }
                else if (progress >= 1.0f)
                {
                    output.state[leg] = ROBOT_LEG_LATE_LANDING;  // 정상 시간 이후 지면을 탐색한다.
                }
                else
                {
                    output.state[leg] = ROBOT_LEG_SWING;  // 정상 Swing 상태를 출력한다.
                }

                output.progress[leg] = progress;  // 다리별 진행률을 출력한다.
            }

            output.startup_phase = (handle->phase_index == 0U);  // 첫 위상을 표시한다.
        }
    }
    else if (tripod_enable && (tripod_mode == ROBOT_TRIPOD_LAND_ALL))
    {
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            output.state[leg] = contact[leg] ? ROBOT_LEG_STANCE : ROBOT_LEG_LATE_LANDING;  // 미접촉 발만 내린다.
            output.progress[leg] = 1.0f;                                                    // 정상 Swing이 끝났음을 표시한다.
        }
    }
    else if (tripod_enable &&
             ((tripod_mode == ROBOT_TRIPOD_RECOVERY_135) ||
              (tripod_mode == ROBOT_TRIPOD_RECOVERY_246)))
    {
        const bool recover_135 = (tripod_mode == ROBOT_TRIPOD_RECOVERY_135);  // 복구 그룹을 선택한다.

        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            const bool group_135 = ((leg % 2U) == 0U);  // 현재 다리 그룹을 계산한다.

            if (group_135 == recover_135)
            {
                output.state[leg] = ROBOT_LEG_RECOVERY_SWING;  // 선택 그룹을 복구 Swing으로 둔다.
                output.progress[leg] = fminf(fmaxf(recovery_progress, 0.0f), 1.0f);  // 복구 진행률을 제한한다.
            }
        }
    }

    output.enabled_internal = (tripod_mode == ROBOT_TRIPOD_NORMAL)
                            ? handle->run_enable
                            : tripod_enable;  // 실제 내부 Enable을 반환한다.
    return output;
}
