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

/* 지정한 위상의 Swing 그룹을 비트로 만든다. */
static uint8_t GaitManager_SwingMask(uint32_t phase_index)
{
    uint8_t mask = 0U;  // Swing 다리 비트를 준비한다.
    uint32_t leg;       // 확인할 다리 번호를 저장한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (GaitManager_IsSwingLeg(phase_index, leg))
        {
            mask |= (uint8_t)(1U << leg);  // 선택한 위상의 Swing 다리를 표시한다.
        }
    }

    return mask;
}

/* 현재 위상에서 이미 지면을 밀고 있는 발을 비트로 만든다. */
static uint8_t GaitManager_SupportMask(const GaitManager_Handle_t *handle)
{
    uint8_t mask = 0U;  // 현재 지지발 비트를 준비한다.
    uint32_t leg;       // 확인할 다리 번호를 저장한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (!GaitManager_IsSwingLeg(handle->phase_index, leg) || handle->landed[leg])
        {
            mask |= (uint8_t)(1U << leg);  // 기존 Stance와 Early Landing 발을 포함한다.
        }
    }
    return mask;
}

/* 지정한 지지발이 모두 확정 접촉인지 확인한다. */
static bool GaitManager_AllMaskContact(uint8_t mask,
                                       const bool contact[ROBOT_LEG_COUNT])
{
    uint32_t leg;  // 확인할 다리 번호를 저장한다.

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (((mask & (uint8_t)(1U << leg)) != 0U) && !contact[leg])
        {
            return false;  // 한 지지발이라도 미접촉이면 대기를 유지한다.
        }
    }
    return true;
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
RobotGaitPhase_t GaitManager_StepContacts(
    GaitManager_Handle_t *handle,
    bool normal_mode_enable,
    bool tripod_enable,
    bool phase_validation_done,
    bool phase_validation_accepted,
    RobotTripodMode_t tripod_mode,
    float recovery_progress,
    const bool contact[ROBOT_LEG_COUNT],
    const bool contact_raw[ROBOT_LEG_COUNT])
{
    RobotGaitPhase_t output;  // 이번 다리 상태 출력을 저장한다.
    uint32_t leg;             // 처리할 다리 번호를 저장한다.

    memset(&output, 0, sizeof(output));  // 기본 STANCE 출력을 준비한다.

    if ((handle == NULL) || (contact == NULL) || (contact_raw == NULL))
    {
        return output;
    }

    if (!tripod_enable)
    {
        handle->late_landing_hold = false;  // 입력 해제에서 탐색 한계 정지를 재가동 가능 상태로 만든다.
    }

    if (tripod_mode == ROBOT_TRIPOD_NORMAL)
    {
        if (!normal_mode_enable)
        {
            handle->initialized = false;       // 수동 모드 밖에서 정상 위상을 초기화한다.
            handle->phase_index = 0U;          // 다음 수동 진입을 첫 Tripod로 준비한다.
            handle->phase_cycle_count = 0U;    // 현재 위상 주기를 제거한다.
            handle->phase_time_s = 0.0f;       // 진행 중인 위상 시간을 제거한다.
            handle->run_enable = false;        // 내부 보행을 비활성화한다.
            handle->stop_pending = false;      // 남은 정지 예약을 제거한다.
            handle->resume_phase = false;      // 다음 보행을 최초 위상으로 준비한다.
            handle->next_phase_enable = false; // 다음 위상 Enable을 제거한다.
            handle->next_phase_locked = false; // 다음 위상 결정을 제거한다.
            handle->late_landing_hold = false;  // 수동 모드 밖에서 탐색 정지를 제거한다.
            handle->support_recovery_mask = 0U;       // 지지발 재착지 대상을 제거한다.
            handle->support_recovery_active = false;  // 지지발 재착지 상태를 제거한다.

            memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));  // 비접촉 이력을 제거한다.
            memset(handle->touchdown_seen, 0, sizeof(handle->touchdown_seen));  // 첫 접촉 이력을 제거한다.
            memset(handle->landed, 0, sizeof(handle->landed));                // 착지 이력을 제거한다.
        }
        else if (handle->late_landing_hold)
        {
            handle->run_enable = false;   // 입력 해제 전까지 보행을 정지한다.
            handle->initialized = false;  // 진행 중인 정상 위상을 비활성화한다.
            handle->stop_pending = false; // 중복 정지 예약을 제거한다.
        }
        else if (tripod_enable)
        {
            if (!handle->run_enable)
            {
                handle->run_enable = true;      // 새 정상 보행을 시작한다.
                handle->initialized = false;    // 첫 위상을 다시 초기화한다.
                if (!handle->resume_phase)
                {
                    handle->phase_index = 0U;   // 최초 보행은 1·3·5 Swing부터 시작한다.
                }
                handle->phase_cycle_count = 0U;    // 첫 위상 주기를 초기화한다.
                handle->phase_time_s = 0.0f;       // 위상 시간을 초기화한다.
                handle->next_phase_enable = false; // 확정 전 Enable을 초기화한다.
                handle->next_phase_locked = false; // 첫 속도 확정을 준비한다.
                handle->support_recovery_mask = 0U;       // 이전 재착지 대상을 제거한다.
                handle->support_recovery_active = false;  // 새 보행의 재착지를 준비한다.
                memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));  // 비접촉 이력을 제거한다.
                memset(handle->touchdown_seen, 0, sizeof(handle->touchdown_seen));  // 첫 접촉 이력을 제거한다.
                memset(handle->landed, 0, sizeof(handle->landed));                // 착지 이력을 제거한다.
            }

            handle->stop_pending = false;  // 활성 명령에서 정지 요청을 해제한다.
        }
        else if (handle->run_enable && !handle->initialized &&
                 !handle->next_phase_locked)
        {
            handle->run_enable = false;        // 확정 전 해제된 첫 보행을 취소한다.
            handle->phase_cycle_count = 0U;    // 사용하지 않은 위상 주기를 제거한다.
            handle->phase_time_s = 0.0f;       // 사용하지 않은 위상 시간을 제거한다.
            handle->stop_pending = false;      // 실행되지 않은 정지 예약을 제거한다.
            handle->support_recovery_mask = 0U;       // 취소한 재착지 대상을 제거한다.
            handle->support_recovery_active = false;  // 취소한 재착지 상태를 제거한다.
        }
        else if (handle->run_enable)
        {
            handle->stop_pending = true;   // 현재 Swing 착지 후 정지를 예약한다.
        }
        else
        {
            handle->initialized = false;       // 정지 중 위상 시간만 초기화한다.
            handle->phase_cycle_count = 0U;    // 정지 중 위상 주기를 제거한다.
            handle->phase_time_s = 0.0f;       // 정지 중 위상 시간을 제거한다.
            handle->stop_pending = false;      // 완료한 정지 예약을 제거한다.
            handle->next_phase_enable = false; // 완료한 Enable 결정을 제거한다.
            handle->next_phase_locked = false; // 완료한 위상 결정을 제거한다.
            handle->support_recovery_mask = 0U;       // 정지 중 재착지 대상을 제거한다.
            handle->support_recovery_active = false;  // 정지 중 재착지를 비활성화한다.

            memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));  // 비접촉 이력을 제거한다.
            memset(handle->touchdown_seen, 0, sizeof(handle->touchdown_seen));  // 첫 접촉 이력을 제거한다.
            memset(handle->landed, 0, sizeof(handle->landed));                // 착지 이력을 제거한다.
        }
    }
    else
    {
        handle->initialized = false;       // 특수 착지에서 정상 위상을 초기화한다.
        handle->phase_index = 0U;          // 다음 정상 보행을 최초 그룹으로 준비한다.
        handle->phase_cycle_count = 0U;    // 정상 위상 주기를 제거한다.
        handle->phase_time_s = 0.0f;       // 정상 위상 시간을 제거한다.
        handle->run_enable = false;        // 정상 보행을 비활성화한다.
        handle->stop_pending = false;      // 정상 보행 정지 예약을 제거한다.
        handle->resume_phase = false;      // 다음 정상 보행을 최초 위상으로 준비한다.
        handle->next_phase_enable = false; // 다음 위상 Enable을 제거한다.
        handle->next_phase_locked = false; // 다음 위상 결정을 제거한다.
        handle->support_recovery_mask = 0U;       // 특수 모드에서 재착지 대상을 제거한다.
        handle->support_recovery_active = false;  // 특수 모드에서 재착지를 비활성화한다.

        memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));  // 비접촉 이력을 제거한다.
        memset(handle->touchdown_seen, 0, sizeof(handle->touchdown_seen));  // 첫 접촉 이력을 제거한다.
        memset(handle->landed, 0, sizeof(handle->landed));                // 착지 이력을 제거한다.
    }

    if ((tripod_mode == ROBOT_TRIPOD_NORMAL) && handle->run_enable)
    {
        float progress;             // 현재 위상 진행률을 저장한다.
        bool all_swing_landed = true;  // Swing 그룹 전체 착지를 저장한다.

        if (!handle->initialized && !handle->next_phase_locked)
        {
            handle->next_phase_enable = tripod_enable;  // 현재 보행 Enable을 시작 명령으로 고정한다.
            handle->next_phase_locked = true;            // 첫 위상 검사 요청을 표시한다.
            output.next_phase_preview = true;            // 첫 위상 세 지점 검사를 요청한다.
            output.next_phase_startup = !handle->resume_phase;  // 최초 반 보폭 여부를 전달한다.
            output.next_phase_swing_mask =
                GaitManager_SwingMask(handle->phase_index);  // 시작할 Tripod 그룹을 전달한다.
            output.startup_phase = !handle->resume_phase;  // 최초 보행의 반 보폭 여부를 유지한다.
            output.waiting_start = true;                   // 한 제어 주기 검사 중 발 목표를 고정한다.
        }
        else if (!handle->initialized &&
                 (!phase_validation_done || !phase_validation_accepted))
        {
            if (phase_validation_done)
            {
                handle->run_enable = false;        // 첫 경로가 위험하면 보행을 시작하지 않는다.
                handle->next_phase_enable = false; // 실패한 Enable 결정을 제거한다.
                handle->next_phase_locked = false; // 다음 첫 위상 검사를 준비한다.
            }
            else
            {
                output.startup_phase = !handle->resume_phase;  // 검사 완료까지 첫 위상 종류를 유지한다.
                output.waiting_start = true;                   // 검사 완료까지 서보 목표를 고정한다.
            }
        }
        else
        {
            if (!handle->initialized)
            {
                handle->initialized = true;  // 세 지점 검사 후 정상 위상을 활성화한다.
                if (!handle->resume_phase)
                {
                    handle->phase_index = 0U;  // 최초 보행 위상을 선택한다.
                }
                handle->phase_cycle_count = 0U;    // 첫 위상 주기를 초기화한다.
                handle->phase_time_s = 0.0f;       // 첫 위상 시간을 초기화한다.
                handle->next_phase_enable = false; // 다음 위상 Enable을 준비한다.
                handle->next_phase_locked = false; // 다음 착륙 검사를 준비한다.
                handle->support_recovery_mask = 0U;       // 첫 위상의 재착지 대상을 비운다.
                handle->support_recovery_active = false;  // 첫 위상의 재착지를 대기 상태로 둔다.
            }
            else
            {
                if (handle->support_recovery_active &&
                    GaitManager_AllMaskContact(handle->support_recovery_mask, contact))
                {
                    handle->support_recovery_mask = 0U;       // 재접촉을 마친 지지발을 비운다.
                    handle->support_recovery_active = false;  // 기존 위상 진행을 재개한다.
                }

                if (!handle->support_recovery_active)
                {
                    const uint8_t support_mask = GaitManager_SupportMask(handle);  // 현재 지면을 미는 발을 선택한다.

                    if (!GaitManager_AllMaskContact(support_mask, contact))
                    {
                        handle->support_recovery_mask = support_mask;  // 재접촉할 기존 지지발을 고정한다.
                        handle->support_recovery_active = true;        // 현재 위상 진행을 일시정지한다.
                    }
                }

                if (!handle->support_recovery_active)
                {
                    handle->phase_cycle_count++;  // 안정된 지지에서만 위상 주기를 누적한다.
                    handle->phase_time_s =
                        (float)handle->phase_cycle_count *
                        ROBOT_CONTROL_PERIOD_S;  // 정수 주기로 위상 시간을 계산한다.
                }
            }

            progress = fminf(fmaxf(handle->phase_time_s / ROBOT_GAIT_PHASE_TIME_S,
                                   0.0f), 1.0f);  // 위상 진행률을 계산한다.

            if (handle->support_recovery_active)
            {
                output.support_recovery_mask = handle->support_recovery_mask;  // 재접촉 대상 발을 전달한다.
                output.support_recovery_active = true;                        // 일시정지 상태를 전달한다.

                for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
                {
                    const bool support = (handle->support_recovery_mask &
                                          (uint8_t)(1U << leg)) != 0U;  // 기존 지지발 여부를 확인한다.

                    if (!support || contact[leg])
                    {
                        output.state[leg] = ROBOT_LEG_HOLD;  // 나머지 발을 현재 위치에 고정한다.
                    }
                    else if (contact_raw[leg])
                    {
                        output.state[leg] = ROBOT_LEG_TOUCHDOWN_CANDIDATE;  // 접촉 확인 중 하강을 멈춘다.
                    }
                    else
                    {
                        output.state[leg] = ROBOT_LEG_LATE_LANDING;  // 뜬 지지발만 지면을 다시 탐색한다.
                    }
                    output.progress[leg] = progress;  // 정지한 기존 위상 진행률을 유지한다.
                }
                output.startup_phase = (handle->phase_index == 0U) &&
                                       !handle->resume_phase;  // 정지 중 현재 위상 종류를 유지한다.
            }
            else
            {
                for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
                {
                    if (!GaitManager_IsSwingLeg(handle->phase_index, leg))
                    {
                        continue;
                    }

                    if (!contact[leg])
                    {
                        handle->airborne_seen[leg] = true;  // 확정 비접촉을 확인한다.
                    }

                    if ((handle->airborne_seen[leg] || handle->stop_pending) &&
                        contact_raw[leg] &&
                        (progress >= ROBOT_EARLY_LANDING_PROGRESS))
                    {
                        handle->touchdown_seen[leg] = true;  // 첫 접촉 뒤 Swing 재상승을 금지한다.
                    }

                    if ((handle->airborne_seen[leg] || handle->stop_pending) &&
                        contact[leg] && (progress >= ROBOT_EARLY_LANDING_PROGRESS))
                    {
                        handle->landed[leg] = true;  // 5 ms 연속 접촉에서 착지를 확정한다.
                    }

                    if (!handle->landed[leg])
                    {
                        all_swing_landed = false;  // 아직 착지하지 않은 Swing을 기록한다.
                    }
                }

                if (all_swing_landed &&
                    GaitManager_AllMaskContact(
                        (uint8_t)((1U << ROBOT_LEG_COUNT) - 1U), contact))
                {
                    bool preview_started = false;  // 이번 주기의 새 검사 요청 여부를 저장한다.

                    if (!handle->next_phase_locked)
                    {
                        handle->next_phase_enable = tripod_enable;  // 여섯 발 착륙 시점의 Enable을 고정한다.
                        handle->next_phase_locked = true;            // 다음 위상 결정을 고정한다.
                        preview_started = handle->next_phase_enable; // 활성 명령에서 새 검사를 표시한다.
                        if (preview_started)
                        {
                            output.next_phase_preview = true;             // 다음 위상 세 지점 검사를 요청한다.
                            output.next_phase_startup = false;             // 반복 위상 전체 보폭을 선택한다.
                            output.next_phase_swing_mask =
                                GaitManager_SwingMask(handle->phase_index + 1U);  // 반대 Tripod 그룹을 전달한다.
                        }
                    }

                    if (!handle->next_phase_enable ||
                        (!preview_started && phase_validation_done &&
                         !phase_validation_accepted))
                    {
                        handle->run_enable = false;         // 정지 입력 또는 검사 실패면 착지 후 정지한다.
                        handle->stop_pending = false;       // 완료한 정지 예약을 제거한다.
                        handle->initialized = false;        // 현재 정상 위상을 종료한다.
                        handle->phase_index++;              // 재시작할 다음 Tripod 그룹을 보존한다.
                        handle->phase_cycle_count = 0U;     // 완료한 위상 주기를 제거한다.
                        handle->phase_time_s = 0.0f;        // 완료한 위상 시간을 제거한다.
                        handle->resume_phase = true;        // 정지 자세에서 다음 그룹 재개를 준비한다.
                        handle->next_phase_enable = false;  // 완료한 Enable 결정을 제거한다.
                        handle->next_phase_locked = false;  // 다음 위상 결정을 준비한다.
                    }
                    else if (!preview_started && phase_validation_done &&
                             phase_validation_accepted)
                    {
                        handle->phase_index++;              // 다음 Tripod 그룹으로 전환한다.
                        handle->phase_cycle_count = 0U;     // 새 위상 주기를 초기화한다.
                        handle->phase_time_s = 0.0f;        // 새 위상 시간을 초기화한다.
                        handle->next_phase_enable = false;  // 다음 위상 Enable을 준비한다.
                        handle->next_phase_locked = false;  // 다음 착륙 검사를 준비한다.
                        progress = 0.0f;                    // 새 위상 진행률을 0으로 둔다.
                    }
                    else
                    {
                        output.waiting_start = true;  // 세 지점 검사 완료까지 여섯 발을 고정한다.
                    }

                    handle->support_recovery_mask = 0U;       // 완료한 위상의 재착지 대상을 제거한다.
                    handle->support_recovery_active = false;  // 새 위상의 재착지를 준비한다.
                    if (!handle->run_enable || !handle->next_phase_locked)
                    {
                        memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));  // 새 위상의 비접촉 이력을 초기화한다.
                        memset(handle->touchdown_seen, 0, sizeof(handle->touchdown_seen));  // 새 위상의 첫 접촉 이력을 초기화한다.
                        memset(handle->landed, 0, sizeof(handle->landed));                // 새 위상의 착지 이력을 초기화한다.
                    }
                }

                if (handle->run_enable)
                {
                    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
                    {
                        const bool swing = GaitManager_IsSwingLeg(handle->phase_index, leg);  // 현재 Swing 그룹을 확인한다.
                        const bool candidate = swing && !handle->landed[leg] &&
                            handle->touchdown_seen[leg] &&
                            contact_raw[leg] &&
                            (progress >= ROBOT_EARLY_LANDING_PROGRESS);  // 접촉 확인 대기 조건을 계산한다.

                        if (!swing || handle->landed[leg])
                        {
                            output.state[leg] = ROBOT_LEG_STANCE;  // 지지 또는 확정 착지 상태를 출력한다.
                        }
                        else if (candidate)
                        {
                            output.state[leg] = ROBOT_LEG_TOUCHDOWN_CANDIDATE;  // 접촉 확인까지 현재 발을 고정한다.
                        }
                        else if (handle->touchdown_seen[leg] || (progress >= 1.0f))
                        {
                            output.state[leg] = ROBOT_LEG_LATE_LANDING;  // 첫 접촉 취소 뒤 재상승 없이 지면을 탐색한다.
                        }
                        else
                        {
                            output.state[leg] = ROBOT_LEG_SWING;  // 정상 Swing 상태를 출력한다.
                        }

                        output.progress[leg] = progress;  // 다리별 진행률을 출력한다.
                    }

                    output.startup_phase = (handle->phase_index == 0U) &&
                                           !handle->resume_phase;  // 최초 반 보폭 위상만 표시한다.
                }
            }
        }
    }
    else if (tripod_enable && !handle->late_landing_hold &&
             (tripod_mode == ROBOT_TRIPOD_LAND_ALL))
    {
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            output.state[leg] = contact[leg] ? ROBOT_LEG_STANCE :
                                (contact_raw[leg] ? ROBOT_LEG_TOUCHDOWN_CANDIDATE :
                                                    ROBOT_LEG_LATE_LANDING);  // 접촉 확인 전 발만 내린다.
            output.progress[leg] = 1.0f;  // 정상 Swing이 끝났음을 표시한다.
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

    for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
    {
        if (output.state[leg] == ROBOT_LEG_LATE_LANDING)
        {
            const bool exhausted =
                (handle->late_landing_time_s[leg] >=
                 ROBOT_LATE_LANDING_MAX_TIME_S);  // 직전 주기까지 최대 탐색을 완료했는지 확인한다.

            if (!exhausted)
            {
                handle->late_landing_time_s[leg] = fminf(
                    handle->late_landing_time_s[leg] + ROBOT_CONTROL_PERIOD_S,
                    ROBOT_LATE_LANDING_MAX_TIME_S);  // 이번 5 ms 탐색 시간을 누적한다.
            }
            output.late_landing_exhausted[leg] = exhausted;  // 100 mm 완료 다음 주기에 정지를 표시한다.
            output.late_landing_stop = output.late_landing_stop ||
                                       output.late_landing_exhausted[leg];  // 한 다리의 한계 도달을 전체 정지로 올린다.
        }
        else if (output.state[leg] != ROBOT_LEG_TOUCHDOWN_CANDIDATE)
        {
            handle->late_landing_time_s[leg] = 0.0f;  // Late Landing 밖에서 탐색 시간을 초기화한다.
        }
    }

    if (output.late_landing_stop)
    {
        handle->late_landing_hold = true;  // 탐색 한계에서 전원 유지 정지를 Latch한다.
        output.late_landing_hold = true;   // 이번 주기부터 발 목표 고정을 요청한다.

        if (tripod_mode == ROBOT_TRIPOD_NORMAL)
        {
            handle->run_enable = false;         // 현재 정상 보행을 정지한다.
            handle->initialized = false;        // 완료하지 못한 위상을 비활성화한다.
            handle->stop_pending = false;       // 남은 정지 예약을 제거한다.
            handle->resume_phase = true;        // 같은 Tripod 그룹 재시작을 준비한다.
            handle->phase_cycle_count = 0U;     // 완료하지 못한 위상 주기를 제거한다.
            handle->phase_time_s = 0.0f;        // 완료하지 못한 위상 시간을 제거한다.
            handle->next_phase_enable = false;  // 다음 위상 Enable 결정을 제거한다.
            handle->next_phase_locked = false;  // 다음 위상 결정을 준비한다.
            handle->support_recovery_mask = 0U;       // 중단한 재착지 대상을 제거한다.
            handle->support_recovery_active = false;  // 중단한 재착지 상태를 제거한다.

            memset(handle->airborne_seen, 0, sizeof(handle->airborne_seen));  // 비접촉 이력을 제거한다.
            memset(handle->touchdown_seen, 0, sizeof(handle->touchdown_seen));  // 첫 접촉 이력을 제거한다.
            memset(handle->landed, 0, sizeof(handle->landed));                // 착지 이력을 제거한다.
        }
    }
    else if (handle->late_landing_hold && tripod_enable)
    {
        output.waiting_start = true;      // 입력 해제 전 모든 발 목표를 유지한다.
        output.late_landing_hold = true;  // 탐색 한계 정지 상태를 전달한다.
    }

    output.enabled_internal = (tripod_mode == ROBOT_TRIPOD_NORMAL)
                            ? handle->run_enable
                            : (tripod_enable &&
                               !handle->late_landing_hold);  // 실제 내부 Enable을 반환한다.
    return output;
}

/* 확정 접촉만 사용하는 시험·단순 호출을 접촉 상태기에 연결한다. */
RobotGaitPhase_t GaitManager_Step(GaitManager_Handle_t *handle,
                                  bool normal_mode_enable,
                                  bool tripod_enable,
                                  bool phase_validation_done,
                                  bool phase_validation_accepted,
                                  RobotTripodMode_t tripod_mode,
                                  float recovery_progress,
                                  const bool contact[ROBOT_LEG_COUNT])
{
    bool confirmed[ROBOT_LEG_COUNT];  // 단순 호출용 확정 접촉을 저장한다.
    bool raw[ROBOT_LEG_COUNT];        // 단순 호출용 접촉 후보를 저장한다.
    uint32_t leg;                     // 보완할 지지발 번호를 저장한다.

    if (contact == NULL)
    {
        memset(confirmed, 0, sizeof(confirmed));  // NULL 입력을 안전한 미접촉으로 바꾼다.
    }
    else
    {
        memcpy(confirmed, contact, sizeof(confirmed));  // 기존 단순 접촉 입력을 복사한다.
    }
    memcpy(raw, confirmed, sizeof(raw));  // 단순 호출에서는 후보와 확정을 같게 둔다.

    if ((handle != NULL) && (tripod_mode == ROBOT_TRIPOD_NORMAL) &&
        handle->run_enable && handle->initialized)
    {
        for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
        {
            if (!GaitManager_IsSwingLeg(handle->phase_index, leg) || handle->landed[leg])
            {
                confirmed[leg] = true;  // 기존 시험 호출에서는 지지발 접촉을 정상으로 가정한다.
                raw[leg] = true;        // 지지발 접촉 후보도 정상으로 가정한다.
            }
        }
    }

    return GaitManager_StepContacts(handle,
                                    normal_mode_enable,
                                    tripod_enable,
                                    phase_validation_done,
                                    phase_validation_accepted,
                                    tripod_mode,
                                    recovery_progress,
                                    confirmed,
                                    raw);  // 단순 호출을 접촉 후보·확정 입력으로 변환한다.
}
