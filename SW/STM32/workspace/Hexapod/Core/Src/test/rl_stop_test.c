#include "test/rl_stop_test.h"

#include "high_control/gait_manager.h"

#include <string.h>

/* 정상 접촉으로 첫 Tripod 위상을 시작한다. */
static bool RlStopTest_Start(GaitManager_Handle_t *handle,
                             bool contact[ROBOT_LEG_COUNT])
{
    uint32_t cycle;                                          // 시작 대기와 검사 주기를 저장한다.

    GaitManager_Init(handle);                                              // 새 정상 보행 상태를 준비한다.
    memset(contact, 1, ROBOT_LEG_COUNT * sizeof(contact[0]));              // 모든 지지발 접촉을 준비한다.
    for (cycle = 0U; cycle < ROBOT_GAIT_START_DELAY_CYCLES + 4U; ++cycle)
    {
        (void)GaitManager_StepContacts(handle, true, true, true, true,
                                      ROBOT_TRIPOD_NORMAL, 0.0f,
                                      contact, contact);                   // 기본 경로 검사를 통과해 첫 위상을 시작한다.
        if (handle->initialized)
        {
            return handle->run_enable;                                     // 첫 위상의 실제 실행 여부를 확인한다.
        }
    }
    return false;
}

/* 첫 Swing을 착지시킨 뒤 반대 그룹의 추가 걸음을 차단한다. */
static bool RlStopTest_CheckAirborneStop(void)
{
    GaitManager_Handle_t handle;                                             // 진행 중인 첫 위상을 저장한다.
    RobotGaitPhase_t output;                                                 // 정지 중 다리 상태를 저장한다.
    bool contact[ROBOT_LEG_COUNT];                                           // 현재 발 접촉을 저장한다.
    uint32_t cycle;                                                          // 착지 대기 주기를 저장한다.
    uint32_t leg;                                                            // 이륙·착지할 다리 번호를 저장한다.
    const uint32_t timeout_cycles =
        (uint32_t)(ROBOT_GAIT_PHASE_TIME_S / ROBOT_CONTROL_PERIOD_S) + 10U;  // 현재 위상을 마칠 시간을 확보한다.

    if (!RlStopTest_Start(&handle, contact))
    {
        return false;
    }
    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        contact[leg] = false;                                                     // 첫 Swing 그룹의 비접촉을 준비한다.
    }
    (void)GaitManager_StepContacts(&handle, true, true, true, true,
                                  ROBOT_TRIPOD_NORMAL, 0.0f,
                                  contact, contact);                              // 현재 발의 실제 이륙을 기록한다.
    GaitManager_SetStopAfterLanding(&handle, true);                               // 현재 발 착지 후 강제 정지를 요청한다.
    if (!handle.initialized || !handle.run_enable || (handle.phase_index != 0U))
    {
        return false;
    }

    memset(contact, 1, sizeof(contact));                                    // 현재 Swing 발의 정상 접촉을 준비한다.
    for (cycle = 0U; cycle < timeout_cycles; ++cycle)
    {
        GaitManager_SetStopAfterLanding(&handle, true);                     // 매 제어 주기의 정지 요청을 반복한다.
        output = GaitManager_StepContacts(&handle, true, true, true, true,
                                          ROBOT_TRIPOD_NORMAL, 0.0f,
                                          contact, contact);                // 새 이동 입력에도 현재 위상만 정리한다.
        if (output.next_phase_preview || (handle.phase_index > 1U))
        {
            return false;
        }
        if (!output.enabled_internal)
        {
            for (leg = 0U; leg < ROBOT_LEG_COUNT; ++leg)
            {
                if (output.state[leg] != ROBOT_LEG_STANCE)
                {
                    return false;
                }
            }
            return !handle.initialized && handle.resume_phase &&
                   (handle.phase_index == 1U);                              // 추가 걸음 없이 다음 그룹 재개 위치를 보존한다.
        }
    }
    return false;
}

/* 시작 전 경로 검사와 첫 입력 대기를 정지 요청으로 취소한다. */
static bool RlStopTest_CheckPreviewCancel(void)
{
    GaitManager_Handle_t handle;                 // 아직 이륙하지 않은 보행 상태를 저장한다.
    RobotGaitPhase_t output;                     // 경로 검사와 이륙 출력을 저장한다.
    bool contact[ROBOT_LEG_COUNT];               // 정상 접촉 상태를 저장한다.
    uint32_t cycle;                              // 시작 대기 주기를 저장한다.

    GaitManager_Init(&handle);                                                          // 초기 정지 상태를 준비한다.
    memset(contact, 1, sizeof(contact));                                                // 모든 발 접촉을 준비한다.
    for (cycle = 0U; cycle < ROBOT_GAIT_START_DELAY_CYCLES + 2U; ++cycle)
    {
        output = GaitManager_StepContacts(&handle, true, true, false, false,
                                          ROBOT_TRIPOD_NORMAL, 0.0f,
                                          contact, contact);                            // 경로 검사를 요청하되 이륙은 대기한다.
        if (output.next_phase_preview)
        {
            break;
        }
    }
    if (!output.next_phase_preview || handle.initialized || !handle.next_phase_locked)
    {
        return false;
    }

    GaitManager_SetStopAfterLanding(&handle, true);                                          // 시작 전 계획을 취소한다.
    output = GaitManager_StepContacts(&handle, true, true, true, true,
                                      ROBOT_TRIPOD_NORMAL, 0.0f,
                                      contact, contact);                                     // 취소된 검사가 늦게 통과하는 상황을 준비한다.
    if (output.enabled_internal || output.next_phase_preview || handle.initialized ||
        handle.next_phase_locked || handle.next_phase_enable || (handle.phase_index != 0U))
    {
        return false;
    }

    GaitManager_SetStopAfterLanding(&handle, false);                              // 명시적으로 새 시작을 허가한다.
    output = GaitManager_StepContacts(&handle, true, true, false, false,
                                      ROBOT_TRIPOD_NORMAL, 0.0f,
                                      contact, contact);                          // 새로운 시작 대기를 준비한다.
    if (!output.enabled_internal || !output.waiting_start || handle.initialized)
    {
        return false;
    }

    GaitManager_SetStopAfterLanding(&handle, true);                     // 입력 대기 중에도 즉시 취소한다.
    output = GaitManager_StepContacts(&handle, true, true, true, true,
                                      ROBOT_TRIPOD_NORMAL, 0.0f,
                                      contact, contact);                // 취소 후 남은 시작 명령을 차단한다.
    return !output.enabled_internal && !handle.initialized &&
           !handle.next_phase_enable && (handle.phase_index == 0U);     // 이륙 없이 초기 위상을 유지한다.
}

/* 현재 Swing이 착지한 뒤 공개된 다음 위상 검사도 취소한다. */
static bool RlStopTest_CheckBoundaryCancel(void)
{
    GaitManager_Handle_t handle;                                             // 착지 경계의 보행 상태를 저장한다.
    RobotGaitPhase_t output;                                                 // 다음 위상 검사 출력을 저장한다.
    bool contact[ROBOT_LEG_COUNT];                                           // 첫 위상 접촉을 저장한다.
    uint32_t cycle;                                                          // 위상 진행 횟수를 저장한다.
    const uint32_t timeout_cycles =
        (uint32_t)(ROBOT_GAIT_PHASE_TIME_S / ROBOT_CONTROL_PERIOD_S) + 10U;  // 첫 위상의 착지 경계를 기다린다.

    if (!RlStopTest_Start(&handle, contact))
    {
        return false;
    }
    memset(contact, 0, sizeof(contact));                                                 // 단순 호출에서 Swing 비접촉을 준비한다.
    (void)GaitManager_Step(&handle, true, true, false, false,
                           ROBOT_TRIPOD_NORMAL, 0.0f, contact);                          // 지지발은 유지하며 첫 이륙을 기록한다.
    memset(contact, 1, sizeof(contact));                                                 // 첫 그룹의 착지를 준비한다.
    for (cycle = 0U; cycle < timeout_cycles; ++cycle)
    {
        output = GaitManager_StepContacts(&handle, true, true, false, false,
                                          ROBOT_TRIPOD_NORMAL, 0.0f,
                                          contact, contact);                             // 반대 그룹의 경로 검사까지 진행한다.
        if (output.next_phase_preview)
        {
            break;
        }
    }
    if (!output.next_phase_preview || !handle.initialized || !handle.next_phase_enable)
    {
        return false;
    }

    GaitManager_SetStopAfterLanding(&handle, true);                     // 이미 공개된 다음 위상 검사를 취소한다.
    output = GaitManager_StepContacts(&handle, true, true, true, true,
                                      ROBOT_TRIPOD_NORMAL, 0.0f,
                                      contact, contact);                // 취소된 검사 결과가 통과해도 새 이륙을 차단한다.
    return !output.enabled_internal && !output.next_phase_preview &&
           (handle.phase_index == 1U);                                  // 착지한 첫 그룹까지만 완료한다.
}

/* 지지발 복구의 발 역할과 위상 진행률을 정지 중에도 유지한다. */
static bool RlStopTest_CheckSupportRecovery(void)
{
    GaitManager_Handle_t handle;                   // 지지발 복구 상태를 저장한다.
    RobotGaitPhase_t output;                       // 정지 중 발 역할을 저장한다.
    bool contact[ROBOT_LEG_COUNT];                 // 지지발 손실 접촉을 저장한다.
    uint32_t phase_cycle;                          // 복구 전 정지한 위상 주기를 저장한다.

    if (!RlStopTest_Start(&handle, contact))
    {
        return false;
    }
    contact[0] = false;                                                              // 공중에 있는 첫 Swing 발을 준비한다.
    contact[1] = false;                                                              // 기존 지지발의 접촉 손실을 준비한다.
    (void)GaitManager_StepContacts(&handle, true, true, true, true,
                                  ROBOT_TRIPOD_NORMAL, 0.0f,
                                  contact, contact);                                 // 기존 지지발 복구를 시작한다.
    phase_cycle = handle.phase_cycle_count;                                          // 복구에서 정지한 위상 시간을 저장한다.
    GaitManager_SetStopAfterLanding(&handle, true);                                  // 복구 중 현재 위상 종료를 요청한다.
    output = GaitManager_StepContacts(&handle, true, true, true, true,
                                      ROBOT_TRIPOD_NORMAL, 0.0f,
                                      contact, contact);                             // 지지발 복구를 계속 수행한다.
    return output.enabled_internal && output.support_recovery_active &&
           (output.state[1] == ROBOT_LEG_LATE_LANDING) &&
           (output.state[0] == ROBOT_LEG_HOLD) &&
           (handle.phase_cycle_count == phase_cycle) && (handle.phase_index == 0U);  // 발 역할과 복구 중 위상 정지를 유지한다.
}

/* 강제 정지가 Late Landing 한계 정지를 자동 해제하지 않는지 검사한다. */
static bool RlStopTest_CheckLateLandingHold(void)
{
    GaitManager_Handle_t handle;                                                // 지면 탐색 중인 보행 상태를 저장한다.
    RobotGaitPhase_t output;                                                    // 탐색 한계 이후 정지 출력을 저장한다.
    bool contact[ROBOT_LEG_COUNT];                                              // 착지하지 못한 접촉 상태를 저장한다.
    uint32_t cycle;                                                             // 탐색 한계 대기 주기를 저장한다.
    uint32_t leg;                                                               // 미접촉 Swing 다리 번호를 저장한다.
    const uint32_t timeout_cycles =
        (uint32_t)((ROBOT_GAIT_PHASE_TIME_S + ROBOT_LATE_LANDING_MAX_TIME_S) /
                   ROBOT_CONTROL_PERIOD_S) + 10U;                               // 정상 Swing과 최대 탐색 시간을 확보한다.

    if (!RlStopTest_Start(&handle, contact))
    {
        return false;
    }
    for (leg = 0U; leg < ROBOT_LEG_COUNT; leg += 2U)
    {
        contact[leg] = false;                                               // 첫 Swing 그룹의 착지 손실을 준비한다.
    }
    GaitManager_SetStopAfterLanding(&handle, true);                         // 착지 확인이 필요한 정지를 요청한다.
    for (cycle = 0U; cycle < timeout_cycles; ++cycle)
    {
        output = GaitManager_StepContacts(&handle, true, true, true, true,
                                          ROBOT_TRIPOD_NORMAL, 0.0f,
                                          contact, contact);                // 현재 발의 지면 탐색을 계속한다.
        if (output.late_landing_hold)
        {
            break;
        }
    }
    if (!output.late_landing_hold)
    {
        return false;
    }

    for (cycle = 0U; cycle < 5U; ++cycle)
    {
        GaitManager_SetStopAfterLanding(&handle, true);                                     // 정지 상태를 반복해서 유지한다.
        output = GaitManager_StepContacts(&handle, true, true, true, true,
                                          ROBOT_TRIPOD_NORMAL, 0.0f,
                                          contact, contact);                                // 강제로 가려진 입력이 탐색 정지를 해제하지 않게 한다.
        if (!output.late_landing_hold || !output.waiting_start || output.enabled_internal)
        {
            return false;
        }
    }

    GaitManager_SetStopAfterLanding(&handle, false);                    // 새 세션의 보행 마스크를 해제한다.
    output = GaitManager_StepContacts(&handle, true, true, true, true,
                                      ROBOT_TRIPOD_NORMAL, 0.0f,
                                      contact, contact);                // 이동 입력만으로 탐색 한계가 풀리지 않는지 확인한다.
    if (!output.late_landing_hold || output.enabled_internal)
    {
        return false;
    }

    output = GaitManager_StepContacts(&handle, true, false, true, true,
                                      ROBOT_TRIPOD_NORMAL, 0.0f,
                                      contact, contact);                 // 기존의 명시적인 입력 해제를 적용한다.
    return !handle.late_landing_hold && !output.late_landing_hold;       // 입력 해제 후에만 탐색 정지 재무장을 허가한다.
}

/* 강화학습 종료의 추가 이륙 차단과 기존 착지 복구 유지를 검사한다. */
bool RlStopTest_Run(void)
{
    return RlStopTest_CheckAirborneStop() &&
           RlStopTest_CheckPreviewCancel() &&
           RlStopTest_CheckBoundaryCancel() &&
           RlStopTest_CheckSupportRecovery() &&
           RlStopTest_CheckLateLandingHold();    // 정지 전이의 각 제어 경계를 함께 검증한다.
}
