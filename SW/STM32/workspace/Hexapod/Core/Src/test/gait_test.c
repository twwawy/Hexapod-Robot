#include "test/gait_test.h"

#include "high_control/gait_manager.h"

#include <string.h>

/* 명시적 접촉 입력으로 Tripod 상태 전환과 짧은 Enable 변화를 검사한다. */
bool GaitTest_Run(void)
{
    GaitManager_Handle_t manager;                  // 시험용 Tripod 상태를 저장한다.
    RobotGaitPhase_t phase;                         // 한 주기 다리 상태를 저장한다.
    bool contact[ROBOT_LEG_COUNT];                  // 명시적 접촉 입력을 저장한다.
    uint32_t cycle;                                 // 진행할 제어 주기 번호를 저장한다.

    memset(contact, 0, sizeof(contact));  // Swing 비접촉 상태를 만든다.
    GaitManager_Init(&manager);           // Tripod 위상을 정지로 초기화한다.
    phase = GaitManager_Step(&manager, true, ROBOT_TRIPOD_NORMAL,
                             0.0f, contact);  // 정상 보행을 시작한다.
    if (!phase.enabled_internal)
    {
        return false;
    }

    for (cycle = 0U; cycle < 60U; ++cycle)
    {
        phase = GaitManager_Step(&manager, true, ROBOT_TRIPOD_NORMAL,
                                 0.0f, contact);  // Swing 50% 이후까지 진행한다.
    }
    contact[0] = true;  // 1번 Swing 다리의 Early Landing 입력을 넣는다.
    phase = GaitManager_Step(&manager, true, ROBOT_TRIPOD_NORMAL,
                             0.0f, contact);  // 접촉 적응을 실행한다.
    if (phase.state[0] == ROBOT_LEG_SWING)
    {
        return false;
    }

    phase = GaitManager_Step(&manager, false, ROBOT_TRIPOD_NORMAL,
                             0.0f, contact);  // 한 주기 Enable OFF를 넣는다.
    phase = GaitManager_Step(&manager, true, ROBOT_TRIPOD_NORMAL,
                             0.0f, contact);  // 즉시 Enable을 복구한다.
    return phase.enabled_internal;  // 짧은 OFF에서 위상이 유지되는지 확인한다.
}
