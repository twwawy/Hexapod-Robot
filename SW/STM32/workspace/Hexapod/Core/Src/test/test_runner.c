#include "test/test_runner.h"

#include "common/robot_calibration.h"
#include "test/calibration_algorithm_test.h"
#include "test/communication_test.h"
#include "test/controller_test.h"
#include "test/gait_test.h"
#include "test/kinematics_test.h"
#include "test/mode_transition_test.h"
#include "test/rl_controller_test.h"
#include "test/rl_stop_test.h"
#include "test/safety_test.h"
#include "test/user_command_test.h"
#include "test/workspace_test.h"

#include <stddef.h>
#include <string.h>

/* 선택한 알고리즘 검증 함수를 실행한다. */
static bool TestRunner_Execute(AlgorithmTestStep_t step)
{
    switch (step)
    {
        case ALGORITHM_TEST_CALIBRATION_TABLE:
            return RobotCalibration_IsComplete(&g_robot_calibration);  // 모든 실측값 완료를 검사한다.

        case ALGORITHM_TEST_CALIBRATION_CONVERSION:
            return CalibrationAlgorithmTest_Run();  // 중앙 실측값의 변환을 검사한다.

        case ALGORITHM_TEST_USER_COMMAND:
            return UserCommandTest_Run();  // 중앙 CRSF 값으로 명령 변환을 검사한다.

        case ALGORITHM_TEST_KINEMATICS:
            return KinematicsTest_Run(0.0001f, 0.001f);  // FK·IK 왕복 오차를 검사한다.

        case ALGORITHM_TEST_CONTROLLER:
            return ControllerTest_Run();  // 상위 제어 계산을 검사한다.

        case ALGORITHM_TEST_WORKSPACE:
            return WorkspaceTest_Run();  // 작업공간 제한을 검사한다.

        case ALGORITHM_TEST_GAIT:
            return GaitTest_Run();  // Tripod 위상과 접촉 적응을 검사한다.

        case ALGORITHM_TEST_MODE_TRANSITION:
            return ModeTransitionTest_Run();  // 모드 전환 연속성을 검사한다.

        case ALGORITHM_TEST_SAFETY:
            return SafetyTest_Run();  // 복구 없는 Fault Latch를 검사한다.

        case ALGORITHM_TEST_COMMUNICATION:
            return CommunicationTest_Run();  // 관제 패킷 알고리즘을 검사한다.

        case ALGORITHM_TEST_RL_INPUT:
            return RlControllerTest_Run();  // 관측·세션·순번·수치 범위를 검사한다.

        case ALGORITHM_TEST_RL_STOP:
            return RlStopTest_Run();  // 이륙 취소와 현재 착지 완료를 검사한다.

        default:
            return false;
    }
}

/* 알고리즘 검증 순서를 첫 단계로 초기화한다. */
void TestRunner_Init(TestRunner_Handle_t *handle)
{
    if (handle != NULL)
    {
        memset(handle, 0, sizeof(*handle));                   // 이전 시험 결과를 제거한다.
        handle->next_step = ALGORITHM_TEST_CALIBRATION_TABLE; // 중앙 테이블 검사부터 시작한다.
        handle->last_step = ALGORITHM_TEST_COUNT;             // 아직 실행한 단계가 없음을 표시한다.
    }
}

/* 선택한 알고리즘 단계 하나를 실행하고 결과를 기록한다. */
bool TestRunner_RunStep(TestRunner_Handle_t *handle,
                        AlgorithmTestStep_t step)
{
    bool passed;  // 선택 단계 실행 결과를 저장한다.

    if ((handle == NULL) || (step >= ALGORITHM_TEST_COUNT) || handle->stopped)
    {
        return false;
    }

    passed = TestRunner_Execute(step);  // 실제 알고리즘 검증을 실행한다.
    handle->last_step = step;           // 최근 실행 단계를 기록한다.
    handle->passed[step] = passed;      // 단계 결과를 기록한다.

    if (!passed)
    {
        handle->stopped = true;  // 실패 원인을 확인할 때까지 이후 단계를 막는다.
        return false;
    }

    if (step == handle->next_step)
    {
        handle->next_step = (AlgorithmTestStep_t)((uint32_t)step + 1U);  // 순서대로 통과하면 다음 단계로 이동한다.
    }
    return true;
}

/* 중앙 테이블 검사부터 정해진 다음 알고리즘 단계를 실행한다. */
bool TestRunner_RunNext(TestRunner_Handle_t *handle)
{
    if ((handle == NULL) || (handle->next_step >= ALGORITHM_TEST_COUNT))
    {
        return false;
    }

    return TestRunner_RunStep(handle, handle->next_step);  // 현재 순서의 테스트만 실행한다.
}

/* 모든 알고리즘 단계가 순서대로 통과했는지 반환한다. */
bool TestRunner_IsComplete(const TestRunner_Handle_t *handle)
{
    return (handle != NULL) && !handle->stopped &&
           (handle->next_step >= ALGORITHM_TEST_COUNT);  // 마지막 단계 다음까지 이동했는지 확인한다.
}

/* 알고리즘 테스트 단계의 짧은 이름을 반환한다. */
const char *TestRunner_GetStepName(AlgorithmTestStep_t step)
{
    static const char *const names[ALGORITHM_TEST_COUNT] =
    {
        "CALIBRATION_TABLE", // 중앙 실측값 검사를 표시한다.
        "CALIBRATION_CONVERSION", // 실측값 변환 검사를 표시한다.
        "USER_COMMAND",      // 사용자 명령 검사를 표시한다.
        "KINEMATICS",        // FK·IK 검사를 표시한다.
        "CONTROLLER",        // 상위 제어 검사를 표시한다.
        "WORKSPACE",         // 작업공간 검사를 표시한다.
        "GAIT",              // 보행 위상 검사를 표시한다.
        "MODE_TRANSITION",   // 모드 전환 검사를 표시한다.
        "SAFETY",            // Safety 검사를 표시한다.
        "COMMUNICATION",     // 관제 패킷 검사를 표시한다.
        "RL_INPUT",          // 강화학습 입력 검사를 표시한다.
        "RL_STOP"            // 강화학습 정지 검사를 표시한다.
    };

    return (step < ALGORITHM_TEST_COUNT) ? names[step] : "INVALID";  // 범위 밖 단계를 구분한다.
}
