#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <stdbool.h>
#include <stdint.h>

typedef enum
{
    ALGORITHM_TEST_CALIBRATION_TABLE = 0,  // 중앙 실측값 완료를 검사한다.
    ALGORITHM_TEST_CALIBRATION_CONVERSION, // 실측값 변환 알고리즘을 검사한다.
    ALGORITHM_TEST_USER_COMMAND,           // CRSF 변환과 Failsafe를 검사한다.
    ALGORITHM_TEST_KINEMATICS,             // 좌표변환과 FK·IK를 검사한다.
    ALGORITHM_TEST_CONTROLLER,             // 상위 제어 계산을 검사한다.
    ALGORITHM_TEST_WORKSPACE,              // 동적 작업공간 제한을 검사한다.
    ALGORITHM_TEST_GAIT,                   // Tripod 위상과 접촉 적응을 검사한다.
    ALGORITHM_TEST_MODE_TRANSITION,        // 모드 전환 연속성을 검사한다.
    ALGORITHM_TEST_SAFETY,                 // Fault Latch를 검사한다.
    ALGORITHM_TEST_COMMUNICATION,          // 관제 패킷 생성을 검사한다.
    ALGORITHM_TEST_COUNT
} AlgorithmTestStep_t;

typedef struct
{
    AlgorithmTestStep_t next_step;             // 다음에 실행할 알고리즘 단계를 저장한다.
    AlgorithmTestStep_t last_step;             // 최근 실행 단계를 저장한다.
    bool passed[ALGORITHM_TEST_COUNT];          // 단계별 성공 여부를 저장한다.
    bool stopped;                              // 첫 실패 후 정지 여부를 저장한다.
} TestRunner_Handle_t;

void TestRunner_Init(TestRunner_Handle_t *handle);  // 첫 알고리즘 단계로 초기화한다.
bool TestRunner_RunStep(TestRunner_Handle_t *handle,
                        AlgorithmTestStep_t step);  // 선택한 알고리즘 단계 하나를 실행한다.
bool TestRunner_RunNext(TestRunner_Handle_t *handle);  // 정해진 다음 알고리즘 단계를 실행한다.
bool TestRunner_IsComplete(const TestRunner_Handle_t *handle);  // 모든 알고리즘 통과 여부를 반환한다.
const char *TestRunner_GetStepName(AlgorithmTestStep_t step);  // 단계 이름을 반환한다.

#endif
