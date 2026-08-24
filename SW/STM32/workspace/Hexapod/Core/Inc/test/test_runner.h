#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <stdbool.h>
#include <stdint.h>

typedef bool (*TestRunner_InitFunction_t)(void *context);     // 테스트 초기화 함수 형식을 정의한다.
typedef bool (*TestRunner_ProcessFunction_t)(void *context);  // 테스트 반복 함수 형식을 정의한다.

typedef enum
{
    TEST_RUNNER_IDLE = 0,  // 선택 전 상태를 나타낸다.
    TEST_RUNNER_RUNNING,   // 실행 중 상태를 나타낸다.
    TEST_RUNNER_PASSED,    // 정상 완료 상태를 나타낸다.
    TEST_RUNNER_FAILED     // 실패 상태를 나타낸다.
} TestRunner_State_t;

typedef struct
{
    TestRunner_InitFunction_t initialize;  // 선택 테스트 초기화 함수를 저장한다.
    TestRunner_ProcessFunction_t process;  // 선택 테스트 반복 함수를 저장한다.
    void *context;                         // 테스트별 상태를 저장한다.
    TestRunner_State_t state;              // 현재 실행 상태를 저장한다.
    uint32_t process_count;                // 반복 실행 횟수를 저장한다.
} TestRunner_Handle_t;

bool TestRunner_Start(TestRunner_Handle_t *handle,
                      TestRunner_InitFunction_t initialize,
                      TestRunner_ProcessFunction_t process,
                      void *context);  // 선택한 테스트 하나를 시작한다.

TestRunner_State_t TestRunner_Process(TestRunner_Handle_t *handle);  // 선택 테스트를 한 번 진행한다.

void TestRunner_Fail(TestRunner_Handle_t *handle);  // 외부 검사 실패를 기록한다.

#endif
