################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/test/calibration_algorithm_test.c \
../Core/Src/test/communication_test.c \
../Core/Src/test/controller_test.c \
../Core/Src/test/gait_test.c \
../Core/Src/test/kinematics_test.c \
../Core/Src/test/mode_transition_test.c \
../Core/Src/test/rc_command_generator.c \
../Core/Src/test/safety_test.c \
../Core/Src/test/test_runner.c \
../Core/Src/test/user_command_test.c \
../Core/Src/test/workspace_test.c 

OBJS += \
./Core/Src/test/calibration_algorithm_test.o \
./Core/Src/test/communication_test.o \
./Core/Src/test/controller_test.o \
./Core/Src/test/gait_test.o \
./Core/Src/test/kinematics_test.o \
./Core/Src/test/mode_transition_test.o \
./Core/Src/test/rc_command_generator.o \
./Core/Src/test/safety_test.o \
./Core/Src/test/test_runner.o \
./Core/Src/test/user_command_test.o \
./Core/Src/test/workspace_test.o 

C_DEPS += \
./Core/Src/test/calibration_algorithm_test.d \
./Core/Src/test/communication_test.d \
./Core/Src/test/controller_test.d \
./Core/Src/test/gait_test.d \
./Core/Src/test/kinematics_test.d \
./Core/Src/test/mode_transition_test.d \
./Core/Src/test/rc_command_generator.d \
./Core/Src/test/safety_test.d \
./Core/Src/test/test_runner.d \
./Core/Src/test/user_command_test.d \
./Core/Src/test/workspace_test.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/test/%.o Core/Src/test/%.su Core/Src/test/%.cyclo: ../Core/Src/test/%.c Core/Src/test/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-test

clean-Core-2f-Src-2f-test:
	-$(RM) ./Core/Src/test/calibration_algorithm_test.cyclo ./Core/Src/test/calibration_algorithm_test.d ./Core/Src/test/calibration_algorithm_test.o ./Core/Src/test/calibration_algorithm_test.su ./Core/Src/test/communication_test.cyclo ./Core/Src/test/communication_test.d ./Core/Src/test/communication_test.o ./Core/Src/test/communication_test.su ./Core/Src/test/controller_test.cyclo ./Core/Src/test/controller_test.d ./Core/Src/test/controller_test.o ./Core/Src/test/controller_test.su ./Core/Src/test/gait_test.cyclo ./Core/Src/test/gait_test.d ./Core/Src/test/gait_test.o ./Core/Src/test/gait_test.su ./Core/Src/test/kinematics_test.cyclo ./Core/Src/test/kinematics_test.d ./Core/Src/test/kinematics_test.o ./Core/Src/test/kinematics_test.su ./Core/Src/test/mode_transition_test.cyclo ./Core/Src/test/mode_transition_test.d ./Core/Src/test/mode_transition_test.o ./Core/Src/test/mode_transition_test.su ./Core/Src/test/rc_command_generator.cyclo ./Core/Src/test/rc_command_generator.d ./Core/Src/test/rc_command_generator.o ./Core/Src/test/rc_command_generator.su ./Core/Src/test/safety_test.cyclo ./Core/Src/test/safety_test.d ./Core/Src/test/safety_test.o ./Core/Src/test/safety_test.su ./Core/Src/test/test_runner.cyclo ./Core/Src/test/test_runner.d ./Core/Src/test/test_runner.o ./Core/Src/test/test_runner.su ./Core/Src/test/user_command_test.cyclo ./Core/Src/test/user_command_test.d ./Core/Src/test/user_command_test.o ./Core/Src/test/user_command_test.su ./Core/Src/test/workspace_test.cyclo ./Core/Src/test/workspace_test.d ./Core/Src/test/workspace_test.o ./Core/Src/test/workspace_test.su

.PHONY: clean-Core-2f-Src-2f-test

