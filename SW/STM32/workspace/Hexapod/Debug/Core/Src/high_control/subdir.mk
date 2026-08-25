################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/high_control/body_position_estimator.c \
../Core/Src/high_control/body_posture_controller.c \
../Core/Src/high_control/contact_adaptation.c \
../Core/Src/high_control/control_priority.c \
../Core/Src/high_control/drone_controller.c \
../Core/Src/high_control/foot_trajectory.c \
../Core/Src/high_control/gait_manager.c \
../Core/Src/high_control/gait_pose_controller.c \
../Core/Src/high_control/leg_kinematics.c \
../Core/Src/high_control/safety.c \
../Core/Src/high_control/stance_trajectory.c \
../Core/Src/high_control/stand_landing.c \
../Core/Src/high_control/swing_trajectory.c \
../Core/Src/high_control/workspace_limiter.c 

OBJS += \
./Core/Src/high_control/body_position_estimator.o \
./Core/Src/high_control/body_posture_controller.o \
./Core/Src/high_control/contact_adaptation.o \
./Core/Src/high_control/control_priority.o \
./Core/Src/high_control/drone_controller.o \
./Core/Src/high_control/foot_trajectory.o \
./Core/Src/high_control/gait_manager.o \
./Core/Src/high_control/gait_pose_controller.o \
./Core/Src/high_control/leg_kinematics.o \
./Core/Src/high_control/safety.o \
./Core/Src/high_control/stance_trajectory.o \
./Core/Src/high_control/stand_landing.o \
./Core/Src/high_control/swing_trajectory.o \
./Core/Src/high_control/workspace_limiter.o 

C_DEPS += \
./Core/Src/high_control/body_position_estimator.d \
./Core/Src/high_control/body_posture_controller.d \
./Core/Src/high_control/contact_adaptation.d \
./Core/Src/high_control/control_priority.d \
./Core/Src/high_control/drone_controller.d \
./Core/Src/high_control/foot_trajectory.d \
./Core/Src/high_control/gait_manager.d \
./Core/Src/high_control/gait_pose_controller.d \
./Core/Src/high_control/leg_kinematics.d \
./Core/Src/high_control/safety.d \
./Core/Src/high_control/stance_trajectory.d \
./Core/Src/high_control/stand_landing.d \
./Core/Src/high_control/swing_trajectory.d \
./Core/Src/high_control/workspace_limiter.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/high_control/%.o Core/Src/high_control/%.su Core/Src/high_control/%.cyclo: ../Core/Src/high_control/%.c Core/Src/high_control/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-high_control

clean-Core-2f-Src-2f-high_control:
	-$(RM) ./Core/Src/high_control/body_position_estimator.cyclo ./Core/Src/high_control/body_position_estimator.d ./Core/Src/high_control/body_position_estimator.o ./Core/Src/high_control/body_position_estimator.su ./Core/Src/high_control/body_posture_controller.cyclo ./Core/Src/high_control/body_posture_controller.d ./Core/Src/high_control/body_posture_controller.o ./Core/Src/high_control/body_posture_controller.su ./Core/Src/high_control/contact_adaptation.cyclo ./Core/Src/high_control/contact_adaptation.d ./Core/Src/high_control/contact_adaptation.o ./Core/Src/high_control/contact_adaptation.su ./Core/Src/high_control/control_priority.cyclo ./Core/Src/high_control/control_priority.d ./Core/Src/high_control/control_priority.o ./Core/Src/high_control/control_priority.su ./Core/Src/high_control/drone_controller.cyclo ./Core/Src/high_control/drone_controller.d ./Core/Src/high_control/drone_controller.o ./Core/Src/high_control/drone_controller.su ./Core/Src/high_control/foot_trajectory.cyclo ./Core/Src/high_control/foot_trajectory.d ./Core/Src/high_control/foot_trajectory.o ./Core/Src/high_control/foot_trajectory.su ./Core/Src/high_control/gait_manager.cyclo ./Core/Src/high_control/gait_manager.d ./Core/Src/high_control/gait_manager.o ./Core/Src/high_control/gait_manager.su ./Core/Src/high_control/gait_pose_controller.cyclo ./Core/Src/high_control/gait_pose_controller.d ./Core/Src/high_control/gait_pose_controller.o ./Core/Src/high_control/gait_pose_controller.su ./Core/Src/high_control/leg_kinematics.cyclo ./Core/Src/high_control/leg_kinematics.d ./Core/Src/high_control/leg_kinematics.o ./Core/Src/high_control/leg_kinematics.su ./Core/Src/high_control/safety.cyclo ./Core/Src/high_control/safety.d ./Core/Src/high_control/safety.o ./Core/Src/high_control/safety.su ./Core/Src/high_control/stance_trajectory.cyclo ./Core/Src/high_control/stance_trajectory.d ./Core/Src/high_control/stance_trajectory.o ./Core/Src/high_control/stance_trajectory.su ./Core/Src/high_control/stand_landing.cyclo ./Core/Src/high_control/stand_landing.d ./Core/Src/high_control/stand_landing.o ./Core/Src/high_control/stand_landing.su ./Core/Src/high_control/swing_trajectory.cyclo ./Core/Src/high_control/swing_trajectory.d ./Core/Src/high_control/swing_trajectory.o ./Core/Src/high_control/swing_trajectory.su ./Core/Src/high_control/workspace_limiter.cyclo ./Core/Src/high_control/workspace_limiter.d ./Core/Src/high_control/workspace_limiter.o ./Core/Src/high_control/workspace_limiter.su

.PHONY: clean-Core-2f-Src-2f-high_control

