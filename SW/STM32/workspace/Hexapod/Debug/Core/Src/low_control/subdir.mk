################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/low_control/relay.c \
../Core/Src/low_control/servo_pwm.c 

OBJS += \
./Core/Src/low_control/relay.o \
./Core/Src/low_control/servo_pwm.o 

C_DEPS += \
./Core/Src/low_control/relay.d \
./Core/Src/low_control/servo_pwm.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/low_control/%.o Core/Src/low_control/%.su Core/Src/low_control/%.cyclo: ../Core/Src/low_control/%.c Core/Src/low_control/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-low_control

clean-Core-2f-Src-2f-low_control:
	-$(RM) ./Core/Src/low_control/relay.cyclo ./Core/Src/low_control/relay.d ./Core/Src/low_control/relay.o ./Core/Src/low_control/relay.su ./Core/Src/low_control/servo_pwm.cyclo ./Core/Src/low_control/servo_pwm.d ./Core/Src/low_control/servo_pwm.o ./Core/Src/low_control/servo_pwm.su

.PHONY: clean-Core-2f-Src-2f-low_control

