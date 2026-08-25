################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/common/robot_calibration.c 

OBJS += \
./Core/Src/common/robot_calibration.o 

C_DEPS += \
./Core/Src/common/robot_calibration.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/common/%.o Core/Src/common/%.su Core/Src/common/%.cyclo: ../Core/Src/common/%.c Core/Src/common/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-common

clean-Core-2f-Src-2f-common:
	-$(RM) ./Core/Src/common/robot_calibration.cyclo ./Core/Src/common/robot_calibration.d ./Core/Src/common/robot_calibration.o ./Core/Src/common/robot_calibration.su

.PHONY: clean-Core-2f-Src-2f-common

