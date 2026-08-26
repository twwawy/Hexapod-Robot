################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/app/control_timing_debug.c \
../Core/Src/app/hexapod_app.c \
../Core/Src/app/pressure_load_calibration.c 

OBJS += \
./Core/Src/app/control_timing_debug.o \
./Core/Src/app/hexapod_app.o \
./Core/Src/app/pressure_load_calibration.o 

C_DEPS += \
./Core/Src/app/control_timing_debug.d \
./Core/Src/app/hexapod_app.d \
./Core/Src/app/pressure_load_calibration.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/app/%.o Core/Src/app/%.su Core/Src/app/%.cyclo: ../Core/Src/app/%.c Core/Src/app/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-app

clean-Core-2f-Src-2f-app:
	-$(RM) ./Core/Src/app/control_timing_debug.cyclo ./Core/Src/app/control_timing_debug.d ./Core/Src/app/control_timing_debug.o ./Core/Src/app/control_timing_debug.su ./Core/Src/app/hexapod_app.cyclo ./Core/Src/app/hexapod_app.d ./Core/Src/app/hexapod_app.o ./Core/Src/app/hexapod_app.su ./Core/Src/app/pressure_load_calibration.cyclo ./Core/Src/app/pressure_load_calibration.d ./Core/Src/app/pressure_load_calibration.o ./Core/Src/app/pressure_load_calibration.su

.PHONY: clean-Core-2f-Src-2f-app

