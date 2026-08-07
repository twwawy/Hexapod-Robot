################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/sensor/gps.c \
../Core/Src/sensor/imu.c \
../Core/Src/sensor/nav_kalman.c 

OBJS += \
./Core/Src/sensor/gps.o \
./Core/Src/sensor/imu.o \
./Core/Src/sensor/nav_kalman.o 

C_DEPS += \
./Core/Src/sensor/gps.d \
./Core/Src/sensor/imu.d \
./Core/Src/sensor/nav_kalman.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/sensor/%.o Core/Src/sensor/%.su Core/Src/sensor/%.cyclo: ../Core/Src/sensor/%.c Core/Src/sensor/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-sensor

clean-Core-2f-Src-2f-sensor:
	-$(RM) ./Core/Src/sensor/gps.cyclo ./Core/Src/sensor/gps.d ./Core/Src/sensor/gps.o ./Core/Src/sensor/gps.su ./Core/Src/sensor/imu.cyclo ./Core/Src/sensor/imu.d ./Core/Src/sensor/imu.o ./Core/Src/sensor/imu.su ./Core/Src/sensor/nav_kalman.cyclo ./Core/Src/sensor/nav_kalman.d ./Core/Src/sensor/nav_kalman.o ./Core/Src/sensor/nav_kalman.su

.PHONY: clean-Core-2f-Src-2f-sensor

