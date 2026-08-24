################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/module/leg6_test.c \
../Core/Src/module/lora.c \
../Core/Src/module/relay.c 

OBJS += \
./Core/Src/module/leg6_test.o \
./Core/Src/module/lora.o \
./Core/Src/module/relay.o 

C_DEPS += \
./Core/Src/module/leg6_test.d \
./Core/Src/module/lora.d \
./Core/Src/module/relay.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/module/%.o Core/Src/module/%.su Core/Src/module/%.cyclo: ../Core/Src/module/%.c Core/Src/module/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-module

clean-Core-2f-Src-2f-module:
	-$(RM) ./Core/Src/module/leg6_test.cyclo ./Core/Src/module/leg6_test.d ./Core/Src/module/leg6_test.o ./Core/Src/module/leg6_test.su ./Core/Src/module/lora.cyclo ./Core/Src/module/lora.d ./Core/Src/module/lora.o ./Core/Src/module/lora.su ./Core/Src/module/relay.cyclo ./Core/Src/module/relay.d ./Core/Src/module/relay.o ./Core/Src/module/relay.su

.PHONY: clean-Core-2f-Src-2f-module

