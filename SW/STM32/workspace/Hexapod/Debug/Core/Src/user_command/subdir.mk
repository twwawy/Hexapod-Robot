################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/user_command/crsf_protocol.c \
../Core/Src/user_command/crsf_receiver.c \
../Core/Src/user_command/user_command.c 

OBJS += \
./Core/Src/user_command/crsf_protocol.o \
./Core/Src/user_command/crsf_receiver.o \
./Core/Src/user_command/user_command.o 

C_DEPS += \
./Core/Src/user_command/crsf_protocol.d \
./Core/Src/user_command/crsf_receiver.d \
./Core/Src/user_command/user_command.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/user_command/%.o Core/Src/user_command/%.su Core/Src/user_command/%.cyclo: ../Core/Src/user_command/%.c Core/Src/user_command/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-user_command

clean-Core-2f-Src-2f-user_command:
	-$(RM) ./Core/Src/user_command/crsf_protocol.cyclo ./Core/Src/user_command/crsf_protocol.d ./Core/Src/user_command/crsf_protocol.o ./Core/Src/user_command/crsf_protocol.su ./Core/Src/user_command/crsf_receiver.cyclo ./Core/Src/user_command/crsf_receiver.d ./Core/Src/user_command/crsf_receiver.o ./Core/Src/user_command/crsf_receiver.su ./Core/Src/user_command/user_command.cyclo ./Core/Src/user_command/user_command.d ./Core/Src/user_command/user_command.o ./Core/Src/user_command/user_command.su

.PHONY: clean-Core-2f-Src-2f-user_command

