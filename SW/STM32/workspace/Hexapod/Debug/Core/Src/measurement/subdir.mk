################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/measurement/adc_mapping_measurement.c \
../Core/Src/measurement/crsf_calibration_measurement.c \
../Core/Src/measurement/foot_pressure_measurement.c \
../Core/Src/measurement/joint_calibration_measurement.c \
../Core/Src/measurement/measurement_debug.c \
../Core/Src/measurement/measurement_runner.c \
../Core/Src/measurement/measurement_stage0.c \
../Core/Src/measurement/measurement_stage1.c \
../Core/Src/measurement/measurement_stage2.c \
../Core/Src/measurement/measurement_stage3.c \
../Core/Src/measurement/measurement_stage4.c \
../Core/Src/measurement/measurement_stage5.c \
../Core/Src/measurement/sensor_raw_measurement.c \
../Core/Src/measurement/servo_relay_measurement.c 

OBJS += \
./Core/Src/measurement/adc_mapping_measurement.o \
./Core/Src/measurement/crsf_calibration_measurement.o \
./Core/Src/measurement/foot_pressure_measurement.o \
./Core/Src/measurement/joint_calibration_measurement.o \
./Core/Src/measurement/measurement_debug.o \
./Core/Src/measurement/measurement_runner.o \
./Core/Src/measurement/measurement_stage0.o \
./Core/Src/measurement/measurement_stage1.o \
./Core/Src/measurement/measurement_stage2.o \
./Core/Src/measurement/measurement_stage3.o \
./Core/Src/measurement/measurement_stage4.o \
./Core/Src/measurement/measurement_stage5.o \
./Core/Src/measurement/sensor_raw_measurement.o \
./Core/Src/measurement/servo_relay_measurement.o 

C_DEPS += \
./Core/Src/measurement/adc_mapping_measurement.d \
./Core/Src/measurement/crsf_calibration_measurement.d \
./Core/Src/measurement/foot_pressure_measurement.d \
./Core/Src/measurement/joint_calibration_measurement.d \
./Core/Src/measurement/measurement_debug.d \
./Core/Src/measurement/measurement_runner.d \
./Core/Src/measurement/measurement_stage0.d \
./Core/Src/measurement/measurement_stage1.d \
./Core/Src/measurement/measurement_stage2.d \
./Core/Src/measurement/measurement_stage3.d \
./Core/Src/measurement/measurement_stage4.d \
./Core/Src/measurement/measurement_stage5.d \
./Core/Src/measurement/sensor_raw_measurement.d \
./Core/Src/measurement/servo_relay_measurement.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/measurement/%.o Core/Src/measurement/%.su Core/Src/measurement/%.cyclo: ../Core/Src/measurement/%.c Core/Src/measurement/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src-2f-measurement

clean-Core-2f-Src-2f-measurement:
	-$(RM) ./Core/Src/measurement/adc_mapping_measurement.cyclo ./Core/Src/measurement/adc_mapping_measurement.d ./Core/Src/measurement/adc_mapping_measurement.o ./Core/Src/measurement/adc_mapping_measurement.su ./Core/Src/measurement/crsf_calibration_measurement.cyclo ./Core/Src/measurement/crsf_calibration_measurement.d ./Core/Src/measurement/crsf_calibration_measurement.o ./Core/Src/measurement/crsf_calibration_measurement.su ./Core/Src/measurement/foot_pressure_measurement.cyclo ./Core/Src/measurement/foot_pressure_measurement.d ./Core/Src/measurement/foot_pressure_measurement.o ./Core/Src/measurement/foot_pressure_measurement.su ./Core/Src/measurement/joint_calibration_measurement.cyclo ./Core/Src/measurement/joint_calibration_measurement.d ./Core/Src/measurement/joint_calibration_measurement.o ./Core/Src/measurement/joint_calibration_measurement.su ./Core/Src/measurement/measurement_debug.cyclo ./Core/Src/measurement/measurement_debug.d ./Core/Src/measurement/measurement_debug.o ./Core/Src/measurement/measurement_debug.su ./Core/Src/measurement/measurement_runner.cyclo ./Core/Src/measurement/measurement_runner.d ./Core/Src/measurement/measurement_runner.o ./Core/Src/measurement/measurement_runner.su ./Core/Src/measurement/measurement_stage0.cyclo ./Core/Src/measurement/measurement_stage0.d ./Core/Src/measurement/measurement_stage0.o ./Core/Src/measurement/measurement_stage0.su ./Core/Src/measurement/measurement_stage1.cyclo ./Core/Src/measurement/measurement_stage1.d ./Core/Src/measurement/measurement_stage1.o ./Core/Src/measurement/measurement_stage1.su ./Core/Src/measurement/measurement_stage2.cyclo ./Core/Src/measurement/measurement_stage2.d ./Core/Src/measurement/measurement_stage2.o ./Core/Src/measurement/measurement_stage2.su ./Core/Src/measurement/measurement_stage3.cyclo ./Core/Src/measurement/measurement_stage3.d ./Core/Src/measurement/measurement_stage3.o ./Core/Src/measurement/measurement_stage3.su ./Core/Src/measurement/measurement_stage4.cyclo ./Core/Src/measurement/measurement_stage4.d ./Core/Src/measurement/measurement_stage4.o ./Core/Src/measurement/measurement_stage4.su ./Core/Src/measurement/measurement_stage5.cyclo ./Core/Src/measurement/measurement_stage5.d ./Core/Src/measurement/measurement_stage5.o ./Core/Src/measurement/measurement_stage5.su ./Core/Src/measurement/sensor_raw_measurement.cyclo ./Core/Src/measurement/sensor_raw_measurement.d ./Core/Src/measurement/sensor_raw_measurement.o ./Core/Src/measurement/sensor_raw_measurement.su ./Core/Src/measurement/servo_relay_measurement.cyclo ./Core/Src/measurement/servo_relay_measurement.d ./Core/Src/measurement/servo_relay_measurement.o ./Core/Src/measurement/servo_relay_measurement.su

.PHONY: clean-Core-2f-Src-2f-measurement

