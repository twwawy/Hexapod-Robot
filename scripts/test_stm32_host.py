#!/usr/bin/env python3
"""Run portable algorithm tests, without board/DMA/actuator execution."""
from pathlib import Path
import subprocess
import tempfile
root=Path(__file__).resolve().parents[1]
project=root/'SW/STM32/workspace/Hexapod'
core=project/'Core'
names=['calibration_algorithm','user_command','kinematics','controller','workspace','gait','mode_transition','safety','communication','rl_controller','rl_stop']
functions=['CalibrationAlgorithmTest_Run()','UserCommandTest_Run()','KinematicsTest_Run(.0001f,.001f)','ControllerTest_Run()','WorkspaceTest_Run()','GaitTest_Run()','ModeTransitionTest_Run()','SafetyTest_Run()','CommunicationTest_Run()','RlControllerTest_Run()','RlStopTest_Run()']
with tempfile.TemporaryDirectory(prefix='hexapod-native-') as temp:
    main=Path(temp)/'main.c'
    text='#include <stdio.h>\n#include <stdlib.h>\n#include "stm32f4xx_hal.h"\nHAL_StatusTypeDef HAL_SPI_TransmitReceive_DMA(SPI_HandleTypeDef *h,const uint8_t *t,uint8_t *r,uint16_t n){abort();}\nvoid HAL_GPIO_WritePin(GPIO_TypeDef *p,uint16_t n,GPIO_PinState s){abort();}\n'+''.join(f'#include "test/{n}_test.h"\n' for n in names)
    text+='int main(void) {int failures=0;\n'
    for name,fn in zip(names,functions):
        text+=f'{{int ok={fn}; printf("{name}: %s\\n",ok?"PASS":"FAIL"); failures+=!ok;}}\n'
    main.write_text(text+'return failures?1:0;}\n')
    sources=list((core/'Src/high_control').glob('*.c'))+[core/'Src/test'/f'{n}_test.c' for n in names]
    sources += [core/'Src'/n for n in ['sensor/joint_feedback.c','sensor/imu.c','sensor/mcp3008.c','sensor/foot_pressure.c','low_control/servo_pwm.c','common/robot_calibration.c','user_command/user_command.c','user_command/crsf_protocol.c','communication/jetson_spi.c','communication/adaptive_spi_protocol.c','communication/robot_telemetry.c','communication/manipulator_link.c']]
    includes=[core/'Inc', project/'Drivers/CMSIS/Include',project/'Drivers/CMSIS/Device/ST/STM32F4xx/Include',project/'Drivers/STM32F4xx_HAL_Driver/Inc']
    command=['gcc','-std=c11','-O1','-ffunction-sections','-fdata-sections','-DSTM32F446xx','-DUSE_HAL_DRIVER','-Wno-pointer-to-int-cast','-Wno-int-to-pointer-cast',*[f'-I{x}' for x in includes],str(main),*map(str,sources),'-Wl,--gc-sections','-lm','-o',str(Path(temp)/'tests')]
    subprocess.run(command,check=True)
    subprocess.run([str(Path(temp)/'tests')],check=True)
