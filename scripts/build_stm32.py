#!/usr/bin/env python3
"""Out-of-tree ARM build; includes newly added Core sources. Never flashes."""
import argparse
from pathlib import Path
import shutil
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--compiler',default='arm-none-eabi-gcc')
p.add_argument('--output',type=Path,default=Path('/tmp/hexapod-stm32-v4'))
a=p.parse_args()
compiler=shutil.which(a.compiler)
if compiler is None:
    p.error('ARM GCC not found; pass --compiler /absolute/path/to/arm-none-eabi-gcc')
root=Path(__file__).resolve().parents[1]/'SW/STM32/workspace/Hexapod'
a.output.mkdir(parents=True,exist_ok=True)
flags=['-mcpu=cortex-m4','-mthumb','-mfpu=fpv4-sp-d16','-mfloat-abi=hard']
includes=[root/'Core/Inc',root/'Drivers/CMSIS/Include',root/'Drivers/CMSIS/Device/ST/STM32F4xx/Include',root/'Drivers/STM32F4xx_HAL_Driver/Inc']
sources=sorted((root/'Core/Src').rglob('*.c'))+sorted((root/'Drivers/STM32F4xx_HAL_Driver/Src').glob('*.c'))
sources=[s for s in sources if not s.name.endswith('_template.c')]
sources+=[root/'Core/Startup/startup_stm32f446retx.s']
objects=[]
for source in sources:
    output=a.output/('_'.join(source.relative_to(root).parts)+'.o')
    subprocess.run([compiler,*flags,'-DUSE_HAL_DRIVER','-DSTM32F446xx','-Os','-g3','-ffunction-sections','-fdata-sections',
                    *['-I'+str(x) for x in includes],'-c',str(source),'-o',str(output)],check=True)
    objects.append(str(output))
subprocess.run([compiler,*flags,*objects,'-T'+str(root/'STM32F446RETX_FLASH.ld'),'--specs=nano.specs','--specs=nosys.specs',
                '-Wl,--gc-sections','-Wl,-Map='+str(a.output/'Hexapod.map'),'-Wl,--start-group','-lc','-lm','-Wl,--end-group',
                '-o',str(a.output/'Hexapod.elf')],check=True)
print(a.output/'Hexapod.elf')
