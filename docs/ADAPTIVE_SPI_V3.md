# Adaptive SPI v3 — 128-byte explicit wire contract

Policy contract is **24-D v4**; transport version is **SPI v3**. They are different versions.
All integers are little endian. Signed fields are two's complement. CRC-16/CCITT-FALSE:
polynomial 0x1021, initial 0xffff, no reflection, no xorout; CRC covers bytes 0–125.
SPI mode 0, MSB-first, DRDY/DMA authority remains STM32. Each transaction is 128 bytes.
This integration app initializes v3; existing v2 parsers remain for legacy tests but a v2
master cannot communicate with the default v3 app. No automatic frame-length negotiation.

## Command (0x32)

| Offset | Bytes | Meaning / scale |
|---|---:|---|
| 0 | 1 | 0xa5 |
| 1 | 1 | 0x32: version 3, command type 2 |
| 2 | 2 | execution sequence u16 |
| 4 | 1 | bit0 execute; other bits zero |
| 5 | 1 | action/execution semantic version 4 |
| 6 | 4 | nonzero session u32 |
| 10 | 2 | source observation sequence u16 |
| 12 | 2 | advertised plan ID u16 |
| 14 | 1 | advertised swing mask, RF RM RB LF LM LB |
| 15 | 1 | requested gait: Tripod 0, Wave 1 |
| 16 | 2 | phase duration u16 milliseconds |
| 18,20 | 2 each | absolute roll/pitch i16 × 0.0001 rad |
| 22 | 2 | body height offset i16 millimetres |
| 24,26,28,30 | 2 each | applied vx,vy,vz,wz i16 × 0.0001 SI; vz=0 |
| 32+12i | 6 | final leg XYZ, three i16 millimetres |
| 38+12i | 2 | clearance u16 millimetres |
| 40+12i,42+12i | 2 each | apex/transfer u16 × 0.0001 |
| 104–125 | 22 | zero reserved |
| 126 | 2 | CRC |

Final XYZ uses the **phase-entry, pre-posture controller frame**, x forward/y left/z up.
It is an IK endpoint, including the simulator's 32-mm foot-centre convention after terrain
height conversion. It is not the policy XY residual or odom XYZ. Hardware foot geometry
must be calibrated before deployment. No landing-Z policy channel exists.

execute=0 is a validated HOLD/gait negotiation message: it can keep the command lease alive
but never permits lift. It still references the advertised old plan/mask. STM32 requests a
new gait preview at an all-contact boundary, validates the new execution plan, then changes
the active pattern on launch. Jetson must wait for the new advertised pattern/mask/plan ID.
Duplicate or old sequence, wrong session/plan, stale observation and invalid values do not
refresh the command lease. All-zero receive bytes are a read-only NOP, not a heartbeat.

## Observation state (0x31)

Bytes 0–15 share the command header, except sequence/source both name the capture and byte15
is actual gait. Flags: bit0 valid plan, bit1 leg ACK valid, bit2 startup, bit3 gait enabled.

| Offset | Content |
|---|---|
| 16 | elapsed phase milliseconds i16 |
| 18–23 | IMU roll/pitch/yaw i16 × 0.0001 rad |
| 24–29 | processed command vx/vy/wz i16 × 0.0001 |
| 30–37 | applied vx/vy/vz/wz i16 × 0.0001 |
| 38,39 | confirmed/raw contact masks |
| 40,41 | RL state, flags |
| 42,44 | execution ACK sequence, plan ID u16 |
| 46,47 | ACK mask, phase index modulo 6 |
| 48–83 | 18 measured joint angles i16 × 0.0001 rad, controller order/signs |
| 84–119 | six nominal XYZ triples i16 millimetres |
| 120,122 | active duration ms, applied body height mm |
| 124,125 | submit result enum, **advertised plan gait** |
| 126–127 | CRC |

## Observation detail (0x35)

Same header/session/sequence/plan/mask/flags as its state page. Bytes16–51: six start XYZ;
52–87: six current pre-height foot-memory XYZ, all i16 millimetres. Bytes88–93: six main
leg-state enums. Bytes94–99: applied posture-command roll/pitch/yaw i16 × 0.0001 rad.
Bytes100–103: original STM32 capture timestamp milliseconds u32. Bytes104–125 reserved zero.
CRC remains126. Both pages come from one stored snapshot, not consecutive control ticks.

The receiver assembles only matching pages. Reading the second page does not refresh the
original timestamp. Observation lease remains60 ms; execution lease100 ms. Measure end-to-end
latency and map/odom clock synchronization before live operation. Two pages require two
transactions; this is not a claim of 200-Hz neural inference.

C encoding uses explicit offsets and scales; struct assignment is internal only. DMA buffers
hold encoded bytes. ISR completion handling only flags a finished transfer; parsing and app
submission run in the main loop. Observation values saturate to wire ranges; command
serialization validates inputs and angular quantization rounds toward zero.
