"""SPI v3 explicit 128-byte LE codec; no C/Python struct layout dependency."""
import binascii
import struct
from adaptive_execution_plan import AdaptiveExecutionPlan, LegPlan
SIZE = 128
CONTRACT = 4

def crc(data):
    return binascii.crc_hqx(data, 0xffff)

def encode_execution(plan):
    plan.validate()
    f = bytearray(SIZE)
    f[0:2] = b'\xa5\x32'
    struct.pack_into('<HBBIHHBB', f, 2, plan.sequence, int(plan.execute), CONTRACT,
                     plan.session_id, plan.observation_sequence, plan.plan_id, plan.swing_mask, plan.requested_gait)
    # Toward-zero angular encoding prevents a rounded hard-limit request exceeding the limit.
    struct.pack_into('<Hhhh4h', f, 16, round(plan.duration*1000), int(plan.roll*10000),
                     int(plan.pitch*10000), round(plan.body_height*1000), *(int(x*10000) for x in plan.twist))
    for i, leg in enumerate(plan.legs):
        struct.pack_into('<3h3H', f, 32+12*i, *(round(x*1000) for x in leg.landing),
                         round(leg.clearance*1000), round(leg.apex*10000), round(leg.transfer*10000))
    struct.pack_into('<H', f, 126, crc(f[:126]))
    return bytes(f)

def validate_frame(f, types):
    if len(f) != SIZE or f[0] != 0xa5 or f[1] not in types or f[5] != CONTRACT or struct.unpack_from('<H',f,126)[0] != crc(f[:126]):
        raise ValueError('SPI v3 length/version/contract/type/CRC mismatch')

def decode_observation(f):
    validate_frame(f, (0x31,0x35))
    sequence, flags, _, session, source, plan, mask, gait = struct.unpack_from('<HBBIHHBB', f, 2)
    if source != sequence:
        raise ValueError('observation sequence mismatch')
    out = dict(session_id=session, sequence=sequence, plan_id=plan, swing_mask=mask,
               gait=gait, flags=flags, page='detail' if f[1]==0x35 else 'state')
    def fixed(offset, count, scale):
        return tuple(v/scale for v in struct.unpack_from('<'+'h'*count,f,offset))
    if f[1] == 0x35:
        out.update(start=[fixed(16+6*i,3,1000) for i in range(6)],
                   feet=[fixed(52+6*i,3,1000) for i in range(6)], leg_state=list(f[88:94]),
                   posture_command=fixed(94,3,10000), timestamp_ms=struct.unpack_from('<I',f,100)[0])
    else:
        out.update(elapsed=fixed(16,1,1000)[0], imu=fixed(18,3,10000),
                   command=fixed(24,3,10000), applied_twist=fixed(30,4,10000),
                   contacts=f[38], raw_contacts=f[39], state=f[40],
                   ack_sequence=struct.unpack_from('<H',f,42)[0], ack_plan=struct.unpack_from('<H',f,44)[0],
                   ack_mask=f[46], next_phase=f[47], joints=fixed(48,18,10000),
                   nominal=[fixed(84+6*i,3,1000) for i in range(6)],
                   duration=fixed(120,1,1000)[0], body_height=fixed(122,1,1000)[0], result=f[124], planned_gait=f[125])
    return out

class ObservationReceiver:
    """Two bounded slots; never combine different session/sequence/plan pages."""
    def __init__(self):
        self.pages = {}
    def receive(self, frame):
        page = decode_observation(frame)
        key = (page['session_id'],page['sequence'],page['plan_id'])
        if self.pages and key != self.pages.get('key'):
            self.pages.clear()
        self.pages['key'] = key
        self.pages[page['page']] = page
        if 'state' in self.pages and 'detail' in self.pages:
            state, detail = self.pages['state'], self.pages['detail']
            if any(state[k] != detail[k] for k in ('flags','swing_mask','gait')):
                self.pages.clear()
                raise ValueError('torn observation pages')
            self.pages.clear()
            return dict(state, **{k:v for k,v in detail.items() if k not in state})
        return None
