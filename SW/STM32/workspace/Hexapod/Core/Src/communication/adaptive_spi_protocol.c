#include "communication/adaptive_spi_protocol.h"
#include <string.h>
#include <math.h>
static uint16_t u16(const uint8_t *p) {return (uint16_t)p[0]|((uint16_t)p[1]<<8);}
static int16_t i16(const uint8_t *p) {return (int16_t)u16(p);}
static void put16(uint8_t *p,uint16_t x) {p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);}
static void put32(uint8_t *p,uint32_t x) {put16(p,(uint16_t)x);put16(p+2,(uint16_t)(x>>16));}
static void fixed(uint8_t *p,float x,float scale) {
    /* Observation saturation is explicit. Commands are rejected, never saturated. */
    put16(p,(uint16_t)(int16_t)lrintf(fminf(fmaxf(x*scale,-32768.0f),32767.0f)));
}
uint16_t AdaptiveSpi_Crc(const uint8_t *p,unsigned n) {
    uint16_t crc=0xffff;
    for(unsigned i=0;i<n;++i) {crc^=(uint16_t)p[i]<<8;
        for(unsigned j=0;j<8;++j) crc=(crc&0x8000)?(uint16_t)((crc<<1)^0x1021):(uint16_t)(crc<<1);}
    return crc;
}
bool AdaptiveSpi_DecodeExecution(const uint8_t f[128],RobotAdaptiveExecutionPlan_t *p) {
    if(!f||!p||f[0]!=0xa5||f[1]!=ADAPTIVE_SPI_COMMAND||f[5]!=4||
        (f[4]&~1U)||u16(f+126)!=AdaptiveSpi_Crc(f,126)) return false;
    for(unsigned i=104;i<126;++i) if(f[i]) return false;
    memset(p,0,sizeof(*p));
    p->sequence=u16(f+2);p->execute=(f[4]&1)!=0;
    p->session_id=(uint32_t)u16(f+6)|((uint32_t)u16(f+8)<<16);
    p->observation_sequence=u16(f+10);p->plan_id=u16(f+12);
    p->swing_mask=f[14];p->requested_gait_pattern=(RobotGaitPattern_t)f[15];
    p->phase_duration_s=u16(f+16)*.001f;
    p->posture_reference_rad.roll=i16(f+18)*.0001f;
    p->posture_reference_rad.pitch=i16(f+20)*.0001f;
    p->body_height_offset_m=i16(f+22)*.001f;
    p->applied_twist.vx=i16(f+24)*.0001f;p->applied_twist.vy=i16(f+26)*.0001f;
    p->applied_twist.vz=i16(f+28)*.0001f;p->applied_twist.wz=i16(f+30)*.0001f;
    for(unsigned i=0;i<6;++i) {const uint8_t *q=f+32+12*i;
        p->leg[i].landing=(RobotVec3_t){i16(q)*.001f,i16(q+2)*.001f,i16(q+4)*.001f};
        p->leg[i].clearance_m=u16(q+6)*.001f;
        p->leg[i].apex_phase=u16(q+8)*.0001f;p->leg[i].transfer_phase=u16(q+10)*.0001f;
    }
    return true; /* Semantic validation belongs to RlController in the main loop. */
}
void AdaptiveSpi_EncodeObservation(uint8_t f[128],const AdaptiveSpi_Observation_t *o,bool detail) {
    memset(f,0,128);f[0]=0xa5;f[1]=detail?ADAPTIVE_SPI_DETAIL:ADAPTIVE_SPI_OBSERVATION;
    put16(f+2,o->sequence);f[4]=o->flags;f[5]=4;put32(f+6,o->session_id);
    put16(f+10,o->sequence);put16(f+12,o->plan_id);f[14]=o->swing_mask;f[15]=o->gait;
    if(detail) {
        for(unsigned i=0;i<6;++i) {unsigned k=16+6*i;
            fixed(f+k,o->start[i].x,1000);fixed(f+k+2,o->start[i].y,1000);fixed(f+k+4,o->start[i].z,1000);
            k=52+6*i;fixed(f+k,o->feet[i].x,1000);fixed(f+k+2,o->feet[i].y,1000);fixed(f+k+4,o->feet[i].z,1000);
            f[88+i]=o->leg_state[i];}
        fixed(f+94,o->posture_command.roll,10000);fixed(f+96,o->posture_command.pitch,10000);
        fixed(f+98,o->posture_command.yaw,10000);put32(f+100,o->timestamp_ms);
    } else {
        fixed(f+16,o->elapsed_s,1000);fixed(f+18,o->imu.roll,10000);fixed(f+20,o->imu.pitch,10000);
        fixed(f+22,o->imu.yaw,10000);fixed(f+24,o->command.vx,10000);fixed(f+26,o->command.vy,10000);
        fixed(f+28,o->command.wz,10000);fixed(f+30,o->applied.vx,10000);fixed(f+32,o->applied.vy,10000);
        fixed(f+34,o->applied.vz,10000);fixed(f+36,o->applied.wz,10000);
        f[38]=o->contacts;f[39]=o->raw_contacts;f[40]=o->state;f[41]=o->flags;
        put16(f+42,o->ack_sequence);put16(f+44,o->ack_plan);f[46]=o->ack_mask;f[47]=o->next_phase;
        for(unsigned i=0;i<18;++i) fixed(f+48+2*i,o->joints[i],10000);
        for(unsigned i=0;i<6;++i) {unsigned k=84+6*i;
            fixed(f+k,o->nominal[i].x,1000);fixed(f+k+2,o->nominal[i].y,1000);fixed(f+k+4,o->nominal[i].z,1000);}
        fixed(f+120,o->duration_s,1000);fixed(f+122,o->height_m,1000);f[124]=o->result;f[125]=o->planned_gait;
    }
    put16(f+126,AdaptiveSpi_Crc(f,126));
}
