#include <Servo.h>
#include <EEPROM.h>
#include <math.h>
#include <string.h>

/*
 * Arduino Uno manipulator controller
 *
 * Servo signal pins:
 *   J1~J6 = D5, D6, D7, D8, D9, D10
 * IMPORTANT:
 * - Servo power never comes from the Uno 5 V pin.
 * - HOME_US and SERVO_DIRECTION must be checked on the real arm.
 * - UART diagnostics are printed on D1/TX and the USB Serial Monitor.
 */

namespace Config {

constexpr uint8_t JOINT_COUNT = 6;
constexpr uint8_t SERVO_PIN[JOINT_COUNT] = {5, 6, 7, 8, 9, 10};

// Measured software-zero fallback values. A completed J1~J6 calibration
// stored by ManipulatorCalibration overrides all six values at boot.
constexpr int16_t DEFAULT_HOME_US[JOINT_COUNT] = {
  1290, 1600, 1480, 1510, 1610, 1540
};
int16_t HOME_US[JOINT_COUNT] = {
  1500, 1500, 1500, 1500, 1500, 1500
};

constexpr int16_t CALIBRATION_MIN_US = 1000;
constexpr int16_t CALIBRATION_MAX_US = 2000;
constexpr uint16_t CALIBRATION_MAGIC = 0x4D43;
constexpr uint8_t CALIBRATION_VERSION = 1;
constexpr int CALIBRATION_MAGIC_ADDRESS = 0;
constexpr int CALIBRATION_VERSION_ADDRESS = 2;
constexpr int CALIBRATION_DATA_ADDRESS = 4;

// Change only the corresponding value to -1 if that joint moves backwards.
constexpr int8_t SERVO_DIRECTION[JOINT_COUNT] = {
  +1, +1, -1, +1, +1, +1
};

// J1~J4: DS51150-270, J5~J6: SPT5435LV-180.
constexpr float US_PER_DEG[JOINT_COUNT] = {
  2000.0f / 270.0f,
  2000.0f / 270.0f,
  2000.0f / 270.0f,
  2000.0f / 270.0f,
  2000.0f / 180.0f,
  2000.0f / 180.0f
};

// Conservative software pulse limits. Widen only after one-joint testing.
constexpr int16_t MIN_US[JOINT_COUNT] = {600, 600, 600, 600, 800, 800};
constexpr int16_t MAX_US[JOINT_COUNT] = {2400, 2400, 2400, 2400, 2200, 2200};

// Approximate folded ARM-ready pose based on image 2.
// J1 base, J2 shoulder, J3 elbow, J4 wrist pitch, J5 wrist roll, J6 gripper.
// This is deliberately conservative; it is not a measured Fusion joint export.
constexpr float READY_DEG[JOINT_COUNT] = {
  0.0f, +35.0f, -100.0f, -25.0f, 0.0f, 0.0f
};

constexpr float JOINT_MIN_DEG[JOINT_COUNT] = {
  -90.0f, -10.0f, -125.0f, -120.0f, -90.0f, 0.0f
};
constexpr float JOINT_MAX_DEG[JOINT_COUNT] = {
  +90.0f, +80.0f,  -20.0f,  +60.0f, +90.0f, 45.0f
};

// Measure joint-axis distances in Fusion and replace these two values.
// They affect Cartesian speed/IK response, not the initial ready pose.
constexpr float LINK1_MM = 300.0f;
constexpr float LINK2_MM = 300.0f;

// Controller motion limits. IK converts these Cartesian velocities into
// coordinated J1~J4 motion while keeping the tool pitch approximately fixed.
constexpr float MAX_XZ_SPEED_MM_S = 90.0f;
constexpr float MAX_Y_SPEED_MM_S = 70.0f;
constexpr float MAX_WRIST_SPEED_DEG_S = 50.0f;
constexpr float MAX_JOINT_SPEED_DEG_S = 38.0f;
constexpr float AXIS_FILTER_ALPHA = 0.18f;
constexpr int16_t AXIS_DEADBAND = 35;

constexpr float GRIPPER_OPEN_DEG = 0.0f;
constexpr float GRIPPER_CLOSED_DEG = 35.0f;

constexpr uint16_t CONTROL_PERIOD_MS = 20;
constexpr uint16_t ATTACH_INTERVAL_MS = 250;
constexpr uint16_t HOME_HOLD_MS = 2000;
constexpr uint16_t PAYLOAD_TIMEOUT_MS = 100;
constexpr uint16_t DEBUG_PERIOD_MS = 500;
// At 20 ms/cycle, the longest READY<->HOME travel takes about 7.4 s toward
// HOME and about 3.0 s toward READY. Active control is faster but still slew
// limited separately for the heavy arm joints and the wrist/gripper.
constexpr int16_t HOME_SLEW_US_PER_CYCLE = 2;
constexpr int16_t READY_SLEW_US_PER_CYCLE = 5;
constexpr int16_t ACTIVE_SLEW_US_PER_CYCLE[JOINT_COUNT] = {
  6, 6, 6, 6, 10, 8
};

}  // namespace Config

// Semantic payload used by the arm controller. The transport packet layout,
// framing, CRC and receive code will be connected later.
struct ManipulatorPayload {
  int16_t throttle;
  int16_t yaw;
  int16_t roll;
  int16_t pitch;
  uint8_t sa;
  uint8_t sc;
  uint8_t flags;
  uint8_t switches;
  bool armMode;
  bool controlEnabled;
  bool kill;
  bool fault;
};

namespace UartProtocol {

constexpr uint8_t PACKET_SIZE = 16;
constexpr uint8_t SYNC_0 = 0xA5;
constexpr uint8_t SYNC_1 = 0x5A;
constexpr uint8_t VERSION = 0x01;
// Flags bit 0: controller connected, bit 1: motion armed,
// bit 2: STM32 ARM mode allowed. Switches.SC must also equal 1.
constexpr uint8_t REQUIRED_SAFETY_FLAGS = 0x07;

uint16_t crc16CcittFalse(const uint8_t *data, uint8_t length) {
  uint16_t crc = 0xFFFFU;

  for (uint8_t i = 0; i < length; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8U;

    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000U)
                ? static_cast<uint16_t>((crc << 1U) ^ 0x1021U)
                : static_cast<uint16_t>(crc << 1U);
    }
  }

  return crc;
}

int16_t readI16LE(const uint8_t *data) {
  const uint16_t raw = static_cast<uint16_t>(data[0]) |
                       (static_cast<uint16_t>(data[1]) << 8U);
  return static_cast<int16_t>(raw);
}

class Receiver {
 public:
  bool poll(Stream &stream, ManipulatorPayload &out) {
    bool received = false;

    while (stream.available() > 0) {
      const uint8_t byte = static_cast<uint8_t>(stream.read());
      ++byteCount_;

      if (index_ == 0U) {
        if (byte == SYNC_0) buffer_[index_++] = byte;
        continue;
      }

      if (index_ == 1U) {
        if (byte == SYNC_1) {
          buffer_[index_++] = byte;
        } else if (byte == SYNC_0) {
          buffer_[0] = byte;
        } else {
          index_ = 0U;
        }
        continue;
      }

      buffer_[index_++] = byte;

      if (index_ == PACKET_SIZE) {
        ManipulatorPayload decoded = {};

        if (decode(decoded)) {
          out = decoded;
          received = true;
          index_ = 0U;
        } else if (duplicate_) {
          // A complete duplicate frame is ignored but framing remains valid.
          index_ = 0U;
        } else {
          resynchronize();
        }
      }
    }

    return received;
  }

  uint32_t byteCount() const { return byteCount_; }
  uint32_t validFrameCount() const { return validFrameCount_; }
  uint32_t versionErrorCount() const { return versionErrorCount_; }
  uint32_t crcErrorCount() const { return crcErrorCount_; }
  uint32_t duplicateCount() const { return duplicateCount_; }

 private:
  bool decode(ManipulatorPayload &out) {
    duplicate_ = false;
    if (buffer_[2] != VERSION) {
      ++versionErrorCount_;
      return false;
    }

    const uint16_t receivedCrc = static_cast<uint16_t>(buffer_[14]) |
                                 (static_cast<uint16_t>(buffer_[15]) << 8U);

    if (crc16CcittFalse(buffer_, 14U) != receivedCrc) {
      ++crcErrorCount_;
      return false;
    }

    const uint8_t sequence = buffer_[3];
    if (sequenceSeen_ && (sequence == lastSequence_)) {
      duplicate_ = true;
      ++duplicateCount_;
      return false;
    }

    lastSequence_ = sequence;
    sequenceSeen_ = true;

    const uint8_t flags = buffer_[4];
    const uint8_t switches = buffer_[5];
    out.roll = readI16LE(&buffer_[6]);
    out.pitch = readI16LE(&buffer_[8]);
    out.throttle = readI16LE(&buffer_[10]);
    out.yaw = readI16LE(&buffer_[12]);
    out.sa = switches & 0x01U;
    const uint8_t sc = (switches >> 3U) & 0x03U;
    out.sc = sc;
    out.flags = flags;
    out.switches = switches;
    out.armMode = (sc == 1U);
    out.controlEnabled =
        (flags & REQUIRED_SAFETY_FLAGS) == REQUIRED_SAFETY_FLAGS;
    out.kill = false;
    out.fault = false;
    ++validFrameCount_;
    return true;
  }

  void resynchronize() {
    for (uint8_t start = 1U; start < (PACKET_SIZE - 1U); ++start) {
      if ((buffer_[start] == SYNC_0) &&
          (buffer_[start + 1U] == SYNC_1)) {
        const uint8_t remaining = PACKET_SIZE - start;
        memmove(buffer_, &buffer_[start], remaining);
        index_ = remaining;
        return;
      }
    }

    if (buffer_[PACKET_SIZE - 1U] == SYNC_0) {
      buffer_[0] = SYNC_0;
      index_ = 1U;
    } else {
      index_ = 0U;
    }
  }

  uint8_t buffer_[PACKET_SIZE] = {};
  uint8_t index_ = 0U;
  uint8_t lastSequence_ = 0U;
  bool sequenceSeen_ = false;
  bool duplicate_ = false;
  uint32_t byteCount_ = 0U;
  uint32_t validFrameCount_ = 0U;
  uint32_t versionErrorCount_ = 0U;
  uint32_t crcErrorCount_ = 0U;
  uint32_t duplicateCount_ = 0U;
};

}  // namespace UartProtocol

int16_t clampAxisInput(int16_t value) {
  if (value < -1000) return -1000;
  if (value > 1000) return 1000;
  return value;
}

Servo servos[Config::JOINT_COUNT];
UartProtocol::Receiver uartReceiver;
ManipulatorPayload command = {};

float jointDeg[Config::JOINT_COUNT] = {};
int16_t currentUs[Config::JOINT_COUNT] = {};
int16_t targetUs[Config::JOINT_COUNT] = {};
float filteredRoll = 0.0f;
float filteredPitch = 0.0f;
float filteredThrottle = 0.0f;
float filteredYaw = 0.0f;

bool armActive = false;
bool readyReached = false;
uint8_t attachedCount = 0;
uint32_t lastAttachMs = 0;
uint32_t lastControlMs = 0;
uint32_t lastDebugMs = 0;
uint32_t lastPayloadMs = 0;
uint32_t homeHoldStartMs = 0;

enum ArmEntryPhase : uint8_t {
  ARM_ENTRY_ATTACH_START = 0,
  ARM_ENTRY_MOVE_HOME,
  ARM_ENTRY_HOLD_HOME,
  ARM_ENTRY_MOVE_READY,
  ARM_ENTRY_ACTIVE
};

ArmEntryPhase armEntryPhase = ARM_ENTRY_ATTACH_START;

float clampFloat(float value, float minimum, float maximum) {
  if (value < minimum) return minimum;
  if (value > maximum) return maximum;
  return value;
}

void loadHomeCalibration() {
  for (uint8_t joint = 0U; joint < Config::JOINT_COUNT; ++joint) {
    Config::HOME_US[joint] = Config::DEFAULT_HOME_US[joint];
  }

  uint16_t magic = 0U;
  EEPROM.get(Config::CALIBRATION_MAGIC_ADDRESS, magic);
  const uint8_t version = EEPROM.read(Config::CALIBRATION_VERSION_ADDRESS);
  if ((magic != Config::CALIBRATION_MAGIC) ||
      (version != Config::CALIBRATION_VERSION)) {
    return;
  }

  // A valid calibration stores J1~J6 consecutively from EEPROM address 4.
  for (uint8_t joint = 0U; joint < Config::JOINT_COUNT; ++joint) {
    int16_t savedUs = 0;
    const int address = Config::CALIBRATION_DATA_ADDRESS +
                        static_cast<int>(joint) * sizeof(int16_t);
    EEPROM.get(address, savedUs);

    if ((savedUs >= Config::CALIBRATION_MIN_US) &&
        (savedUs <= Config::CALIBRATION_MAX_US)) {
      Config::HOME_US[joint] = savedUs;
    } else {
      Config::HOME_US[joint] = Config::DEFAULT_HOME_US[joint];
    }
  }
}

void printHomeCalibration() {
  Serial.print(F("HOME_US={"));
  for (uint8_t joint = 0U; joint < Config::JOINT_COUNT; ++joint) {
    Serial.print(Config::HOME_US[joint]);
    if (joint + 1U < Config::JOINT_COUNT) Serial.print(',');
  }
  Serial.println('}');
}

float axisNormalized(int16_t value) {
  const int16_t magnitude = (value < 0) ? -value : value;
  if (magnitude <= Config::AXIS_DEADBAND) return 0.0f;

  const float normalized =
      static_cast<float>(magnitude - Config::AXIS_DEADBAND) /
      static_cast<float>(1000 - Config::AXIS_DEADBAND);
  return (value < 0) ? -normalized : normalized;
}

void resetFilteredAxes() {
  filteredRoll = 0.0f;
  filteredPitch = 0.0f;
  filteredThrottle = 0.0f;
  filteredYaw = 0.0f;
}

void updateFilteredAxes() {
  const float alpha = Config::AXIS_FILTER_ALPHA;
  filteredRoll += alpha * (axisNormalized(command.roll) - filteredRoll);
  filteredPitch += alpha * (axisNormalized(command.pitch) - filteredPitch);
  filteredThrottle +=
      alpha * (axisNormalized(command.throttle) - filteredThrottle);
  filteredYaw += alpha * (axisNormalized(command.yaw) - filteredYaw);
}

int16_t jointToPulse(uint8_t joint, float angleDeg) {
  const float pulse = static_cast<float>(Config::HOME_US[joint]) +
                      static_cast<float>(Config::SERVO_DIRECTION[joint]) *
                      angleDeg * Config::US_PER_DEG[joint];

  return static_cast<int16_t>(clampFloat(
      pulse,
      static_cast<float>(Config::MIN_US[joint]),
      static_cast<float>(Config::MAX_US[joint])));
}

void updateTargetPulses() {
  for (uint8_t joint = 0; joint < Config::JOINT_COUNT; ++joint) {
    jointDeg[joint] = clampFloat(jointDeg[joint],
                                 Config::JOINT_MIN_DEG[joint],
                                 Config::JOINT_MAX_DEG[joint]);
    targetUs[joint] = jointToPulse(joint, jointDeg[joint]);
  }
}

void setReadyPoseTarget() {
  for (uint8_t joint = 0; joint < Config::JOINT_COUNT; ++joint) {
    jointDeg[joint] = Config::READY_DEG[joint];
  }

  updateTargetPulses();
}

void setHomePoseTarget() {
  for (uint8_t joint = 0; joint < Config::JOINT_COUNT; ++joint) {
    jointDeg[joint] = 0.0f;
    targetUs[joint] = Config::HOME_US[joint];
  }
}

void beginArmMode(uint32_t nowMs) {
  armActive = true;
  readyReached = false;
  attachedCount = 0;
  lastAttachMs = nowMs - Config::ATTACH_INTERVAL_MS;
  armEntryPhase = ARM_ENTRY_ATTACH_START;
  resetFilteredAxes();

  // Keep the previous physical pose while channels are attached one by one.
  for (uint8_t joint = 0; joint < Config::JOINT_COUNT; ++joint) {
    targetUs[joint] = currentUs[joint];
  }
}

void detachAll() {
  for (uint8_t joint = 0; joint < Config::JOINT_COUNT; ++joint) {
    if (servos[joint].attached()) servos[joint].detach();
  }

  armActive = false;
  readyReached = false;
  attachedCount = 0;
}

void serviceSequentialAttach(uint32_t nowMs) {
  if (!armActive ||
      (attachedCount >= Config::JOINT_COUNT) ||
      ((nowMs - lastAttachMs) < Config::ATTACH_INTERVAL_MS)) {
    return;
  }

  const uint8_t joint = attachedCount;
  servos[joint].attach(Config::SERVO_PIN[joint], 500, 2500);

  // On the first boot, currentUs is the calibrated READY pulse. After re-entry,
  // retain the last commanded pulse so the arm does not jump to another pose.
  servos[joint].writeMicroseconds(currentUs[joint]);

  ++attachedCount;
  lastAttachMs = nowMs;

  if (attachedCount == Config::JOINT_COUNT) {
    armEntryPhase = ARM_ENTRY_MOVE_HOME;
    setHomePoseTarget();
  }
}

void integrateCartesianCommand(float dtSec) {
  const float vx = filteredPitch * Config::MAX_XZ_SPEED_MM_S;
  // Hold SA to use the analog Throttle axis for proportional gripper control.
  // Release SA to hold the last gripper angle and restore vertical motion.
  const float vz = command.sa
      ? 0.0f
      : filteredThrottle * Config::MAX_XZ_SPEED_MM_S;
  const float vy = filteredRoll * Config::MAX_Y_SPEED_MM_S;

  const float q2 = radians(jointDeg[1]);
  const float q3 = radians(jointDeg[2]);
  const float q23 = q2 + q3;

  const float s2 = sin(q2);
  const float c2 = cos(q2);
  const float s23 = sin(q23);
  const float c23 = cos(q23);

  // q2=0 is the straight-up software zero shown in image 1.
  const float j11 = Config::LINK1_MM * c2 + Config::LINK2_MM * c23;
  const float j12 = Config::LINK2_MM * c23;
  const float j21 = -Config::LINK1_MM * s2 - Config::LINK2_MM * s23;
  const float j22 = -Config::LINK2_MM * s23;
  const float determinant = j11 * j22 - j12 * j21;

  if (fabs(determinant) > 1000.0f) {
    float q2Dot = ( j22 * vx - j12 * vz) / determinant;
    float q3Dot = (-j21 * vx + j11 * vz) / determinant;
    const float maxJointRate = radians(Config::MAX_JOINT_SPEED_DEG_S);

    q2Dot = clampFloat(q2Dot, -maxJointRate, maxJointRate);
    q3Dot = clampFloat(q3Dot, -maxJointRate, maxJointRate);

    jointDeg[1] += degrees(q2Dot * dtSec);
    jointDeg[2] += degrees(q3Dot * dtSec);
  }

  const float radialMm = max(100.0f,
      Config::LINK1_MM * sin(q2) + Config::LINK2_MM * sin(q23));
  float q1Dot = vy / radialMm;
  const float maxBaseRate = radians(Config::MAX_JOINT_SPEED_DEG_S);
  q1Dot = clampFloat(q1Dot, -maxBaseRate, maxBaseRate);
  jointDeg[0] += degrees(q1Dot * dtSec);

  // Keep the tool pitch used by the folded ready pose.
  const float readyToolPitch = Config::READY_DEG[1] +
                               Config::READY_DEG[2] +
                               Config::READY_DEG[3];
  jointDeg[3] = readyToolPitch - jointDeg[1] - jointDeg[2];

  jointDeg[4] += filteredYaw * Config::MAX_WRIST_SPEED_DEG_S * dtSec;
  if (command.sa) {
    const float gripperRatio = clampFloat(
        (filteredThrottle + 1.0f) * 0.5f, 0.0f, 1.0f);
    jointDeg[5] = Config::GRIPPER_OPEN_DEG +
        gripperRatio *
        (Config::GRIPPER_CLOSED_DEG - Config::GRIPPER_OPEN_DEG);
  }

  updateTargetPulses();
}

bool slewServos() {
  bool allAtTarget = true;

  for (uint8_t joint = 0; joint < attachedCount; ++joint) {
    int16_t slewStep = Config::READY_SLEW_US_PER_CYCLE;
    if (armEntryPhase == ARM_ENTRY_MOVE_HOME) {
      slewStep = Config::HOME_SLEW_US_PER_CYCLE;
    } else if (armEntryPhase == ARM_ENTRY_ACTIVE) {
      slewStep = Config::ACTIVE_SLEW_US_PER_CYCLE[joint];
    }

    const int16_t error = targetUs[joint] - currentUs[joint];

    if (error > slewStep) {
      currentUs[joint] += slewStep;
      allAtTarget = false;
    } else if (error < -slewStep) {
      currentUs[joint] -= slewStep;
      allAtTarget = false;
    } else {
      currentUs[joint] = targetUs[joint];
    }

    servos[joint].writeMicroseconds(currentUs[joint]);
  }

  return (attachedCount == Config::JOINT_COUNT) && allAtTarget;
}

bool payloadAllowsArm() {
  return command.armMode &&
         command.controlEnabled &&
         !command.kill &&
         !command.fault;
}

void handlePayload(const ManipulatorPayload &received, uint32_t nowMs) {
  command = received;
  command.throttle = clampAxisInput(command.throttle);
  command.yaw = clampAxisInput(command.yaw);
  command.roll = clampAxisInput(command.roll);
  command.pitch = clampAxisInput(command.pitch);
  command.sa = command.sa ? 1U : 0U;

  // Every valid packet updates the latest controller state. ARM-disabled
  // packets do not detach or return home; they simply stop target updates.
  lastPayloadMs = nowMs;
}

void printDebugStatus(uint32_t nowMs) {
  if ((nowMs - lastDebugMs) < Config::DEBUG_PERIOD_MS) return;
  lastDebugMs = nowMs;

  Serial.print(F("DBG bytes="));
  Serial.print(uartReceiver.byteCount());
  Serial.print(F(" ok="));
  Serial.print(uartReceiver.validFrameCount());
  Serial.print(F(" verErr="));
  Serial.print(uartReceiver.versionErrorCount());
  Serial.print(F(" crcErr="));
  Serial.print(uartReceiver.crcErrorCount());
  Serial.print(F(" dup="));
  Serial.print(uartReceiver.duplicateCount());
  Serial.print(F(" ageMs="));
  if (lastPayloadMs == 0U) {
    Serial.print(F("NONE"));
  } else {
    Serial.print(nowMs - lastPayloadMs);
  }

  Serial.print(F(" ready="));
  Serial.print(readyReached ? 1 : 0);
  Serial.print(F(" flags=0x"));
  Serial.print(command.flags, HEX);
  Serial.print(F(" sw=0x"));
  Serial.print(command.switches, HEX);
  Serial.print(F(" SC="));
  Serial.print(command.sc);
  Serial.print(F(" SA="));
  Serial.print(command.sa);
  Serial.print(F(" safety="));
  Serial.print(command.controlEnabled ? 1 : 0);
  Serial.print(F(" arm="));
  Serial.print(command.armMode ? 1 : 0);
  Serial.print(F(" allow="));
  Serial.print(payloadAllowsArm() ? 1 : 0);
  Serial.print(F(" R="));
  Serial.print(command.roll);
  Serial.print(F(" P="));
  Serial.print(command.pitch);
  Serial.print(F(" T="));
  Serial.print(command.throttle);
  Serial.print(F(" Y="));
  Serial.println(command.yaw);
}

void setup() {
  Serial.begin(115200);
  loadHomeCalibration();
  printHomeCalibration();

  for (uint8_t joint = 0; joint < Config::JOINT_COUNT; ++joint) {
    // The physical arm is expected to be left in the folded READY pose.
    // Build its startup pulses from the calibrated software-zero values.
    jointDeg[joint] = Config::READY_DEG[joint];
    currentUs[joint] = jointToPulse(joint, jointDeg[joint]);
    targetUs[joint] = currentUs[joint];
  }

  // Startup positioning is independent of communication and ARM mode.
  beginArmMode(millis());
}

void loop() {
  const uint32_t nowMs = millis();
  ManipulatorPayload received = {};

  if (uartReceiver.poll(Serial, received)) {
    handlePayload(received, nowMs);
  }

  printDebugStatus(nowMs);

  if (!armActive) return;

  serviceSequentialAttach(nowMs);

  if ((nowMs - lastControlMs) < Config::CONTROL_PERIOD_MS) return;
  lastControlMs = nowMs;

  const bool payloadFresh =
      (lastPayloadMs != 0U) &&
      ((nowMs - lastPayloadMs) <= Config::PAYLOAD_TIMEOUT_MS);

  // Controller input starts after boot positioning, fresh data, safety flags,
  // and the SC == 1 ARM-mode selection are all valid.
  const bool controllerEnabled =
      readyReached && payloadFresh && payloadAllowsArm();

  if (controllerEnabled) {
    updateFilteredAxes();
    integrateCartesianCommand(Config::CONTROL_PERIOD_MS * 0.001f);
  } else {
    resetFilteredAxes();
  }

  const bool allAtTarget = slewServos();

  if (allAtTarget) {
    if (armEntryPhase == ARM_ENTRY_MOVE_HOME) {
      armEntryPhase = ARM_ENTRY_HOLD_HOME;
      homeHoldStartMs = nowMs;
    } else if ((armEntryPhase == ARM_ENTRY_HOLD_HOME) &&
               ((nowMs - homeHoldStartMs) >= Config::HOME_HOLD_MS)) {
      armEntryPhase = ARM_ENTRY_MOVE_READY;
      setReadyPoseTarget();
    } else if (armEntryPhase == ARM_ENTRY_MOVE_READY) {
      armEntryPhase = ARM_ENTRY_ACTIVE;
      readyReached = true;
    }
  }
}
