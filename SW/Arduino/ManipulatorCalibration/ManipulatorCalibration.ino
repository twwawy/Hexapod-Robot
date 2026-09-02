#include <Servo.h>
#include <EEPROM.h>

/*
 * Arduino Uno manipulator software-zero calibration.
 *
 * Servo order and signal pins:
 *   J1~J6 = D5, D6, D7, D8, D9, D10
 *
 * Serial Monitor:
 *   115200 baud, Newline or Both NL & CR
 *   up   : increase the current pulse by 10 us
 *   down : decrease the current pulse by 10 us
 *   ok   : save the current joint and advance to the next joint
 *
 * Servo power must come from the external servo supplies, never Uno 5 V.
 */

namespace Config {

constexpr uint8_t JOINT_COUNT = 6;
constexpr uint8_t SERVO_PIN[JOINT_COUNT] = {5, 6, 7, 8, 9, 10};

// Previously measured zero values are safe starting points for fine tuning.
constexpr int16_t START_HOME_US[JOINT_COUNT] = {
  1290, 1600, 1480, 1510, 1610, 1540
};

constexpr int16_t MIN_PULSE_US = 1000;
constexpr int16_t MAX_PULSE_US = 2000;
constexpr int16_t STEP_US = 10;
constexpr uint16_t ATTACH_SETTLE_MS = 1000;
constexpr uint16_t NEXT_JOINT_DELAY_MS = 500;

// Shared EEPROM layout with ManipulatorController.ino.
constexpr uint16_t CALIBRATION_MAGIC = 0x4D43;
constexpr uint8_t CALIBRATION_VERSION = 1;
constexpr int CALIBRATION_MAGIC_ADDRESS = 0;
constexpr int CALIBRATION_VERSION_ADDRESS = 2;
constexpr int CALIBRATION_DATA_ADDRESS = 4;

}  // namespace Config

Servo servos[Config::JOINT_COUNT];
int16_t zeroPulseUs[Config::JOINT_COUNT] = {};
uint8_t currentJoint = 0U;
int16_t currentPulseUs = Config::START_HOME_US[0];
bool calibrationFinished = false;

void printPosition() {
  Serial.print(F("J"));
  Serial.print(currentJoint + 1U);
  Serial.print(F(" (D"));
  Serial.print(Config::SERVO_PIN[currentJoint]);
  Serial.print(F(") pulse="));
  Serial.print(currentPulseUs);
  Serial.println(F(" us"));
}

void startCurrentJoint() {
  currentPulseUs = Config::START_HOME_US[currentJoint];
  zeroPulseUs[currentJoint] = currentPulseUs;

  servos[currentJoint].attach(Config::SERVO_PIN[currentJoint], 500, 2500);
  servos[currentJoint].writeMicroseconds(currentPulseUs);
  delay(Config::ATTACH_SETTLE_MS);

  Serial.println(F("-------------------------"));
  Serial.print(F("Start calibration: J"));
  Serial.print(currentJoint + 1U);
  Serial.print(F(" / D"));
  Serial.println(Config::SERVO_PIN[currentJoint]);
  printPosition();
}

void printAllValues() {
  Serial.println();
  Serial.println(F("=== J1~J6 calibration complete ==="));
  Serial.print(F("HOME_US={"));
  for (uint8_t joint = 0U; joint < Config::JOINT_COUNT; ++joint) {
    Serial.print(zeroPulseUs[joint]);
    if (joint + 1U < Config::JOINT_COUNT) Serial.print(',');
  }
  Serial.println(F("}"));
  Serial.println(F("Upload ManipulatorController.ino to use these values."));
}

void finishCalibration() {
  for (uint8_t joint = 0U; joint < Config::JOINT_COUNT; ++joint) {
    const int address = Config::CALIBRATION_DATA_ADDRESS +
                        static_cast<int>(joint) * sizeof(int16_t);
    EEPROM.put(address, zeroPulseUs[joint]);
  }

  // Write the validity marker last so an interrupted calibration is ignored.
  EEPROM.put(Config::CALIBRATION_MAGIC_ADDRESS, Config::CALIBRATION_MAGIC);
  EEPROM.update(Config::CALIBRATION_VERSION_ADDRESS,
                Config::CALIBRATION_VERSION);
  calibrationFinished = true;
  printAllValues();
}

void acceptCurrentJoint() {
  zeroPulseUs[currentJoint] = currentPulseUs;

  Serial.print(F("Saved J"));
  Serial.print(currentJoint + 1U);
  Serial.print(F(": "));
  Serial.print(currentPulseUs);
  Serial.println(F(" us"));

  if (currentJoint + 1U >= Config::JOINT_COUNT) {
    finishCalibration();
    return;
  }

  ++currentJoint;
  delay(Config::NEXT_JOINT_DELAY_MS);
  startCurrentJoint();
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);

  // Invalidate any older complete record until all six new values are saved.
  EEPROM.update(Config::CALIBRATION_VERSION_ADDRESS, 0U);

  Serial.println();
  Serial.println(F("=== J1~J6 manipulator zero calibration ==="));
  Serial.println(F("up/down: 10 us step, ok: save and continue"));
  startCurrentJoint();
}

void loop() {
  if (Serial.available() <= 0) return;

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toLowerCase();

  if (calibrationFinished) {
    Serial.println(F("Calibration is complete. Upload the controller sketch."));
    return;
  }

  if (command == "up") {
    currentPulseUs = constrain(currentPulseUs + Config::STEP_US,
                               Config::MIN_PULSE_US,
                               Config::MAX_PULSE_US);
    servos[currentJoint].writeMicroseconds(currentPulseUs);
    printPosition();
  } else if (command == "down") {
    currentPulseUs = constrain(currentPulseUs - Config::STEP_US,
                               Config::MIN_PULSE_US,
                               Config::MAX_PULSE_US);
    servos[currentJoint].writeMicroseconds(currentPulseUs);
    printPosition();
  } else if (command == "ok") {
    acceptCurrentJoint();
  } else if (command.length() > 0U) {
    Serial.println(F("Enter up, down, or ok."));
  }
}
