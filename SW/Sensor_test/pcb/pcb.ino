#include <SPI.h>

/*
  Arduino Mega + MCP3008 보드
  읽는 입력:
    1-1 -> U1 MCP3008 CH0
    1-4 -> U1 MCP3008 CH3

  배선:
    보드 GND  -> Mega GND
    보드 5V   -> Mega 5V
    보드 CLK  -> Mega 52 (SCK)
    보드 DOUT -> Mega 50 (MISO)
    보드 DIN  -> Mega 51 (MOSI)
    보드 CS_1 -> Mega 53
*/

static const uint8_t MCP3008_CS = 53;
static const float ADC_REFERENCE_VOLTAGE = 5.0f;

// MCP3008의 단일 종단 채널 0~7을 읽는다.
uint16_t readMCP3008(uint8_t channel)
{
  if (channel > 7) {
    return 0;
  }

  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));

  digitalWrite(MCP3008_CS, LOW);

  SPI.transfer(0x01);                         // Start bit
  uint8_t highByte = SPI.transfer(0x80 | (channel << 4));
  uint8_t lowByte  = SPI.transfer(0x00);

  digitalWrite(MCP3008_CS, HIGH);

  SPI.endTransaction();

  return ((highByte & 0x03) << 8) | lowByte; // 10비트 값: 0~1023
}

void setup()
{
  Serial.begin(115200);

  pinMode(MCP3008_CS, OUTPUT);
  digitalWrite(MCP3008_CS, HIGH);

  SPI.begin();

  Serial.println("MCP3008 reading start");
}

void loop()
{
  // 사진상 예상 채널 배치
  uint16_t value_1_1 = readMCP3008(0); // 1-1
  uint16_t value_1_4 = readMCP3008(3); // 1-4

  float voltage_1_1 = value_1_1 * ADC_REFERENCE_VOLTAGE / 1023.0f;
  float voltage_1_4 = value_1_4 * ADC_REFERENCE_VOLTAGE / 1023.0f;

  Serial.print("1-1: ");
  Serial.print(value_1_1);
  Serial.print("  ");
  Serial.print(voltage_1_1, 3);
  Serial.print(" V");

  Serial.print("    1-4: ");
  Serial.print(value_1_4);
  Serial.print("  ");
  Serial.print(voltage_1_4, 3);
  Serial.println(" V");

  delay(100);
}