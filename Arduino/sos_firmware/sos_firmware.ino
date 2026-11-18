/*
  VisionMate emergency SOS subsystem (Section 4.9 of the paper)

  Board:    Arduino Uno R3
  GPS:      u-blox NEO-6M  (NMEA 0183, 9600 baud)
  GSM:      SIM800C        (AT commands, 9600 baud, 2G networks only)
  Trigger:  momentary push-button between D2 and GND (internal pull-up)

  The Arduino runs independently of the Raspberry Pi. It reads the GPS
  continuously and keeps the most recent valid fix in memory. Pressing the SOS
  button triggers an interrupt that only sets a flag; the main loop then
  formats the held coordinates into a text message and has the SIM800C send it
  by SMS to the preconfigured contact. Because a warm fix is already held, the
  SMS goes out without waiting for a new GPS acquisition.

  Wiring (see Arduino/README.md):
    NEO-6M  TX  -> D4   (RX of gpsSerial)
    NEO-6M  RX  <- D5   (not needed for reading; may be left unconnected)
    SIM800C TXD -> D7   (RX of gsmSerial)
    SIM800C RXD <- D8   (TX of gsmSerial; use a level shifter/divider if your
                         SIM800C board has 2.8 V/3.3 V logic)
    SOS button  -> D2 and GND
    Status LED  -> D13 (on-board LED)

  Library: TinyGPSPlus (Arduino Library Manager: "TinyGPSPlus" by Mikal Hart)

  Before uploading, set EMERGENCY_NUMBER below to the contact's number in
  international format, e.g. "+919876543210".
*/

#include <SoftwareSerial.h>
#include <TinyGPSPlus.h>

// ---------------- configuration ----------------
const char EMERGENCY_NUMBER[] = "+91XXXXXXXXXX";   // <-- set the contact's number

const uint8_t SOS_BUTTON_PIN = 2;      // INT0
const uint8_t GPS_RX_PIN = 4;          // Arduino receives from GPS TX
const uint8_t GPS_TX_PIN = 5;
const uint8_t GSM_RX_PIN = 7;          // Arduino receives from SIM800C TXD
const uint8_t GSM_TX_PIN = 8;
const uint8_t STATUS_LED_PIN = 13;

const unsigned long GPS_BAUD = 9600;
const unsigned long GSM_BAUD = 9600;
const unsigned long DEBOUNCE_MS = 2000;   // ignore repeat presses within 2 s
const uint8_t SMS_ATTEMPTS = 3;

// ---------------- state ----------------
SoftwareSerial gpsSerial(GPS_RX_PIN, GPS_TX_PIN);
SoftwareSerial gsmSerial(GSM_RX_PIN, GSM_TX_PIN);
TinyGPSPlus gps;

volatile bool sosRequested = false;
volatile unsigned long lastPressMs = 0;

bool haveFix = false;          // most recent valid fix held in memory
double fixLat = 0.0;
double fixLng = 0.0;
unsigned long fixTakenMs = 0;

// ---------------- interrupt handler: only sets a flag ----------------
void onSosPressed() {
  unsigned long now = millis();
  if (now - lastPressMs > DEBOUNCE_MS) {
    lastPressMs = now;
    sosRequested = true;
  }
}

// ---------------- GSM helpers ----------------
// Read from the SIM800C until `expected` appears or the timeout passes.
bool waitFor(const char *expected, unsigned long timeoutMs) {
  unsigned long start = millis();
  size_t matched = 0;
  size_t len = strlen(expected);
  while (millis() - start < timeoutMs) {
    while (gsmSerial.available()) {
      char c = gsmSerial.read();
      Serial.write(c);
      if (c == expected[matched]) {
        matched++;
        if (matched == len) return true;
      } else {
        matched = (c == expected[0]) ? 1 : 0;
      }
    }
  }
  return false;
}

bool sendCommand(const char *cmd, const char *expected, unsigned long timeoutMs) {
  while (gsmSerial.available()) gsmSerial.read();   // clear old output
  gsmSerial.println(cmd);
  return waitFor(expected, timeoutMs);
}

bool initGsm() {
  gsmSerial.listen();
  for (uint8_t i = 0; i < 5; i++) {
    if (sendCommand("AT", "OK", 1000)) {
      sendCommand("ATE0", "OK", 1000);              // no echo
      sendCommand("AT+CMGF=1", "OK", 1000);         // SMS text mode
      sendCommand("AT+CSCS=\"GSM\"", "OK", 1000);   // GSM character set
      gpsSerial.listen();
      return true;
    }
    delay(1000);
  }
  gpsSerial.listen();
  return false;
}

// Build the SMS text with the held fix (an SMS holds up to 160 characters).
void buildMessage(char *buf, size_t size) {
  if (haveFix) {
    char lat[12], lng[12];
    dtostrf(fixLat, 1, 6, lat);
    dtostrf(fixLng, 1, 6, lng);
    unsigned long ageS = (millis() - fixTakenMs) / 1000UL;
    snprintf(buf, size,
             "SOS! VisionMate user needs help. Location: https://maps.google.com/?q=%s,%s (fix %lus old)",
             lat, lng, ageS);
  } else {
    snprintf(buf, size, "SOS! VisionMate user needs help. GPS location not available yet.");
  }
}

bool sendSms(const char *text) {
  gsmSerial.listen();               // SoftwareSerial can listen to one port at a time
  bool ok = false;
  sendCommand("AT+CMGF=1", "OK", 1000);
  while (gsmSerial.available()) gsmSerial.read();
  gsmSerial.print("AT+CMGS=\"");
  gsmSerial.print(EMERGENCY_NUMBER);
  gsmSerial.println("\"");
  if (waitFor(">", 5000)) {
    gsmSerial.print(text);
    gsmSerial.write(26);            // Ctrl+Z ends the message
    ok = waitFor("+CMGS:", 20000);  // network acknowledgement
  }
  gpsSerial.listen();               // back to reading the GPS
  return ok;
}

// ---------------- setup / loop ----------------
void setup() {
  Serial.begin(9600);
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(SOS_BUTTON_PIN, INPUT_PULLUP);

  gpsSerial.begin(GPS_BAUD);
  gsmSerial.begin(GSM_BAUD);

  delay(3000);                      // let the SIM800C start
  Serial.println(initGsm() ? F("SIM800C ready") : F("SIM800C not responding"));

  gpsSerial.listen();
  attachInterrupt(digitalPinToInterrupt(SOS_BUTTON_PIN), onSosPressed, FALLING);
  Serial.println(F("VisionMate SOS ready"));
}

void loop() {
  // 1. Keep reading NMEA sentences and hold the most recent valid fix.
  while (gpsSerial.available()) {
    if (gps.encode(gpsSerial.read()) && gps.location.isValid() && gps.location.isUpdated()) {
      fixLat = gps.location.lat();
      fixLng = gps.location.lng();
      fixTakenMs = millis();
      haveFix = true;
    }
  }

  // 2. Handle an SOS request raised by the interrupt.
  bool requested;
  noInterrupts();
  requested = sosRequested;
  sosRequested = false;
  interrupts();

  if (requested) {
    digitalWrite(STATUS_LED_PIN, HIGH);
    char message[161];
    buildMessage(message, sizeof(message));
    Serial.print(F("Sending SOS: "));
    Serial.println(message);

    bool sent = false;
    for (uint8_t attempt = 1; attempt <= SMS_ATTEMPTS && !sent; attempt++) {
      sent = sendSms(message);
      Serial.print(F("Attempt "));
      Serial.print(attempt);
      Serial.println(sent ? F(": sent") : F(": failed"));
      if (!sent) delay(2000);
    }
    digitalWrite(STATUS_LED_PIN, LOW);
  }
}
