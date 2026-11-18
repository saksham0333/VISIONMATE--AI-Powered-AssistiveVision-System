# Emergency SOS subsystem (Arduino Uno R3)

Firmware for the SOS path described in Section 4.9 of the paper. It runs on its own Arduino and does not depend on the Raspberry Pi.

## Parts
- Arduino Uno R3
- u-blox NEO-6M GPS module
- SIM800C GSM module with a SIM that works on **2G** (the SIM800C does not support 4G/LTE; Airtel was used in testing)
- Momentary push-button (SOS trigger)

## Wiring

| From | To | Note |
|---|---|---|
| NEO-6M TX | Arduino D4 | GPS data in |
| NEO-6M RX | Arduino D5 | optional |
| SIM800C TXD | Arduino D7 | GSM data in |
| SIM800C RXD | Arduino D8 | use a voltage divider or level shifter if the SIM800C board has 2.8 V/3.3 V logic |
| SOS button | Arduino D2 and GND | uses the internal pull-up and interrupt INT0 |
| GND of all modules | Arduino GND | common ground |
| NEO-6M VCC | 5 V or 3.3 V | as marked on the module |
| SIM800C VCC | 5 V supply | the SIM800C draws current peaks of about 2 A while transmitting; do not power it from the Arduino's 5 V pin |

The subsystem shares only the 5 V power rail with the Raspberry Pi.

## Upload
1. Install the Arduino IDE.
2. Library Manager: install **TinyGPSPlus** (by Mikal Hart).
3. Open `sos_firmware/sos_firmware.ino` and set `EMERGENCY_NUMBER` to the contact's number in international format, for example `"+919876543210"`.
4. Board: *Arduino Uno*. Upload.
5. Open the Serial Monitor at 9600 baud to see "SIM800C ready" and "VisionMate SOS ready".

## Behaviour
- The GPS is read continuously and the most recent valid fix is kept in memory.
- Pressing the button raises an interrupt that only sets a flag (presses within 2 s are ignored).
- The main loop sends an SMS with a Google Maps link to the held fix and its age, for example:
  `SOS! VisionMate user needs help. Location: https://maps.google.com/?q=12.840600,80.153400 (fix 3s old)`
- If no fix has been obtained yet, the SMS says that the location is not available.
- Up to three attempts are made if sending fails. The on-board LED is lit while sending.
