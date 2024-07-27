#include <WiFi.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>
#include "UbidotsEsp32Mqtt.h"

// Konstan untuk WiFi credentials
const char *WIFI_SSID = "Boens1";
const char *WIFI_PASS = "anaLEX0916";

// Pin assignments
#define DHTPIN 27  // DHT22 sensor pin
#define DHTTYPE DHT11  // DHT sensor type

// Global objects
DHT dht(DHTPIN, DHTTYPE);

// Fungsi untuk mendapatkan data suhu
float get_temperature_data() {
  float t = dht.readTemperature();
  if (isnan(t)) {
    Serial.println(F("Error reading temperature!"));
    return 0.0;  // Return default value on error
  } else {
    Serial.print(F("Temperature: "));
    Serial.print(t);
    Serial.println(F("°C"));
    return t;
  }
}

void setup() {
  Serial.begin(115200);
  dht.begin();
}

void loop() {
  get_temperature_data();
  delay(5000);
}