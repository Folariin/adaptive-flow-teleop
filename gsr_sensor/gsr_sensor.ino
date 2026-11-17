#include <Arduino.h>
#include <string.h>

constexpr unsigned long SAMPLE_INTERVAL_MS = 125;
constexpr size_t SESSION_LENGTH = 32;
constexpr size_t COMMAND_CAPACITY = sizeof("START,") + SESSION_LENGTH;

char session[SESSION_LENGTH + 1] = "";
char command[COMMAND_CAPACITY];
size_t commandLength = 0;
bool discardCommand = false;
unsigned long lastSample = 0;

bool validSession(const char *id) {
  if (strlen(id) != SESSION_LENGTH) {
    return false;
  }
  for (size_t i = 0; i < SESSION_LENGTH; ++i) {
    char c = id[i];
    if (!(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f')) {
      return false;
    }
  }
  return true;
}

void handleCommand() {
  if (strncmp(command, "START,", 6) == 0 && validSession(command + 6)) {
    const char *id = command + 6;
    if (strcmp(session, id) != 0) {
      strcpy(session, id);
      lastSample = millis();
    }
    // A retry acknowledges the same session without restarting its cadence.
    Serial.print(F("ACK,"));
    Serial.println(session);
  } else if (strncmp(command, "STOP,", 5) == 0 && session[0] &&
             strcmp(command + 5, session) == 0) {
    session[0] = '\0';
  }
}

void readCommands() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      if (!discardCommand) {
        command[commandLength] = '\0';
        handleCommand();
      }
      commandLength = 0;
      discardCommand = false;
    } else if (!discardCommand && c != '\r') {
      if (commandLength < sizeof(command) - 1) {
        command[commandLength++] = c;
      } else {
        discardCommand = true;
      }
    }
  }
}

void setup() {
  Serial.begin(9600);
}

void loop() {
  readCommands();
  if (!session[0]) {
    return;
  }
  unsigned long now = millis();
  if (now - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = now;
    int raw = analogRead(A0);
    Serial.print(F("DATA,"));
    Serial.print(session);
    Serial.print(',');
    Serial.print(now);
    Serial.print(',');
    Serial.println(raw);
  }
}
