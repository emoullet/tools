/****************************************************/
/* SIGNAUX ET SYSTEMES LINEAIRES - POLYTECH SORBONE */
/*          JANVIER 2026 - GUILLAUME MOREL          */
/****************************************************/
#include <math.h>

// Arduino library for communacting with SPI devices
#include <SPI.h>

// Library for CAN communications
#include <mcp2515_can.h>
const int SPI_CS_PIN = 10;
mcp2515_can CAN(SPI_CS_PIN);  // Creates the mcp2515_can class object for CAN communication, while setting CS pin


// Library for messages (Serial protocol)
#include "MsgManagement.h"
Message MSG;

/******************  GLOBAL VARIABLES *********************/
const int digitalPinNumber = 54;    // pin 0-> 53
const int analogicPinNumber = 16;   // pin 54->69 (A0->A15)
const int sum = digitalPinNumber+analogicPinNumber;
int pinList[sum] = {};
int pinRead[sum] = {};
int lastPinRead[sum] = {};
bool flagDigitalPinRead = false;
bool flagAnalogicPinRead = false;

// FOR ANALOG INPUT

// FOR PRINTING THROUGH THE SERIAL PORT

void readUserMessage(int *pinList) {
  char userCommand;
  int nbOfUserArgs;
  float userArgs[NB_MAX_MSG_ARGS + 1];
  int digitalPinFeedback[] = {};
  //int pinList[] = {};
  int analogicPinList[] = {};
  int digitalInputFeedback[] = {};

  MSG.ReadSerial(); // Gets entries from the serial port to build a message
  MSG.Get(&userCommand, &nbOfUserArgs, &(userArgs[0]));   // Gets the message when ready (otherwise, userCommand=0)
  if (userCommand != '0') { // If a message has arrived, do for interpretation
    switch (userCommand) {
      // Digital_output message ***********************************************************************
      case ('D'):                                         // message structure : D <n°pin> <value>
        for (int i = 1; i < nbOfUserArgs + 1; i = i + 2) {
          digitalWrite(userArgs[i], userArgs[i + 1]);
          Serial.print("ardunio mega : digital_output "); // TEST
          Serial.print("pin[");                           // TEST
          Serial.print(userArgs[i]);                      // TEST
          Serial.print("] = ");                           // TEST
          Serial.println(userArgs[i + 1]);                // TEST
        }
        break;

      // PWM_output message ***************************************************************************
      case ('P'):                                         // message structure : P <n°pin> <value>
        for (int i = 1; i < nbOfUserArgs + 1; i = i + 2) {
          analogWrite(userArgs[i], userArgs[i + 1]);      
          Serial.print("ardunio mega : pwm_output ");     // TEST
          Serial.print("pin[");                           // TEST
          Serial.print(userArgs[i]);                    // TEST
          Serial.print("] = ");                           // TEST
          Serial.print(userArgs[i + 1]);                  // TEST
        }
        break;

      // Setting message ******************************************************************************
      case ('S'):                                         // message structure : S <n°pin> <I/O> <value>
        for (int i = 1; i < nbOfUserArgs + 1; i = i + 2) {
          
          if (userArgs[i + 1] == 1) {                     // check if it's an digital/PWM output pin (1.0)
            if (userArgs[i] > digitalPinNumber and userArgs[i] < digitalPinNumber+analogicPinNumber ){
              Serial.print("ardunio mega WARNING : set ");// TEST
              Serial.print("pin[");                       // TEST
              Serial.print(userArgs[i]);                  // TEST
              Serial.println("] -> OUTPUT not possible"); // TEST
            }
            else if (userArgs[i] < digitalPinNumber) {                       // check if it's an digital pin (0 -> 53)
              pinMode(userArgs[i], OUTPUT);               // pinMode for : 
                                                          //  - Digital pins :  0 -> 53
                                                          //  - PWM pins :      0 -> 12
                                                          //  - Analogic pins : 54 = A0 -> ... -> 69 = A15
              pinList[int(userArgs[i])] = 1;              // store pin's I/O info
              Serial.print("ardunio mega : set ");        // TEST
              Serial.print("pin[");                       // TEST
              Serial.print(userArgs[i]);                  // TEST
              Serial.println("] -> OUTPUT");              // TEST
              
              if (userArgs[i + 2] == 1.0) {               // check if the default digital output pin value is HIGH (1.0)
                digitalWrite(userArgs[i], HIGH);
                Serial.print("ardunio mega : set ");      // TEST
                Serial.print("pin[");                     // TEST
                Serial.print(userArgs[i]);                // TEST
                Serial.println("] = HIGH");               // TEST
              }
              else if (userArgs[i + 2] == 0.0){           // if not, the default digital output pin value is LOW (0.0)
                digitalWrite(userArgs[i], LOW);           
                Serial.print("ardunio mega : set ");      // TEST
                Serial.print("pin[");                     // TEST
                Serial.print(userArgs[i]);                // TEST
                Serial.println("] = LOW");                // TEST
              }
              else {
                analogWrite(userArgs[i], userArgs[i + 2]);
                Serial.print("ardunio mega : set ");      // TEST
                Serial.print("pin[");                     // TEST
                Serial.print(userArgs[i]);                // TEST
                Serial.print("] = ");                     // TEST
                Serial.println(userArgs[i + 2]);          // TEST
              }
            }
            else {
                Serial.print("ardunio mega WARNING : set ");// TEST
                Serial.print("pin[");                     // TEST
                Serial.print(userArgs[i]);                // TEST
                Serial.println("] not accessible");       // TEST
            }
            i = i + 1;                                  // because of 1 param more for digital & analogic output pin
          }
          else if (userArgs[i + 1] == 0){                 // if not, it's an digital/analogic input pin (0.0)
            if (userArgs[i] < digitalPinNumber+analogicPinNumber ){
              pinMode(userArgs[i], INPUT);
              pinList[int(userArgs[i])] = 0;              // store pin's I/O info
              Serial.print("ardunio mega : set ");        // TEST
              Serial.print("pin[");                       // TEST
              Serial.print(userArgs[i]);                  // TEST
              Serial.println("] -> INPUT");               // TEST
            }
            else{
              Serial.print("ardunio mega WARNING : set ");// TEST
              Serial.print("pin[");                       // TEST
              Serial.print(userArgs[i]);                  // TEST
              Serial.println("] not accessible");         // TEST
            }
          }
          else {
            Serial.print("ardunio mega WARNING : set ");  // TEST
            Serial.print("pin[");                         // TEST
            Serial.print(userArgs[i]);                    // TEST
            Serial.println("] -> unknown I/O setting command");// TEST
          }
          
          Serial.print("pinList = [");                    // TEST
          for (int i=0 ; i < digitalPinNumber+analogicPinNumber ; i++){
            Serial.print(pinList[i]);                     // TEST
            Serial.print(" ");                            // TEST
          }                                               // TEST
          Serial.println("]");                            // TEST
        }
        break;

      default:
        Serial.println("COMMANDE INEXISTANTE");
        break;
    }
  }
}

/********************* THIS FUNCTION IS EXECUTED FIRST AND CONTAINS INITIALIZATION ***********/
void setup() {
  int i;
  char serialReceivedChar;
  int nothingReceived;
  
  for (int i = 0 ; i < digitalPinNumber + analogicPinNumber ; i++) {
    pinList[i] = 2;                                       // setting default value
    pinRead[i] = 2;                                       // setting default value
    lastPinRead[i] = 2;                                   // setting default value
  }
  
  // Initialization of the serial link
  Serial.begin(115200);

  // Initialization of the analog input pins
  //default param relay pétanque
  pinMode(2, OUTPUT);
  digitalWrite(2, HIGH);

}

/********************* THIS FUNCTION IS CONTINUOUSLY EXECUTED (INFINITE LOOP) AFTER setup()  **************/
void loop() {
  
  readUserMessage(pinList);

  // READ digital pin Input *********************************
  for (int i = 0 ; i < digitalPinNumber ; i++) {          // 
    if (pinList[i] == 0) {                                // check if it's an Input Pin
      pinRead[i] = digitalRead(i);                        // if that is the case -> digitalRead(<pin n°>)
    }
  }
  for (int i = digitalPinNumber ; i < digitalPinNumber+analogicPinNumber ; i++) {          // 
    if (pinList[i] == 0) {                                // check if it's an Input Pin
      pinRead[i] = analogRead(i);                        // if that is the case -> analogRead(<pin n°>)
    }
  }
  for (int i = 0 ; i < digitalPinNumber ; i++) {          // check if a new digital value as been reading
    if (lastPinRead[i] != pinRead[i]) {
      flagDigitalPinRead = true;
    }
    else {

    }
  } 
  for (int i = digitalPinNumber ; i < digitalPinNumber+analogicPinNumber ; i++) {   // check if a new analogic value as been reading
    if (lastPinRead[i] != pinRead[i]) {
      flagAnalogicPinRead = true;
    }
    else {
      
    }
  }
  
  // PUBLISH digital pin Input *********************************
  if (flagDigitalPinRead) {
    Serial.print("d");
    for (int i = 0 ; i < digitalPinNumber ; i = i + 1) {
      if (lastPinRead[i] != pinRead[i]) {
        Serial.print(" ");
        Serial.print(i);
        Serial.print(" ");
        Serial.print(pinRead[i]);
        lastPinRead[i] = pinRead[i];
      }
    }
    Serial.println("");
    flagDigitalPinRead = false;
  }
  if (flagAnalogicPinRead) {
    Serial.print("a");
    for (int i = digitalPinNumber ; i < digitalPinNumber+analogicPinNumber ; i = i + 1) {
      if (lastPinRead[i] != pinRead[i]) {
        Serial.print(" ");
        Serial.print(i);
        Serial.print(" ");
        Serial.print(pinRead[i]);
        lastPinRead[i] = pinRead[i];
      }
    }
    Serial.println("");
    flagAnalogicPinRead = false;
  }
  
  //delay(1000);
}
