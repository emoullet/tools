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

// FOR ANALOG INPUT

// FOR PRINTING THROUGH THE SERIAL PORT
int counterForPrinting = 0;
int printingPeriodicity = 1;
int printIsOn = FALSE;
int matlabOuputIsOn = FALSE;
int nbMatlabOutputPoints = 0;
int nbMatlabOutputPointsSoFar = 0;

void readUserMessage(void){
  char userCommand;
  int nbOfUserArgs;
  float userArgs[NB_MAX_MSG_ARGS+1];

  MSG.ReadSerial(); // Gets entries from the serial port to build a message
  MSG.Get(&userCommand, &nbOfUserArgs, &(userArgs[0])); // Gets the message when ready (otherwise, userCommand=0)
  if (userCommand!='0') { // If a message has arrived, do for interpretation
    switch(userCommand) {
      case('D'): // PRINT CONTINUOUSLY FOR PLOTTER
        for (int i=1; i<nbOfUserArgs+1; i=i+2){
          /*
          Serial.print("pin[");
          Serial.print(userArgs[i]);
          Serial.print("] = ");
          Serial.println(userArgs[i+1]);
          */
          digitalWrite(userArgs[i],userArgs[i+1]);
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

  // Initialization of the serial link
  Serial.begin(115200);

  // Initialization of the analog input pins
  int digitalPinList[] = {2,3,4,5,6,7,8,9,10,11,12,13};
  
  for (int i = 0; i<sizeof(digitalPinList); i++){
    pinMode(digitalPinList[i], OUTPUT);
    digitalWrite(digitalPinList[i], HIGH);
  }
}

/********************* THIS FUNCTION IS CONTINUOUSLY EXECUTED (INFINITE LOOP) AFTER setup()  **************/
void loop() {

  int i;

  readUserMessage();
  
}
