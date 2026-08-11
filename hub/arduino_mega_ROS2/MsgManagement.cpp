/*
  MsgManagement.cpp 
  Created by Guillaume Morel, Dec 8, 2025.
*/
#include <Arduino.h>
#include "MsgManagement.h"

Message::Message()
{
  int i;

  _userMessageType='0';  // The character defining the function
  for(i=0;i<=NB_MAX_MSG_ARGS;i++) {
    _userNumericalArg[i] = 0.0;
  }
  _argSign = 1.0;
  _nbArgsSoFar = 0;
  _msgStarted = FALSE;
  _msgRecieved = FALSE;
  _inString = "";
  _inString2 = "";
  _getString2 = FALSE;
}

void Message::ReadSerial() {
  int i;
  char inByte;
  unsigned int puissance;

  // Read serial input:
  while (Serial.available() > 0) {
    if (_msgStarted == FALSE) { // Initializes as the message starts
      for (i = 0; i < NB_MAX_MSG_ARGS+1; i++) {
        _userNumericalArg[i] = 0.;
      }
      _userMessageType = (char)(Serial.read()); // The first char is the message type
      _msgStarted = TRUE;
      _nbArgsSoFar = 0;
      _argSign = 1.0;
    } 
    else { // The message has started. The first char has been stored
      inByte = Serial.read();
      if (inByte == ' ') { // separator between 2 arguments
        if (_nbArgsSoFar == 0) { // first separator: do nothing
          _nbArgsSoFar = 1;
        } 
        else { // separator between two numbers
          _userNumericalArg[0]++; // The first element of this table is the number of numerical arguments
          _userNumericalArg[_nbArgsSoFar] = _argSign*((float)(_inString.toInt()));
          puissance = _inString2.length();
          _userNumericalArg[_nbArgsSoFar] += _argSign*( (float)(_inString2.toInt())) / pow(10.0, ((float)(puissance)) );
          _nbArgsSoFar++;
          _inString = "";
          _inString2 = "";
          _getString2 = FALSE;
          _argSign=1.0;
        }
      }
      if (inByte == '.') {
        _getString2=TRUE; // get ready for numbers after "."
      }
      if (inByte == '-') {
        _argSign=-1.0; // it is a negative number
      }
      else {
        if (isDigit(inByte)) {
          if(_getString2) _inString2 += (char)inByte;
          else _inString += (char)inByte;
        }
      }
      if (inByte == '\n') { // End of the message
        _userNumericalArg[0]++;
        _userNumericalArg[_nbArgsSoFar] = _argSign*((float)(_inString.toInt()));
        puissance = _inString2.length();
        _userNumericalArg[_nbArgsSoFar] += _argSign * ( (float)(_inString2.toInt())) / pow(10.0, ((float)(puissance)) );
        _inString = "";
        _inString2 = "";
        _getString2 = FALSE;
        _msgStarted = FALSE;
        _msgRecieved = TRUE;
      }
    }
  }
}


void Message::Get(char* msgType, int* argNumber, float* Args){
  int i;
  if (_msgRecieved) {
    *msgType = _userMessageType;
    *argNumber = _nbArgsSoFar;
    for(i=0;i<=NB_MAX_MSG_ARGS;i++) {
      Args[i] = _userNumericalArg[i];
    }
    _msgRecieved = FALSE; // Ready to read next message
  }
  else{
    *msgType='0'; // Which means : no message.
  }
}
