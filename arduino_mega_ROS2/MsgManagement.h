/*
  MsgManagement.h - Library Dealing with Messages.
  Created by Guillaume Morel, Dec 8, 2025.
  Released into the public domain.
*/
//#ifndef MsgManagement_h

//#define MsgManagement_h
#define NB_MAX_MSG_ARGS 10

#if !defined(FALSE)
  #define FALSE 0
  #define TRUE 1
#endif

class Message
{
public:
  Message();
  void ReadSerial();
  void Get(char* Msgtype, int* Argnum, float* Args);
private:
  char _userMessageType; 
  float _userNumericalArg[NB_MAX_MSG_ARGS+1];  // up to 10 arguments, separated by a space; The first element is the number of received arguments
  float _argSign;
  int _nbArgsSoFar = 0;
  int _msgStarted = FALSE;
  int _msgRecieved = FALSE;
  String _inString = "";
  String _inString2 = "";
  int _getString2 = FALSE;
};

//#endif
