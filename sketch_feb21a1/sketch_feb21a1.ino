#include <AFMotor.h>
AF_DCMotor ledR(1);
AF_DCMotor ledG(2);
AF_DCMotor motorR(4);
AF_DCMotor motorL(3);
void setup() {
    Serial.begin(9600);
  motorL.setSpeed(200);
   motorR.setSpeed(200);
  ledR.setSpeed(1);
  ledG.setSpeed(1);
  motorL.run(RELEASE);
  motorR.run(RELEASE);
  ledR.run(RELEASE);
 ledG.run(RELEASE);
  // put your setup code here, to run once:

}

void loop() {
  char input=Serial.read();

 if(input=='r'){
     ledR.run(FORWARD);
     delay(2000);
     ledR.run(RELEASE);
 }
 if(input=='g')
  {
    ledG.run(FORWARD);
    delay(2000);
    ledG.run(RELEASE);
  }
 
 if(input=='L')
 {
 motorR.run(FORWARD);
 motorL.run(BACKWARD);
 }
 else if (input=='R')
 {
  motorL.run(FORWARD);
 motorR.run(BACKWARD);
  }
 else if(input=='F')
  {motorR.run(FORWARD);
   motorL.run(FORWARD);}
  else if(input=='B')
  {
   motorR.run(BACKWARD);
   motorL.run(BACKWARD); 
   }
  else if(input=='S')
  {
   motorR.run(RELEASE);
   motorL.run(RELEASE); 
   }
    
//
// motorL.run(RELEASE);
// motorR.run(RELEASE);

  // put your main code here, to run repeatedly:

}
