import serial
import cv2
import numpy as np
import imutils
import threading
import time,copy

# configure the serial connections (the parameters differs on the device you are connecting to)
'''ser = serial.Serial(
    port='COM5',
    baudrate=9600)

ser.isOpen()'''
#opencv code starts
class myThread (threading.Thread):
     def __init__(self, threadID, x, y):
         threading.Thread.__init__(self)
         self.threadID = threadID
         self.x=x
         self.y=y
     def run(self):
         print "Starting "
         colortester(self.threadID, self.x, self.y)
         print "Exiting "

global_color = "white"
input='w'

def colortester(clone2,x,y):
    while(1):
        time.sleep(2)
        upper_green=np.array([100,220,255])
        lower_green=np.array([60,110,110])
        
        imag=clone2[15:17,15:17]
        imag=cv2.cvtColor(imag,cv2.COLOR_BGR2HSV)
        tempnum=0
        
        mask=cv2.inRange(imag,lower_green,upper_green)
        
        if(mask[1][1]!=0):
            tempnum=tempnum+1
        if(tempnum==0):
            global_color = "red"
            input='r'
            #ser.write(input)
            
        elif(tempnum==1):
            global_color = "green"
            input='g'
            #ser.write(input)
          
            

img=cv2.imread("capture.png",1)
img=cv2.resize(img,(720,720))
clone11 = copy.copy(img)
clone = copy.copy(img)
clone4 = copy.copy(img)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel=np.ones((3,3),np.uint8)

img  = cv2.dilate(img,kernel,iterations = 3)
img=cv2.bilateralFilter(img,9,75,75)
img = cv2.medianBlur(img,5)
clone  = cv2.dilate(clone,kernel,iterations = 3)
clone=cv2.bilateralFilter(clone,9,75,75)
clone = cv2.medianBlur(clone,5)
clone2=copy.copy(clone)
img1 = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)[1]
cntsc = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntsc = cntsc[0] if imutils.is_cv2() else cntsc[1]
#cv2.drawContours(img1, contours, -1, (0,255,0), 3)
temp=[]
for c in cntsc:
        M = cv2.moments(c)
        if M["m00"]!=0:
            cX=int(M["m10"] / M["m00"])
            cY=int(M["m01"] / M["m00"])
            #print(cX,cY)

            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, 0.04*peri, True)
            if( len(approx) ==4 ):
                (x,y,w,h) = cv2.boundingRect(approx)
                ar = w/float(h)
                temp1=[peri,cX,cY,ar]
                temp.append(temp1)

            # cv2.drawContours(img1, [c], -1, (0, 255, 0), 2)
            # cv2.circle(img1, (cX, cY), 7, (0, 0, 255), -1)
            # cv2.putText(img1, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.imshow("res",img1)
            # cv2.waitKey(0)

temp = sorted(temp)
temp.reverse()

clone3=clone[temp[2][2]-int(temp[2][0]//8):temp[2][2]+int(temp[2][0]//8),temp[2][1]-int(temp[2][0]//8):temp[2][1]+int(temp[2][0]//8)]
thread1 = myThread(clone3, temp[2][1], temp[2][2])
thread1.start()
hsv = cv2.cvtColor(clone2,cv2.COLOR_BGR2HSV)
#getting center of bot
#lower_blue = np.array([110,50,50])
lower_blue = np.array([100,100,100])
upper_blue = np.array([150,255,255])
maskbl = cv2.inRange(hsv, lower_blue, upper_blue)
resbl = cv2.bitwise_and(clone2,clone2, mask= maskbl)
#cv2.imshow("qwer",resbl)
#cv2.waitKey(0)
lower_yellow = np.array([20,100,100])
upper_yellow = np.array([30,255,255])
maskyl = cv2.inRange(hsv, lower_yellow, upper_yellow)
resyl = cv2.bitwise_and(clone2,clone2,mask= maskyl)
#finding the ceter of blue of bot
cnts1 = cv2.findContours(maskbl, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]


# loop over the contours
for c in cnts1:
     # compute the center of the contour
     Mb = cv2.moments(c)
     if Mb["m00"]!=0:
         cXb= int(Mb["m10"] / Mb["m00"])
         cYb = int(Mb["m01"] / Mb["m00"])
         print cXb
         print cYb

         #cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
         #cv2.circle(clone, (cXb, cYb), 7, (255, 255, 255), -1)
         #cv2.putText(clone, "center", (cXb - 20, cYb - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
         #cv2.imshow("clone",clone)
         #cv2.waitKey(0)
        # show the imag
#finding the center of the yellow of bot
'''clone11=cv2.cvtColor(clone2,cv2.COLOR_BGR2GRAY)   
rt, clone11 = cv2.threshold(maskyl, 127, 255,0)

img2cnt,hierarchy = cv2.findContours(clone11,2,1)
i2cnt = img2cnt[0]
cnts2 = cv2.findContours(maskyl, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


# loop over the contours
area=[]
i=0
cnt2 = cnts2[0]

for c in cnts2:
    ret = cv2.matchShapes(i2cnt,c,1,0)
    if(ret<=0.5):

     # compute the center of the contour
        My = cv2.moments(c)
        if My["m00"]!=0:
            cXy= int(My["m10"] / My["m00"])
            cYy = int(My["m01"] / My["m00"])
            cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
            cv2.circle(clone, (cXy, cYy), 7, (255, 255, 255), -1)
            cv2.putText(clone, "center", (cXy - 20, cYy - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow("clone",clone)
            cv2.waitKey(0)'''
#defining roi for layout
print temp

# roi= clone[temp[1][2]-25:temp[1][2]+25,temp[1][3]-15:temp[1][3]+15]
# cv2.imshow("roi",roi)
#
# cv2.waitKey(0)


cv2.destroyAllWindows()

#print 'Enter your commands below.\r\nInsert "exit" to leave the application.'
#print (input)
#while 1 :
    # get keyboard input
        # Python 3 users
        # input = input(">> ")
#    if input == 'exit':
#        ser.close()
#        exit()
#    else:
        # send the character to the device
        # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
#        ser.write(input +'/r/n' )
#        out = ' '
         # let's wait one second before reading output (let's give device time to answer)
        #while ser.inWaiting() > 0:
         #   out += ser.read(1)

        #if out != '':
        #   print ">>" + out
