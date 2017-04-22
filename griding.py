__author__ = 'Shubham'


import cv2
import numpy as np
import heapq
import imutils
import serial
import threading
import time,copy
import cmath
import math

refPt=[]
cropping = False


ic=[]
refPt1=[]
cropping1 = False
colorsp=[]

def on_mouse_click (event,x, y,flags,param):
#    key =cv2.waitKey(0) & 0xFF
    if event==cv2.EVENT_LBUTTONDOWN:
        ic.append((x,y))
        cropping1=True

    elif event==cv2.EVENT_LBUTTONUP:
        refPt1.append((x,y))
        cropping1=False

#        cv2.rectangle(image, ic[0], refPt1[0], (0, 255, 0), 2)
#        cv2.imshow("image", image)

        tr = ( (refPt1[0][0]+ic[0][0])/2,(refPt1[0][1]+ic[0][1])/2 )
        ic.append(tr)
        ar1 = hsv[int(ic[0][1])-2:int(ic[0][1])+2,int(ic[1][0])-2:int(ic[1][0])+2]
        ar2 = np.average(ar1, axis=0)
        ar3 = np.average(ar2, axis=0)
        colorsp.append(ar3)
        cv2.rectangle(hsv,ic[0],refPt1[0],(0,255,0),2)

        ic.pop()
        ic.pop()
        refPt1.pop()


def click_n_crop(event,x,y,flags,param):
    global refPt,cropping

    if event==cv2.EVENT_LBUTTONDOWN:
        refPt=[(x,y)]
        cropping=True

    elif event==cv2.EVENT_LBUTTONUP:
        refPt.append((x,y))
        cropping=False

        cv2.rectangle(image,refPt[0],refPt[1],(0,255,0),2)
        cv2.imshow("image",image)


def cor2theta(ipt,fpt,bmp,ymp):
     vect1=(fpt[0]-ipt[0],fpt[1]-ipt[1])
     vect2=(bmp[0]-ymp[0],bmp[1]-ymp[1])
     thet=math.acos((vect1[0]*vect2[0]+vect1[1]*vect2[1])/(((vect1[0] ** 2 + vect1[1] ** 2) ** 0.5)*((vect2[0] ** 2 + vect2[1] ** 2) ** 0.5)))
     theta=57.2727*thet
     return theta

ser = serial.Serial(
    port='COM7',
    baudrate=9600)

ser.isOpen()


class myThread (threading.Thread):
     def __init__(self, threadID, x, y):
         threading.Thread.__init__(self)
         self.threadID = threadID
         self.x=x
         self.y=y
     def run(self):
         print ("Starting ")
         colortester(self.threadID, self.x, self.y)
         print ("Exiting ")

def dijkstra(adj, costs, s, t):
    ''' Return predecessors and min distance if there exists a shortest path 
        from s to t; Otherwise, return None '''
    Q = []     # priority queue of items; note item is mutable.
    d = {s: 0} # vertex -> minimal distance
    Qd = {}    # vertex -> [d[v], parent_v, v]
    p = {}     # predecessor
    visited_set = set([s])

    for v in adj.get(s, []):
        d[v] = costs[s, v]
        item = [d[v], s, v]
        heapq.heappush(Q, item)
        Qd[v] = item

    while Q:
#        print (Q)
        cost, parent, u = heapq.heappop(Q)
        if u not in visited_set:
#            print ('visit:', u)
            p[u]= parent
            visited_set.add(u)
            if u == t:
                return p, d[u]
            for v in adj.get(u, []):
                if d.get(v):
                    if d[v] > costs[u, v] + d[u]:
                        d[v] =  costs[u, v] + d[u]
                        Qd[v][0] = d[v]    # decrease key
                        Qd[v][1] = u       # update predecessor
                        heapq._siftdown(Q, 0, Q.index(Qd[v]))
                else:
                    d[v] = costs[u, v] + d[u]
                    item = [d[v], u, v]
                    heapq.heappush(Q, item)
                    Qd[v] = item

    return None

def make_undirected(cost):
    ucost = {}
    for k, w in cost.items():
        ucost[k] = w
        ucost[(k[1],k[0])] = w
    return ucost

global_color = "green"
input='w'
def colortester(clone2,x,y):
     while(1):
        time.sleep(2)
        upper_green=np.array([colorsp[1][0]+10,255,255])
        lower_green=np.array([colorsp[1][0]-10,100,100])
        
        imag=clone2[15:17,15:17]
        imag=cv2.cvtColor(imag,cv2.COLOR_BGR2HSV)
        tempnum=0
        
        mask=cv2.inRange(imag,lower_green,upper_green)
        
        if(mask[1][1]!=0):
            tempnum=tempnum+1
        if(tempnum==0):
            global_color = "red"
            input='r'
            ser.write(input)
            
        elif(tempnum==1):
            global_color = "green"
            input='g'
            ser.write(input)

     
adj = {}
cost = {}
for i in range(1,65):
   if(i<=8):
       if(i==1):
           adj[i] = [2,9]
           cost[(i,2)] = 1
           cost[(i,9)] = 1
       elif(i==8):
           adj[i] = [7,16]
           cost[(7,i)] = 1
           cost[(i,16)] = 1
       else:
           adj[i] = [i-1,i+1,i+8]
           cost[(i-1,i)] = 1
           cost[(i,i+1)] = 1
           cost[(i,i+8)] = 1
   elif(i>=57):
       if(i==57):
           adj[i] = [49,58]
           cost[(i,i-8)] = 1
           cost[(i,i+1)] = 1
       elif(i==64):
           adj[i] = [56,63]
           cost[(i-1,i)] = 1
           cost[(i-8,i)] = 1
       else:
           adj[i] = [i-8,i-1,i+1]
           cost[(i-1,i)] = 1
           cost[(i,i+1)] = 1
           cost[(i-8,i)] = 1
   else:
       if(i%8 == 1):
           adj[i] = [i-8,i+1,i+8]
           cost[(i,i+1)] = 1
           cost[(i-8,i)] = 1
           cost[(i,i+8)] = 1
       elif(i%8 == 0):
           adj[i] = [i-8,i-1,i+8]
           cost[(i-1,i)] = 1
           cost[(i-8,i)] = 1
           cost[(i,i+8)] = 1
       else:
           adj[i] = [i-8,i-1,i+1,i+8]
           cost[(i-1,i)] = 1
           cost[(i,i+1)] = 1
           cost[(i-8,i)] = 1
           cost[(i,i+8)] = 1

cost = make_undirected(cost)

def find_grid(x,y):
     return (1+(x//90))+8*(y//90)

cap = cv2.VideoCapture(1)
cap.set(3,720)
cap.set(4,720)
roi = cap.read()
count1 = 0 
while(True):
    
    # Capture frame-by-frame
    if count1 == 0:
        ret, image = cap.read()
        clone=image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image",click_n_crop)

        while True:
            cv2.imshow("image",image)
            key=cv2.waitKey(1) & 0xFF
            if key==ord("r"):
                image = clone.copy()

            elif key==ord("w"):
                break
       
        if len(refPt)==2:
            roi=clone[refPt[0][1]:refPt[1][1],refPt[0][0]:refPt[1][0]]
            
        count1 = count1+1
    elif count1 == 1:
        while True:
            ret, img = cap.read()
            img = img[refPt[0][1]:refPt[1][1],refPt[0][0]:refPt[1][0]]
            cv2.imshow("cropped", img)          
            img=cv2.resize(img,(720,720))
            clone_area = copy.copy(img)
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
            clone7=copy.copy(img1)
            hsv = cv2.cvtColor(clone2,cv2.COLOR_BGR2HSV)

            cv2.imshow("final", clone)
            cv2.waitkey(0)
              
            if cv2.waitKey(0) & 0xFF==ord('q'):
                cv2.destroyAllWindows()
                break

            cv2.imshow("frame34", hsv)
            cv2.waitKey(0)
            cv2.setMouseCallback("frame34", on_mouse_click)
        count1 = count1 + 1
    else:
          ret, img = cap.read()
          img = img[refPt[0][1]:refPt[1][1],refPt[0][0]:refPt[1][0]]
          cv2.imshow("cropped", img)          
          img=cv2.resize(img,(720,720))
          clone_area = copy.copy(img)
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
          clone7=copy.copy(img1)
          hsv = cv2.cvtColor(clone2,cv2.COLOR_BGR2HSV)
          cntsc = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          cntsc = cntsc[0] if imutils.is_cv2() else cntsc[1]
          #cv2.drawContours(img1, contours, -1, (0,255,0), 3)
          temp=[]
          block=[]
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
                          if (ar >= 0.95 and ar <= 1.05):
                              block.append(temp1)

                      #cv2.drawContours(img1, [c], -1, (0, 255, 0), 2)
                      #cv2.circle(img1, (cX, cY), 7, (0, 0, 255), -1)
                      #cv2.putText(img1, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                      #cv2.imshow("res",img1)
                      #cv2.waitKey(0)

          temp = sorted(temp)
          temp.reverse()


          #q(x,y,w,h) = cv2.boundingRect(approx)
          #print temp
          #cv2.imshow("res",img1)
          #cv2.waitKey(0)
          #cv2.destroyAllWindows()
          # for i in range (len(temp)):
          #     cv2.circle(clone, (temp[i][1], temp[i][2]), 7, (0, 0, 255), -1)
          #     cv2.putText(clone, "center", (temp[i][1] - 20, temp[i][2] - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
          #     cv2.imshow("rrrr",clone)
          #     cv2.waitKey(0)

          print temp
          clone3=clone[temp[2][2]-int(temp[2][0]//8):temp[2][2]+int(temp[2][0]//8),temp[2][1]-int(temp[2][0]//8):temp[2][1]+int(temp[2][0]//8)]
          #clone33 = clone[temp[1][2]-((1.4*temp[1][0]))//16:temp[1][2]+(2*temp[1][0])//16,temp[1][1]-((3*temp[1][0])//16):temp[1][1]+((3*temp[1][0])//16)]
          thread1 = myThread(clone3, temp[2][1], temp[2][2])
          thread1.start()

          for i in range(2):
               index = find_grid(temp[i][1],temp[1][2])
               adj[i]=[]
               try:
                    adj[i+1]=0
                    adj[i-1]=0
               except:
                    print ("ho! ho! ho!")

          tep_x = (temp[1][0]//16)
          tep_y = (temp[1][0]//16)
          tx=[]
          ty=[]

          for i in range (-2,3,1):
              if((temp[1][1] + i*tep_x) < 720): tx.append(temp[1][1] + i*tep_x)
              else: tx.append(716)
          tx = tx*3

          for i in range (-1,2,1):
              for j in range (5):
                  if((temp[1][2] + i*tep_y)>0): ty.append(temp[1][2] + i*tep_y)
                  else: ty.append(4)

          collar = []
          # upper_red1=np.array([179,255,255])
          # lower_red1=np.array([160,100,100])
          # upper_red2=np.array([10,255,255])
          # lower_red2=np.array([0,100,100])
          upper_green=np.array([colorsp[1][0]+10,255,255])
          lower_green=np.array([colorsp[1][0]-10,100,100])
          green=[lower_green,upper_green,'green']
          upper_red1=np.array([colorsp[0][0]+10,255,255])
          lower_red1=np.array([colorsp[0][0]-10,100,100])
          red1=[lower_red1,upper_red1,'red']

          #upper_red2=np.array([10,255,255])
          #lower_red2=np.array([0,100,100])
          #red2=[lower_red2,upper_red2,'red']

          colors=[green,red1]

          for i in range(15):
              ar1 = clone[int(tx[i])-2:int(tx[i])+2,int(ty[i])-2:int(ty[i])+2]

              cv2.circle(clone, (int(tx[i]), int(ty[i])), 7, (0, 255, 0), -1)
              cv2.putText(clone, "center", (int(tx[i]) - 20, int(ty[i]) - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

          #    cv2.imshow("clone", clone)
          #    cv2.waitKey(0)
              ar2 = np.average(ar1, axis=0)
              ar3 = np.average(ar2, axis=0)
              tempnum=0
              for color in colors:
                  mask=cv2.inRange(ar1,color[0],color[1])
                  #cv2.imshow('mask',mask)
                  if(mask[1][1]!=0):
                      break
                  tempnum=tempnum+1
              if(tempnum==0):
                  collar.append("green")
              elif(tempnum==1 or tempnum==2):
                  collar.append("red")
              else:
                  collar.append("white")

          print (collar)

          while(1):
               

               #getting center of bot
               #lower_blue = np.array([110,50,50])
               lower_blue = np.array([colorsp[3][0]-10,100,100])
               upper_blue = np.array([colorsp[3][0]+10,255,255])
               maskbl = cv2.inRange(hsv, lower_blue, upper_blue)
               resbl = cv2.bitwise_and(clone2,clone2, mask= maskbl)
               #cv2.imshow("qwer",resbl)
               #cv2.waitKey(0)
               lower_yellow = np.array([colorsp[2][0]-10,100,100])
               upper_yellow = np.array([colorsp[2][0]+10,255,255])
               maskyl = cv2.inRange(hsv, lower_yellow, upper_yellow)
               resyl = cv2.bitwise_and(clone2,clone2,mask= maskyl)


               #getting blocks center on the arena
               #getting blocks center on the arena
               img7=cv2.cvtColor(clone3,cv2.COLOR_BGR2GRAY)
               led_img = cv2.threshold(img7, 100, 255, cv2.THRESH_BINARY)[1]
               #subtract
               h,w=led_img.shape[:2]
               clone7[temp[2][1]-(w)//2:temp[2][1]+(w)//2,temp[2][2]-(h)//2:temp[2][2]+(h)//2]-=255

               cl="white"
               haddi = []

               for i in range (len(block)):
                   upper_green=np.array([colorsp[1][0]+10,255,255])
                   lower_green=np.array([colorsp[1][0]-10,100,100])

                   imag=hsv[block[i][2]-5:block[i][2]+5,block[i][1]-5:block[i][1]+5]
                   tempnum=0
                   mask=cv2.inRange(imag,lower_green,upper_green)
                   if(mask[1][1]!=0):
                       tempnum=tempnum+1
                   if(tempnum==0):
                       cl = "red"
                   elif(tempnum==1):
                       cl = "green"
                   haddi.append(cl) 
          ##         cv2.circle(clone7, (block[i][1], block[i][2]), 7, (0, 0, 255), -1)
          ##         cv2.putText(clone7, cl, (block[i][1] - 20, block[i][2] - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
          ##         cv2.imshow("rrrr",clone7)
          ##         cv2.waitKey(0)
                
               #getting the


               #finding the ceter of blue of bot
               cnts1 = cv2.findContours(maskbl, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
               cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]
               cnts2 = cv2.findContours(maskyl, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
               cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]
               cXb = 0
               cYb = 0
               cXy = 0
               cYy = 0
               hudz=[]
               # loop over the contours
               for c in cnts1:
                    # compute the center of the contour
                    Mb = cv2.moments(c)
                    if Mb["m00"]!=0:
                        cX= int(Mb["m10"] / Mb["m00"])
                        cY = int(Mb["m01"] / Mb["m00"])
                        peri = cv2.arcLength(c,True)
                        approx = cv2.approxPolyDP(c, 0.04*peri, True)
                        if( len(approx) ==4 ):
                            (x,y,w,h) = cv2.boundingRect(approx)
                            ar = w/float(h)
                            temp1=[ar,cX,cY,peri]
                            hudz.append(temp1)
               hudz= sorted(hudz)
               hudz.reverse()
               cXb=hudz[0][1]
               cYb=hudz[0][2]

                        #cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
                        #cv2.circle(clone, (cXb, cYb), 7, (255, 255, 255), -1)
                        #cv2.putText(clone, "center", (cXb - 20, cYb - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        #cv2.imshow("clone",clone)
                        #cv2.waitKey(0)
                       # show the imag
                       
               #finding the center of the yellow of bot
               hadz=[]
               for c in cnts2:
                    M = cv2.moments(c)
                    if M["m00"]!=0:
                        cX=int(M["m10"] / M["m00"])
                        cY=int(M["m01"] / M["m00"])
                        peri = cv2.arcLength(c,True)
                        approx = cv2.approxPolyDP(c, 0.04*peri, True)
                        if( len(approx) ==4 ):
                            (x,y,w,h) = cv2.boundingRect(approx)
                            ar = w/float(h)
                            temp1=[ar,cX,cY,peri]
                            hadz.append(temp1)
               hadz= sorted(hadz)
               hadz.reverse()
               cXy=hadz[0][1]
               cYy=hadz[0][2]

               print(cXy,cYy)


               #defining roi for layout

               # roi= clone[temp[1][2]-25:temp[1][2]+25,temp[1][3]-15:temp[1][3]+15]
               # cv2.imshow("roi",roi)
               # cv2.waitKey(0)

               centerbot_x=(cXb+cXy)//2
               centerbot_y=(cYb+cYy)//2
               index_bot = find_grid(centerbot_x, centerbot_y)

               adj2 = copy.copy(adj)
                
               s = index_bot
               pathX = []
               for i in range (len(haddi)):
                    if(haddi[i] == global_color):
                         vidit = []
                         t = find_grid(block[i][1], block[i][2])
                         j=0
                         for j in range (len(haddi)):
                              if(j!=i):
                                   indice = find_grid(block[j][1],block[j][2])
                                   vidit.append(adj[indice])
                                   adj[indice]=[]
                         predecessors, min_cost = dijkstra(adj, cost, s, t)
                         c = t
                         path = [c]

                         while predecessors.get(c):
                            path.insert(0, predecessors[c])
                            c = predecessors[c]
                         pa =[min_cost, path]
                         pathX.append(pa)

                         adj = adj2
                         
               pathX = sorted(pathX)
               print (pathX[0][1])
               jk = pathX[0][1]

               ipt = (90*(jk[0]//8), 90*(jk[0]%8))
               fpt = (90*(jk[1]//8), 90*(jk[1]%8))
               bc = (cXb, cYb)
               yc = (cXy, cYy)

               theta=cor2theta(ipt,fpt,bc,yc)
               if (theta in (0,20) ):
                  ser.write('F')
               elif(theta in (20,181)):
                  ser.write('L')
              # elif(theta in (-150,-10)):
                  #ser.write('R')
               time.sleep(1)
               
          cv2.destroyAllWindows()
