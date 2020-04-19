import math
import time
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from skimage.color import hsv2rgb
import cv2

def updateTrace(hsv_plane,alf):
    cc = hsv_plane[alf.x_pos,alf.y_pos]
    cv2.ellipse(hsv_plane,(alf.x_pos,alf.y_pos),(alf.islong,alf.iswide),alf.angle,0,360,(alf.hsv[0], alf.hsv[1],min(cc[2]+10,255)),-1)
    return hsv_plane

def showTrace(hsv_plane,alf,side,ch):
    hdplane = np.zeros((side,side,ch),np.uint8)
    #change the whole image from hsv to rgb
    for i in range(hdplane.shape[0]):
        for j in range(hdplane.shape[1]):
            hdplane[j,i] = colorsys.hsv_to_rgb(hsv_plane[j,i][0],hsv_plane[j,i][1],hsv_plane[j,i][2])
    return hdplane


def getRoI(zwk):
            sinzwk = zwk.islong * math.sin(np.pi * zwk.angle / 180)
            coszwk = zwk.islong * math.cos(np.pi * zwk.angle / 180)
            sinzwkw = zwk.iswide * math.sin(np.pi * zwk.angle / 180)
            coszwkw = zwk.iswide * math.cos(np.pi * zwk.angle / 180)
            tail = (int(zwk.x_pos-coszwk),int(zwk.y_pos-sinzwk))
            head = (int(zwk.x_pos+coszwk),int(zwk.y_pos+sinzwk))

            c1 = (int(head[0]+sinzwkw),int(head[1]-coszwkw))
            c2 = (int(head[0]-sinzwkw),int(head[1]+coszwkw))
            c3 = (int(tail[0]-sinzwkw),int(tail[1]+coszwkw))
            c4 = (int(tail[0]+sinzwkw),int(tail[1]-coszwkw))
            offset = 2 #HACK hardcoded offset
            # mins of all
            topleft = (np.min([c1[0],c2[0],c3[0],c4[0]])-offset,np.min([c1[1],c2[1],c3[1],c4[1]])-offset)
            # maxes of all
            bottomright = (np.max([c1[0],c2[0],c3[0],c4[0]])+offset,np.max([c1[1],c2[1],c3[1],c4[1]])+offset)
            return (head, topleft, bottomright)


def normToOne(vallist):
    valsum = sum(vallist)
    return [x/valsum for x in vallist]

"""
Updates position of all Zwierzaks.
They generally like to cluster, if they are suitably far away
"""
def updatePosition(zwk,zwks,x_home,y_home):
    rng = np.random.default_rng()
    newturn = rng.normal(0,10,1)
    zwk.x_prev = zwk.x_pos
    zwk.y_prev = zwk.y_pos
    zwk.angle = zwk.angle + newturn
    dx = int(10 * np.cos(np.pi * zwk.angle/180))
    dy = int(10 * np.sin(np.pi * zwk.angle/180))
    zwk.x_pos = zwk.x_prev+dx
    zwk.y_pos = zwk.y_prev+dy

    return zwk

"""
Handle colisions:
 - boundry conditions? Let's make it reflective, but simulate so that animals keep off the long grass.
 - collisions with other animals: This is going to be async, so if animal wants to move to other position, it will not make this movement. I need occupancy grid for that.
"""
def handleColisions(zwk, borders, zwks_list):
    rside = borders.x_max
    lside = borders.x_min # ASSUME IT IS SQUARE
    zwk.x_pos = max(min(zwk.x_pos,rside-1),lside)
    zwk.y_pos = max(min(zwk.y_pos,rside-1),lside)

    return zwk


class Zwierzak:
    def __init__(self, zwkid, x_init,y_init, hue=0, sat=1):
        self.id = zwkid
        self.x_init=x_init
        self.y_init=y_init
        self.x_pos=x_init
        self.y_pos=y_init
        self.x_prev=x_init
        self.y_prev=y_init
        self.hsv=(hue,sat,0) # initialise as a dim value
        self.angle = 0
        self.islong = 30 #half of width and height as opencv ellipses measurements defined
        self.iswide = 10

"""
This class shows any natural and unnatural boundaries for the environment
"""
class Borders:
    x_min=0
    y_min=0
    x_max=100
    y_max=100
    def __init__(self, xmi,ymi,xma,yma): #isn't that a dumb constructor syntax, heh?
        self.x_min=xmi
        self.y_min=ymi
        self.x_max=xma
        self.y_max=yma

"""
A little loading-time test of current animal setup
"""
def main():
    #cv2.namedWindow('HDplane', cv2.WINDOW_GUI_EXPANDED)
    # cv2.moveWindow('HDplane', 200,200)
    side = 500
    ch = 3 #RGB image displays output
    borders = Borders(0,0,side,side)

    #keeps the information of previous occupancy
    hdplane = np.zeros((side,side,3),np.uint8) 
    hsv_plane = np.zeros((side,side,ch),float) #HSV values are 0.0:1.0 hue, 0.0:1.0 saturation, 0:255 (int) value

    #np.random.seed(0)
    #x_init, y_init = map(int,map(round,np.random.uniform(0, side-1, 2)))
    x_init, y_init = [side//2,side//2]
    home = [x_init, y_init]

    alf0 = Zwierzak('alf0',x_init,y_init, hue=0,sat=1)
    alf1 = Zwierzak('alf1',0,0,hue=0.1,sat=1)
    alf2 = Zwierzak('alf2',0,0,hue=0.2,sat=1)
    alf3 = Zwierzak('alf3',x_init,y_init,hue=0.3,sat=1)
    alf4 = Zwierzak('alf4',0,0,hue=0.4,sat=1)
    alfs = [alf1,alf2,alf3,alf4,alf0]
    #alfs = [alf0]

    #centre, axes W, H, angle, startagnel, endangle, colour, thinkcness
    # cv2.ellipse(hdplane,(100,100),(50,10),30,0,360,(255,255,0),-1)

    for it in range(1000):
        for alf in alfs:
            alf = updatePosition(alf,alfs,home[0],home[1])
            alf = handleColisions(alf,borders,alfs)
            hsv_plane = updateTrace(hsv_plane,alf)

        plane_cur = hdplane.copy()
        for alf in alfs:
            cv2.ellipse(plane_cur,(alf.x_pos,alf.y_pos),(alf.islong,alf.iswide),alf.angle,0,360,colorsys.hsv_to_rgb(alf.hsv[0], alf.hsv[1],255),-1)
            (head, r1,r2) = getRoI(alf)
            cv2.circle(plane_cur,head,3,(0,255,255))
            cv2.rectangle(plane_cur,r1,r2,(0,0,255),2)
        cv2.imshow("hdplane",plane_cur)
        # cv2.waitKey(20)
        key = cv2.waitKey(20)
        if key==ord('q'):
            break

    hdplane = showTrace(hsv_plane,alf,side,ch)
    cv2.imshow("hdplane",hdplane)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()