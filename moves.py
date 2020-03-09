import time
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from skimage.color import hsv2rgb
import cv2

"""
Updates position of all Zwierzaks.
They generally like to cluster, if they are suitably far away
"""
def updatePosition(zwk,zwks,x_home,y_home):
    zwk.x_prev = zwk.x_pos
    zwk.y_prev = zwk.y_pos
    fardist=30.
    dhome_x=zwk.x_prev - x_home
    dhome_y=zwk.x_prev - x_home
    homing_x = 0.01*dhome_x
    homing_y = 0.01*dhome_y
    pl_x=max(0.33+homing_x,0)
    pl_y=max(0.33+homing_y,0)
    dx = np.random.choice([-1,0,1], p=[pl_x, 0.34, 0.66-pl_x])
    dy = np.random.choice([-1,0,1], p=[pl_y, 0.34, 0.66-pl_y])
    # dx, dy = np.round(2*np.random.rand(2,)-1).astype(int)
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

    for other_zwk in zwks_list:
        if other_zwk.id == zwk.id:
            continue
        if zwk.x_pos == other_zwk.x_pos and zwk.y_pos == other_zwk.y_pos:
            zwk.x_pos = zwk.x_prev
            zwk.y_pos = zwk.y_prev
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
    cv2.namedWindow('plane', cv2.WINDOW_GUI_EXPANDED)
    cv2.moveWindow('plane', 200,200)
    side = 100
    ch = 3 #RGB image displays output
    borders = Borders(0,0,side,side)
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

    for it in range(1000):
        for alf in alfs:
            alf = updatePosition(alf,alfs,home[0],home[1])
            alf = handleColisions(alf,borders,alfs)
            cc = hsv_plane[alf.x_pos,alf.y_pos]
            hsv_plane[alf.x_pos,alf.y_pos]=(alf.hsv[0], alf.hsv[1],min(cc[2]+10,255))
            # print(plane[alf.x_pos,alf.y_pos])

        plane = np.zeros((side,side,ch),np.uint8)
        #change the whole image from hsv to rgb
        for i in range(plane.shape[0]):
            for j in range(plane.shape[1]):
                plane[j,i] = colorsys.hsv_to_rgb(hsv_plane[j,i][0],hsv_plane[j,i][1],hsv_plane[j,i][2])
        #planergb = hsv2rgb(plane)

        plane_cur = plane.copy()
        for alf in alfs:
            plane_cur[alf.x_pos,alf.y_pos]=colorsys.hsv_to_rgb(alf.hsv[0], alf.hsv[1],255)

        cv2.imshow("plane",plane_cur)
        cv2.waitKey(20)
#        time.sleep(1)

    plt.savefig('test.png')
    plt.show()

if __name__ == '__main__':
    main()
