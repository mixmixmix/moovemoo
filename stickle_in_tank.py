import moomodel
import math
import yaml, os, argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from skimage.color import hsv2rgb
import cv2
from collections import deque
from scipy.special import softmax


def getRoI(zwk):
            sinzwk = zwk.islong * math.sin(np.pi * zwk.angle[0] / 180)
            coszwk = zwk.islong * math.cos(np.pi * zwk.angle[0] / 180)
            sinzwkw = zwk.iswide * math.sin(np.pi * zwk.angle[0] / 180)
            coszwkw = zwk.iswide * math.cos(np.pi * zwk.angle[0] / 180)
            tail = (int(zwk.pos[0]-coszwk),int(zwk.pos[1]-sinzwk))
            head = (int(zwk.pos[0]+coszwk),int(zwk.pos[1]+sinzwk))

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

"""
Updates position of all Zwierzaks.
They generally like to cluster, if they are suitably far away
"""
def updateSticklePosition(zwk,mm):

    zwk.prev_pos = zwk.pos

    cur_v = mm.updateSpeed()
    cur_pos = mm.updatePosition()

    zwk.angle[0] = mm.getDirection()

    # print(f'We experience {cur_v} and {cur_pos}')

    zwk.pos = [int(cur_pos[0]),int(cur_pos[1]),int(cur_pos[2])] #for blender comment this line!!! HACK
    return zwk


class Stickle:
    def __init__(self, zwkid, init_pos):
        self.id = zwkid
        self.init_pos=init_pos
        self.pos=init_pos
        self.prev_pos=init_pos
        self.angle = [0,0] #angle[0] alpha, angle[1] beta
        self.islong = 30 #half of width and height as opencv ellipses measurements defined
        self.iswide = 10

        #unusual numbers to encourage program loudly crashing
        self.topleft = -111
        self.bottomright = -111
        self.topleft_prev = -111
        self.bottomright_prev = -111


def main():

    #cv2.namedWindow('HDplane', cv2.WINDOW_GUI_EXPANDED)
    # cv2.moveWindow('HDplane', 200,200)
    side = 2000
    ch = 3 #RGB image displays output

    #keeps the information of previous occupancy
    hdplane = np.zeros((side,side,3),np.uint8)
    #cv2.rectangle(hdplane,(20,20),(side-20,side-20),(230,0,0),4)

    #np.random.seed(0)
    #x_init, y_init = map(int,map(round,np.random.uniform(0, side-1, 2)))
    init_pos = [side//2,side//2,side//2]

    stickle = Stickle('s',init_pos)
    mu_s = 0
    sigma_speed = 40
    sigma_angular_velocity = 0.4
    theta_speed = 0.5
    theta_angular_velocity = 0.8

    # mm = Mooveemodel(x_init,y_init, mu_s, sigma_speed,sigma_angular_velocity,theta_speed, theta_angular_velocity)
    mm = moomodel.Mooveemodel(init_pos, mu_s, sigma_speed,sigma_angular_velocity,theta_speed, theta_angular_velocity, border='normal',side=side)
    #centre, axes W, H, angle, startagnel, endangle, colour, thinkcness
    # cv2.ellipse(hdplane,(100,100),(50,10),30,0,360,(255,255,0),-1)

    for it in range(10000):
        plane_cur = hdplane.copy()
        alf = updateSticklePosition(stickle,mm)
        cv2.ellipse(plane_cur,(alf.pos[0],alf.pos[1]),(alf.islong,alf.iswide),alf.angle[0],0,360,(123, 12,255),-1)
        (head, r1,r2) = getRoI(alf)
        cv2.circle(plane_cur,head,3,(0,255,255))

        cv2.imshow("hdplane",plane_cur)
        key = cv2.waitKey(0)
        if key==ord('q'):
            break

    print('Stickle movement generation done!')

if __name__ == '__main__':
    main()
