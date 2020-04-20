import math
import yaml
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
    zwk.x_prev = zwk.x_pos
    zwk.y_prev = zwk.y_pos


    if rng.binomial(1,0.05):
        zwk.state=rng.integers(0,2)

    if zwk.state==0:
        zwk.speed = 1
    if zwk.state == 1:
        zwk.speed = rng.normal(3,1,1)
    if zwk.state == 2:
        zwk.speed = zwk.speed + rng.normal(3,1,1)

    #homing
    vec_home = [x_home - zwk.x_pos, y_home - zwk.y_pos]
    dist_home = np.linalg.norm(vec_home)
    ang_home = np.degrees(np.arctan2(vec_home[1],vec_home[0])) #rng.normal(0,10,1)

    ang_explore = rng.normal(0,max(zwk.speed,0),1)
    new_angle = zwk.angle + ang_explore
    dirhome = (ang_home - new_angle)/360 #between 0,1

    if dist_home > 150:
        ang_tohome = 4
    else:
        ang_tohome = 0
    zwk.angle = new_angle + dirhome * ang_tohome

    dx = int(zwk.speed * np.cos(np.pi * zwk.angle/180))
    dy = int(zwk.speed * np.sin(np.pi * zwk.angle/180))
    vec_explore = [dx,dy]


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
    #zwk.state = 3

    # rng = np.random.default_rng()
    # newturn = rng.normal(0,max(10,0),1)
    # zwk.angle = zwk.angle + newturn

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
        self.speed = 2
        self.state = 0 #0 passive, speed = 1, 1 normal, speed around 3

        #unusual numbers to encourage program loudly crashing
        self.topleft = -111
        self.bottomright = -111
        self.topleft_prev = -111
        self.bottomright_prev = -111

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

    annotations_file = 'output/train_data.yml'
    sequence_file = 'output/seq_data.yml'
    all_imgs = []
    all_seq = []


    #cv2.namedWindow('HDplane', cv2.WINDOW_GUI_EXPANDED)
    # cv2.moveWindow('HDplane', 200,200)
    side = 416
    ch = 3 #RGB image displays output
    borders = Borders(20,20,side-20,side-20)

    #keeps the information of previous occupancy
    hdplane = np.zeros((side,side,3),np.uint8) 
    cv2.rectangle(hdplane,(20,20),(side-20,side-20),(230,0,0),4)
    hsv_plane = np.zeros((side,side,ch),float) #HSV values are 0.0:1.0 hue, 0.0:1.0 saturation, 0:255 (int) value

    #np.random.seed(0)
    #x_init, y_init = map(int,map(round,np.random.uniform(0, side-1, 2)))
    x_init, y_init = [side//2,side//2]
    home = [x_init, y_init]

    alf0 = Zwierzak('alf0',x_init,y_init, hue=0,sat=1)
    alf1 = Zwierzak('alf1',130,130,hue=0.6,sat=1)
    # alf2 = Zwierzak('alf2',2000,50,hue=0.2,sat=1)
    # alf3 = Zwierzak('alf3',x_init,y_init,hue=0.3,sat=1)
    # alf4 = Zwierzak('alf4',42,66,hue=0.4,sat=1)
    # alfs = [alf1,alf2,alf3,alf4,alf0]
    alfs = [alf0,alf1]

    #centre, axes W, H, angle, startagnel, endangle, colour, thinkcness
    # cv2.ellipse(hdplane,(100,100),(50,10),30,0,360,(255,255,0),-1)

    for it in range(100):
        for alf in alfs:
            alf = updatePosition(alf,alfs,home[0],home[1])
            alf = handleColisions(alf,borders,alfs)
            hsv_plane = updateTrace(hsv_plane,alf)

        plane_cur = hdplane.copy()

        #saving all the output:
        save_name = 'im' + '{:04d}'.format(it) + '.jpg'
        img_data = {'object':[]}
        img_data['filename'] = save_name
        img_data['width'] = side
        img_data['height'] = side

        if it > 0:
            seq_data = {'object':[]}
            seq_data['filename'] = save_name
            seq_data['p1_filename'] = 'im' + '{:04d}'.format(it-1) + '.jpg'
            seq_data['p2_filename'] = 'im' + '{:04d}'.format(it-2) + '.jpg'
            seq_data['width'] = 416
            seq_data['height'] = 416

        for alf in alfs:
            cv2.ellipse(plane_cur,(alf.x_pos,alf.y_pos),(alf.islong,alf.iswide),alf.angle,0,360,colorsys.hsv_to_rgb(alf.hsv[0], alf.hsv[1],255),-1)
            (head, r1,r2) = getRoI(alf)
            cv2.circle(plane_cur,head,3,(0,255,255))
            # cv2.rectangle(plane_cur,r1,r2,(0,0,255),2)
            alf.topleft = (float(min(r1[0],r2[0])),float(min(r1[1],r2[1])))
            alf.bottomright = (float(max(r1[0],r2[0])),float(max(r1[1],r2[1])))

            obj = {}
            obj['name'] = 'alf'
            obj['xmin'] = alf.topleft[0]
            obj['ymin'] = alf.topleft[1]
            obj['xmax'] = alf.bottomright[0]
            obj['ymax'] = alf.bottomright[1]
            obj['id'] = alf.id
            obj['time']=it
            img_data['object'] += [obj]

            if it > 0:
                obj = {}
                obj['name'] = 'alf'
                obj['xmin'] = alf.topleft[0]
                obj['ymin'] = alf.topleft[1]
                obj['xmax'] = alf.bottomright[0]
                obj['ymax'] = alf.bottomright[1]
                obj['pxmin'] = alf.topleft_prev[0]
                obj['pymin'] = alf.topleft_prev[1]
                obj['pxmax'] = alf.bottomright_prev[0]
                obj['pymax'] = alf.bottomright_prev[1]

                seq_data['object'] += [obj]
            alf.topleft_prev = alf.topleft
            alf.bottomright_prev = alf.bottomright

        if it > 0:
            all_seq += [seq_data]

        cv2.imshow("hdplane",plane_cur)
        cv2.imwrite('output/images/' + save_name,plane_cur)
        all_imgs += [img_data]





        # cv2.waitKey(20)
        key = cv2.waitKey(20)
        if key==ord('q'):
            break

    with open(annotations_file, 'w') as handle:
        yaml.dump(all_imgs, handle)
    with open(sequence_file, 'w') as handle:
        yaml.dump(all_seq, handle)

    hdplane = showTrace(hsv_plane,alf,side,ch)
    cv2.imshow("hdplane",hdplane)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
