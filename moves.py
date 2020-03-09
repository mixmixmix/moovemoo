import numpy as np
import matplotlib.pyplot as plt
import colorsys
from skimage.color import hsv2rgb

"""
Updates position of all Zwierzaks
"""
def updatePosition(zwk):
    x_p = zwk.x_pos
    y_p = zwk.y_pos
    dx, dy = np.round(2*np.random.rand(2,)-1).astype(int)
    x_n = x_p+dx
    y_n = y_p+dy
    zwk.x_pos = x_n
    zwk.y_pos = y_n
    return zwk

"""
Handle colisions:
 - boundry conditions? Let's make it reflective, but simulate so that animals keep off the long grass.
 - collisions with other animals: This is going to be async, so if animal wants to move to other position, it will not make this movement. I need occupancy grid for that.
"""
def handleColisions(zwk, borders):
    rside = borders.x_max
    lside = borders.x_min # ASSUME IT IS SQUARE
    zwk.x_pos = max(min(zwk.x_pos,rside-1),lside)
    zwk.y_pos = max(min(zwk.y_pos,rside-1),lside)
    return zwk


class Zwierzak:
    # x_pos=0
    # y_pos=0
    # hue = 0
    def __init__(self, x_init,y_init, hue=0, sat=1):
        self.x_pos=x_init
        self.y_pos=y_init
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
    side = 100
    ch = 3 #RGB image displays output
    borders = Borders(0,0,side,side)
    hsv_plane = np.zeros((side,side,ch),float) #HSV values are 0.0:1.0 hue, 0.0:1.0 saturation, 0:255 (int) value

    #np.random.seed(0)
    #x_init, y_init = map(int,map(round,np.random.uniform(0, side-1, 2)))
    x_init, y_init = [side//2,side//2]
    home = [x_init, y_init]

    alf1 = Zwierzak(x_init,y_init, hue=0,sat=1)
    alf2 = Zwierzak(0,0,hue=0.1,sat=1)
    alf3 = Zwierzak(0,0,hue=0.2,sat=1)
    alf4 = Zwierzak(x_init,y_init,hue=0.3,sat=1)
    alf5 = Zwierzak(0,0,hue=0.4,sat=1)
    alfs = [alf1,alf2,alf3,alf4,alf5]

    for it in range(1000):
        for alf in alfs:
            alf = updatePosition(alf)
            alf = handleColisions(alf,borders)
            cc = hsv_plane[alf.x_pos,alf.y_pos]
            hsv_plane[alf.x_pos,alf.y_pos]=(alf.hsv[0], alf.hsv[1],min(cc[2]+10,255))
            # print(plane[alf.x_pos,alf.y_pos])

    plane = np.zeros((side,side,ch),int)
    #change the whole image from hsv to rgb
    for i in range(plane.shape[0]):
        for j in range(plane.shape[1]):
            plane[j,i] = colorsys.hsv_to_rgb(hsv_plane[j,i][0],hsv_plane[j,i][1],hsv_plane[j,i][2])
    #planergb = hsv2rgb(plane)

    plt.imshow(plane)
    plt.savefig('test.png')
    plt.show()

if __name__ == '__main__':
    main()
