import numpy as np
import matplotlib.pyplot as plt

def updatePosition2(x,y,side,home):
    fardist = 30. #test
    #distance from home
    dhome=np.array([home[0]-x,home[1]-y])
    #dx, dy = 3*np.ceil(np.random.rand(2,) + (dhome/fardist)*np.random.rand(2,)).astype(int)-1
    dx, dy = np.round(2*np.random.rand(2,)-1).astype(int)
    #sys.stdout.write(str(int(dhome[0])))
    #sys.stdout.flush()

    #print('dx: {0}, dy: {1}'.format(dx,dy))
    #boundary teleportation: (aka periodic)
    #x = (x+dx)%side
    #y = (y+dy)%side
    #boundary reflective:
    x = max(min(x+dx,side-1),0)
    y = max(min(y+dy,side-1),0)

    return x, y


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
    x_pos=0
    y_pos=0
    def __init__(self, x_init,y_init):
        self.x_pos=x_init
        self.y_pos=y_init

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
    borders = Borders(0,0,side,side)
    plane = np.zeros((side,side),int)

    #np.random.seed(0)
    #x_init, y_init = map(int,map(round,np.random.uniform(0, side-1, 2)))
    x_init, y_init = [side//2,side//2] 
    home = [x_init, y_init]

    alf = Zwierzak(x_init,y_init)

    for it in range(1000):
        alf = updatePosition(alf)
        alf = handleColisions(alf,borders)
        plane[alf.x_pos,alf.y_pos]=plane[alf.x_pos,alf.y_pos]+1

    plt.imshow(plane)
    plt.savefig('test.png')
    plt.show()

if __name__ == '__main__':
    main()
