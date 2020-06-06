import numpy as np
import matplotlib.pyplot as plt
import cv2

class Reci:
    def __init__ (self, loc):
        self.loc = loc
        self.pdet = 0.5

class Swimloc:
    def __init__ (self, typeid):
        self.typeid = typeid
        if typeid == 0:
            self.name = 'lake'
            self.disp = (40,160,60)
            self.p_hold=0.33
            self.p_up=0.34
            self.p_down=0.33
        if typeid == 1:
            self.name = 'river'
            self.disp = (255,10,10)
            self.p_hold=0.3
            self.p_up=0
            self.p_down=0.7
        if typeid == 2:
            self.name = 'canal'
            self.disp = (50,240,250)
            self.p_hold=0.2
            self.p_up=0
            self.p_down=0.8

class Smolt:
    def __init__ (self, fishid):
        self.fishid = fishid
        self.loc = 0
        self.disp = (np.random.randint(256),np.random.randint(256),np.random.randint(256))
        self.alive = 1 #1 alive, 0 dead
        self.insea = 0 #insea == 1, finished

"""
Updates detections of all smolts.
"""
def updateDetections(fishes, receivers, time):
    for reci in receivers:
        for fish in fishes:
            if fish.loc == reci.loc:
                if np.random.choice([0,1],p=[1-reci.pdet,reci.pdet]):
                    logme = "{},{},{}".format(time,fish.fishid,reci.loc)
                    print(logme)


"""
Updates position of all Smolts.
"""
def updatePosition(fishes, landscape):
    for fish in fishes:
        if fish.alive and not fish.insea:
            fpos = landscape[fish.loc]
            dx = np.random.choice([-1,0,1], p=[fpos.p_up, fpos.p_hold, fpos.p_down])
            p_death = 0.01
            mortality_event = np.random.choice([0,1], p=[1-p_death,p_death])
            if mortality_event:
                fish.alive = 0
            else:
                fish.loc = max(fish.loc + dx,0)#border conidtion
                if (fish.loc == len(landscape)):
                    fish.insea = 1

"""
Version without confluence
 - lake
(255,0,0) - river
(180,220,70) - channel

1 box = 500m (so receiver only spans one area)
"""

def drawLandscape(mylandscape, myreceivers,img):
    for p, landscape in enumerate(mylandscape):
        cv2.rectangle(img,(10*p,0),(10+10*p,10),landscape.disp,-1)
    for reci in myreceivers:
        cv2.rectangle(img,(10*reci.loc,0),(10+10*reci.loc,2),(0,0,255),-1)

def updateLandscape(fishes,img):
    for p, fish in enumerate(fishes):
        y_cent = 15 + 3 * fish.fishid
        x_cent = 5 + 10*fish.loc
        cv2.circle(img,(x_cent,y_cent),2,fish.disp,-1)
        if not fish.alive:
            cv2.line(img,(x_cent-2,y_cent-2),(x_cent+2,y_cent+2),(0,0,255),1)


def showLandscape(landscape_view, fishes):
    landscape_view_now = landscape_view.copy()
    updateLandscape(fishes, landscape_view_now)
    cv2.imshow("system",landscape_view_now)
    #cv2.waitKey(0)
    cv2.waitKey(100)

def main():
    cv2.namedWindow('system', cv2.WINDOW_GUI_EXPANDED)
    cv2.moveWindow('system', 200,200)

    landscape_view = np.zeros((40,600,3),np.uint8)
    mylandscape_list = [0,0,1,2,1,0,0,1,1,1,1,1,1,1,1,2,2,1,0,0,0,0,1]
    mylandscape = list(map(Swimloc, mylandscape_list))
    myreceivers = [Reci(3), Reci(10), Reci(15)]

    fishes = list(map(Smolt, list(range(5))))

    drawLandscape(mylandscape,myreceivers, landscape_view)
    #show initial fish positions
    showLandscape(landscape_view,fishes)

    for t in range(100):
        updatePosition(fishes, mylandscape)
        updateDetections(fishes, myreceivers, t)
        showLandscape(landscape_view,fishes)

if __name__ == '__main__':
    main()
