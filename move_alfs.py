
"""
We have to get RoI analytically because otherwise we cannot have overlapping even if we can resolved which recognised contour is which animals bounding box.
There are two things to do
TODO make it more reasonable padding for every postion of the ellipse
TODO make it work with my expanded conciousness (shape i mean physical shape)
"""
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


    return (x,y),(x + w, y + h)



def normToOne(vallist):
    valsum = sum(vallist)
    return [x/valsum for x in vallist]

"""
Updates position of all Zwierzaks.
We are allowing them to run on top of each other for now...
"""
def updateZwkPosition(zwk,zwks,side,mm):

    zwk.x_prev = zwk.x_pos
    zwk.y_prev = zwk.y_pos

    cur_v = mm.updateSpeed()
    cur_pos, is_same_panel = mm.updatePosition(side)


    zwk.angle = mm.getDirection()

    zwk.x_pos = int(cur_pos[0])
    zwk.y_pos = int(cur_pos[1])
    return zwk, is_same_panel


"""
This movement model need to be just the movement model so my position on the map initis etc have to be moved out of here
"""
class Mooveemodel:
    def __init__(self, x_init, y_init, mu_s, sigma_speed, sigma_angular_velocity, theta_speed, theta_angular_velocity):
        # [speed and angular velocity]
        self.mu = np.array([mu_s,0.])
        self.theta = np.array([theta_speed,theta_angular_velocity])
        self.sigma = np.array([sigma_speed,sigma_angular_velocity])
        self.v = np.array(self.mu)
        self.dt = np.ones(2)
        self.rng = np.random.default_rng()
        self.pos = np.array([x_init,y_init])
        self.angle = 0.
        self.os = np.array(self.mu)
        self.s = 0
        self.updateSpeed()

    def updateSpeed(self):
        os1 = self.os
        mu1 = self.mu
        theta1 = self.theta
        dt1 = self.dt
        sigma1 = self.sigma
        rng1 = self.rng

        self.os = (os1
            + theta1 * (mu1 - os1) * dt1
            + sigma1 * rng1.normal(0,np.sqrt(dt1),2)
        )

        self.angle = self.angle + self.os[1] * dt1[1]
        #self.s = np.log1p(np.exp(self.os[0])) #softplus cause it to get stuck in 0.
        self.s = abs(self.os[0])
        self.v[0] = self.s*np.cos(self.angle)
        self.v[1] = self.s*np.sin(self.angle)

        return self.v

    """
    Update the position and tell us if we have moved past the border
    """
    def updatePosition(self, side):
        new_pos = self.pos + self.v * self.dt
        self.pos = new_pos % side
        is_same_panel = True if np.all(new_pos == self.pos) else False
        return self.pos, is_same_panel

    def getDirection(self):
        return np.degrees(np.arctan2(self.v[1],self.v[0]))

"""
Our animal can have different colour or the same
"""

class Zwierzak:
    def __init__(self, zwkid, x_init,y_init, mm, hue=0, sat=1):
        self.mm = mm #movememnt mode, each animus has its own now
        self.id = zwkid
        self.x_pos=x_init
        self.y_pos=y_init
        self.x_prev=x_init
        self.y_prev=y_init
        self.hsv=(hue,sat,0) # initialise as a dim value
        self.angle = 0
        self.islong = 30 #half of width and height as opencv ellipses measurements defined
        self.iswide = 10
        self.speed = 2 #shouldn't that be mu_s?
        self.state = 0 #we will use state to define our little accelreated moments.

        #unusual numbers to encourage program loudly crashing
        self.topleft = -111
        self.bottomright = -111
        self.topleft_prev = -111
        self.bottomright_prev = -111

        self.panelswitcher = deque([False, False, False])

    """
    In case of periodic border condition we need to be able to always see our animal. However, if there are multiple animals in the scene it means that their relative position is messed up.
    It is fine though, we are looking at each 3-frame scenario as a separate tracking problem. Also we exclude frames that have animals close to the border.
    """
    def observationPointSwitch(self, is_same_panel):
        self.panelswitcher.popleft()
        self.panelswitcher.append(is_same_panel)
        return np.all(self.panelswitcher)

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


def main(args):

    debug = True

    #read from commandline
    if not debug:
        ddir = f'output/{args.ddir[0]}'
        dp = args.datapoints[0]
        mu_s = args.muspeed[0]
        sigma_speed = args.sigmaspeed[0]
        sigma_angular_velocity = args.sigmaangularvelocity[0]
        theta_speed = args.thetaspeed[0]
        theta_angular_velocity = args.thetaangularvelocity[0]
        show_img = args.visual
    else:
    #here is for testing:
        ddir = f'output/testrun/'
        dp = 20
        mu_s = 3
        sigma_speed = 20
        sigma_angular_velocity = 0.2
        theta_speed = 0.5
        theta_angular_velocity = 0.5
        show_img = True

    #prepare directories
    an_dir = os.path.join(ddir,"annotations")
    img_dir = os.path.join(ddir,"images")
    os.makedirs(an_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    annotations_file = an_dir + '/train_data.yml'
    sequence_file = an_dir + '/seq_data.yml'
    all_imgs = []
    all_seq = []



    borders = Borders(1,1,side-1,side-1)


    # #cv2.namedWindow('HDplane', cv2.WINDOW_GUI_EXPANDED)
    # # cv2.moveWindow('HDplane', 200,200)
    # ch = 3 #RGB image displays output
    # #keeps the information of previous occupancy
    hdplane = np.zeros((side,side,3),np.uint8)
    # #cv2.rectangle(hdplane,(20,20),(side-20,side-20),(230,0,0),4)
    # hsv_plane = np.zeros((side,side,ch),float) #HSV values are 0.0:1.0 hue, 0.0:1.0 saturation, 0:255 (int) value


    #np.random.seed(0)
    #x_init, y_init = map(int,map(round,np.random.uniform(0, side-1, 2)))
    x_init, y_init = [side//2,side//2]

    mm1 = Mooveemodel(x_init,y_init, mu_s, sigma_speed,sigma_angular_velocity,theta_speed, theta_angular_velocity)
    mm2 = Mooveemodel(x_init,y_init, mu_s, sigma_speed,0,theta_speed, 0)

    alf0 = Zwierzak('alf0',x_init,y_init, mm1, hue=0,sat=1)
    alf1 = Zwierzak('alf1',130,130, mm2, hue=0.6,sat=1)
    # alf2 = Zwierzak('alf2',2000,50,hue=0.2,sat=1)
    # alf3 = Zwierzak('alf3',x_init,y_init,hue=0.3,sat=1)
    # alf4 = Zwierzak('alf4',42,66,hue=0.4,sat=1)
    # alfs = [alf1,alf2,alf3,alf4,alf0]
    alfs = [alf0, alf1]
    # alfs = [alf0]


    #centre, axes W, H, angle, startagnel, endangle, colour, thinkcness
    # cv2.ellipse(hdplane,(100,100),(50,10),30,0,360,(255,255,0),-1)

    for it in range(dp):
        plane_cur = hdplane.copy()
        recthosealfs = [] #all animals must be visible and moving within current panel to be useful for training

        #saving all the output:
        save_name = 'alfim' + '{:05d}'.format(it) + '.jpg'
        img_data = {'object':[]}
        img_data['filename'] = save_name
        img_data['width'] = side
        img_data['height'] = side


        for alf in alfs:
            alf, is_same_panel = updateZwkPosition(alf,alfs,side,alf.mm)
            hsv_plane = updateTrace(hsv_plane,alf)
            cv2.ellipse(plane_cur,(alf.x_pos,alf.y_pos),(alf.islong,alf.iswide),alf.angle,0,360,colorsys.hsv_to_rgb(alf.hsv[0], alf.hsv[1],255),-1)
            (head, r1,r2) = getRoI(alf)

            cv2.circle(plane_cur,head,3,(0,255,255))

            roiNotOnBorder = True #or beyond....
            if \
            r1[0]<=0 or \
            r1[0]>=side or \
            r2[0]<=0 or \
            r2[0]>=side or \
            r1[1]<=0 or \
            r1[1]>=side or \
            r2[1]<=0 or \
            r2[1]>=side:
                roiNotOnBorder = False

            recthosealfs.append(alf.observationPointSwitch((is_same_panel and roiNotOnBorder)))

            #uncomment the following line to see bounding boxez
            # cv2.rectangle(plane_cur,r1,r2,(123,20,255),2) # show bounding box

            alf.topleft = (float(min(r1[0],r2[0])),float(min(r1[1],r2[1])))
            alf.bottomright = (float(max(r1[0],r2[0])),float(max(r1[1],r2[1])))
            obj = dict()
            obj['name'] = 'alf'
            obj['xmin'] = alf.topleft[0]
            obj['ymin'] = alf.topleft[1]
            obj['xmax'] = alf.bottomright[0]
            obj['ymax'] = alf.bottomright[1]
            obj['id'] = alf.id
            obj['time']=it
            img_data['object'] += [obj]

            # print("New TL again: {}".format(alf.topleft[0]))
            # print("Old TL: {}".format(alf.topleft_prev[0]))

        record_the_seq = np.all(recthosealfs)

        if record_the_seq:
            # cv2.putText(plane_cur, "R",  (30,30), cv2. FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0,0,250), 2);
            seq_data = {'object':[]}
            seq_data['filename'] = save_name
            seq_data['p1_filename'] = 'alfim' + '{:05d}'.format(it-1) + '.jpg'
            seq_data['p2_filename'] = 'alfim' + '{:05d}'.format(it-2) + '.jpg'
            seq_data['width'] = 416
            seq_data['height'] = 416

        for alf in alfs:
            if record_the_seq:
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


        if record_the_seq:
            all_seq += [seq_data]

        cv2.imwrite(img_dir + '/' + save_name,plane_cur)
        all_imgs += [img_data]

        if show_img:# and record_the_seq:
            cv2.imshow("hdplane",plane_cur)
            key = cv2.waitKey(0)
            if key==ord('q'):
                break


    with open(annotations_file, 'w') as handle:
        yaml.dump(all_imgs, handle)
    with open(sequence_file, 'w') as handle:
        yaml.dump(all_seq, handle)

    # hdplane = showTrace(hsv_plane,alf,side,ch)
    # cv2.imshow("hdplane",hdplane)
    # cv2.waitKey(0)
    print('don done!')
    cv2.destroyAllWindows()
