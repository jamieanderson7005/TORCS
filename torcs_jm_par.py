import socket
import sys
import getopt
import os
import time
import math
import random
import pickle
from collections import defaultdict
PI= 3.14159265359

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Q_FILE = os.path.join(BASE_DIR, "torcs_qtable.pkl")

try:
    with open(Q_FILE, "rb") as f:
        loaded_Q = pickle.load(f)
    Q = defaultdict(lambda: [0.0, 0.0, 0.0], loaded_Q)
    print("Loaded existing Q-table")
except Exception as e:
    print("Creating new Q-table:", e)
    Q = defaultdict(lambda: [0.0, 0.0, 0.0])

ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.05

data_size = 2**17
CONNECT_TIMEOUT_SECONDS = 3
CONNECT_RETRIES = 20
SERVER_INPUT_TIMEOUT_LIMIT = 200

ophelp=  'Options:\n'
ophelp+= ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp+= ' --port, -p <port>    TORCS port. [3001]\n'
ophelp+= ' --id, -i <id>        ID for server. [SCR]\n'
ophelp+= ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp+= ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp+= ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp+= ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp+= ' --debug, -d          Output full telemetry.\n'
ophelp+= ' --help, -h           Show this help.\n'
ophelp+= ' --version, -v        Show current version.'
usage= 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage= usage + ophelp
version= "20130505-2"

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

def bargraph(x,mn,mx,w,c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return '' # No width!
    if x<mn: x= mn      # Clip to bounds.
    if x>mx: x= mx      # Clip to bounds.
    tx= mx-mn # Total real units possible to show on graph.
    if tx<=0: return 'backwards' # Stupid bounds.
    upw= tx/float(w) # X Units per output char width.
    if upw<=0: return 'what?' # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu= 0,0,0,0
    if mn < 0: # Then there is a negative part to graph.
        if x < 0: # And the plot is on the negative side.
            negpu= -x + min(0,mx)
            negnonpu= -mn + x
        else: # Plot is on pos. Neg side is empty.
            negnonpu= -mn + min(0,mx) # But still show some empty neg.
    if mx > 0: # There is a positive part to the graph
        if x > 0: # And the plot is on the positive side.
            pospu= x - max(0,mn)
            posnonpu= mx - x
        else: # Plot is on neg. Pos side is empty.
            posnonpu= mx - max(0,mn) # But still show some empty pos.
    nnc= int(negnonpu/upw)*'-'
    npc= int(negpu/upw)*c
    ppc= int(pospu/upw)*c
    pnc= int(posnonpu/upw)*'_'
    return '[%s]' % (nnc+npc+ppc+pnc)

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,vision=False):
        self.vision = vision

        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes=1 # "Maximum number of learning episodes to perform"
        self.trackname= 'unknown'
        self.stage= 3 # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug= False
        self.maxSteps= 100000  # 50steps/second
        self.parse_the_command_line()
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        if e: self.maxEpisodes= e
        if t: self.trackname= t
        if s: self.stage= s
        if d: self.debug= d
        self.S= ServerState()
        self.R= DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(CONNECT_TIMEOUT_SECONDS)

        n_fail = CONNECT_RETRIES
        while True:
            a= "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg='%s(init %s)' % (self.sid,a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata= str()
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                n_fail -= 1
                time.sleep(0.3)
                if n_fail <= 0:
                    print("Could not connect to TORCS server after multiple attempts.")
                    self.so.close()
                    self.so = None
                    sys.exit(-1)

            identify = '***identified***'
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                        ['host=','port=','id=','steps=',
                        'episodes=','track=','stage=',
                        'debug','help','version'])
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == '-h' or opt[0] == '--help':
                    print(usage)
                    sys.exit(0)
                if opt[0] == '-d' or opt[0] == '--debug':
                    self.debug= True
                if opt[0] == '-H' or opt[0] == '--host':
                    self.host= opt[1]
                if opt[0] == '-i' or opt[0] == '--id':
                    self.sid= opt[1]
                if opt[0] == '-t' or opt[0] == '--track':
                    self.trackname= opt[1]
                if opt[0] == '-s' or opt[0] == '--stage':
                    self.stage= int(opt[1])
                if opt[0] == '-p' or opt[0] == '--port':
                    self.port= int(opt[1])
                if opt[0] == '-e' or opt[0] == '--episodes':
                    self.maxEpisodes= int(opt[1])
                if opt[0] == '-m' or opt[0] == '--steps':
                    self.maxSteps= int(opt[1])
                if opt[0] == '-v' or opt[0] == '--version':
                    print('%s %s' % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print('Bad parameter \'%s\' for option %s: %s\n%s' % (
                                opt[1], opt[0], why, usage))
            sys.exit(-1)
        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        """Server's input is stored in a ServerState object."""
        if not self.so:
            return False
        sockdata= str()
        timeout_count = 0

        while True:
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
                timeout_count = 0
            except socket.timeout:
                timeout_count += 1
                print('.', end=' ')
                if timeout_count >= SERVER_INPUT_TIMEOUT_LIMIT:
                    print("\nNo response from TORCS server for too long. Shutting down client.")
                    self.shutdown()
                    return False
                continue
            except socket.error as emsg:
                print("\nSocket receive error: %s" % emsg)
                self.shutdown()
                return False

            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                race_pos = self.S.d.get('racePos', 'unknown')
                print((("Server has stopped the race on %d. "+
                        "You were in %s place.") %
                        (self.port, race_pos)))
                self.shutdown()
                return False
            elif '***restart***' in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return False
            elif not sockdata: # Empty?
                continue       # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H") # Clear for steady output.
                    print(self.S)
                return True

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s" % emsg)
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())

    def shutdown(self):
        if not self.so:
            return

        print(("Race terminated or %d steps elapsed. Shutting down %d."
                % (self.maxSteps, self.port)))

        # ===== STEP 4: SAVE Q-TABLE ON EXIT =====
        with open(Q_FILE, "wb") as f:
            pickle.dump(dict(Q), f)
        print("Q-table saved on shutdown")

        self.so.close()
        self.so = None

class ServerState():
    '''What the server is reporting right now.'''
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

    def __repr__(self):
        return self.fancyout()
        out= str()
        for k in sorted(self.d):
            strout= str(self.d[k])
            if type(self.d[k]) is list:
                strlist= [str(i) for i in self.d[k]]
                strout= ', '.join(strlist)
            out+= "%s: %s\n" % (k,strout)
        return out

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out= str()
        sensors= [ # Select the ones you want in the order you want them.
        'stucktimer',
        'fuel',
        'distRaced',
        'distFromStart',
        'opponents',
        'wheelSpinVel',
        'z',
        'speedZ',
        'speedY',
        'speedX',
        'targetSpeed',
        'rpm',
        'skid',
        'slip',
        'track',
        'trackPos',
        'angle',
        ]

        for k in sensors:
            if type(self.d.get(k)) is list: # Handle list type data.
                if k == 'track': # Nice display for track sensors.
                    strout= str()
                    raw_tsens= ['%.1f'%x for x in self.d['track']]
                    strout+= ' '.join(raw_tsens[:9])+'_'+raw_tsens[9]+'_'+' '.join(raw_tsens[10:])
                elif k == 'opponents': # Nice display for opponent sensors.
                    strout= str()
                    for osensor in self.d['opponents']:
                        if   osensor >190: oc= '_'
                        elif osensor > 90: oc= '.'
                        elif osensor > 39: oc= chr(int(osensor/2)+97-19)
                        elif osensor > 13: oc= chr(int(osensor)+65-13)
                        elif osensor >  3: oc= chr(int(osensor)+48-3)
                        else: oc= '?'
                        strout+= oc
                    strout= ' -> '+strout[:18] + ' ' + strout[18:]+' <-'
                else:
                    strlist= [str(i) for i in self.d[k]]
                    strout= ', '.join(strlist)
            else: # Not a list type of value.
                if k == 'gear': # This is redundant now since it's part of RPM.
                    gs= '_._._._._._._._._'
                    p= int(self.d['gear']) * 2 + 2  # Position
                    l= '%d'%self.d['gear'] # Label
                    if l=='-1': l= 'R'
                    if l=='0':  l= 'N'
                    strout= gs[:p]+ '(%s)'%l + gs[p+3:]
                elif k == 'damage':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,10000,50,'~'))
                elif k == 'fuel':
                    strout= '%6.0f %s' % (self.d[k], bargraph(self.d[k],0,100,50,'f'))
                elif k == 'speedX':
                    cx= 'X'
                    if self.d[k]<0: cx= 'R'
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-30,300,50,cx))
                elif k == 'speedY': # This gets reversed for display to make sense.
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k]*-1,-25,25,50,'Y'))
                elif k == 'speedZ':
                    strout= '%6.1f %s' % (self.d[k], bargraph(self.d[k],-13,13,50,'Z'))
                elif k == 'z':
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k],.3,.5,50,'z'))
                elif k == 'trackPos': # This gets reversed for display to make sense.
                    cx='<'
                    if self.d[k]<0: cx= '>'
                    strout= '%6.3f %s' % (self.d[k], bargraph(self.d[k]*-1,-1,1,50,cx))
                elif k == 'stucktimer':
                    if self.d[k]:
                        strout= '%3d %s' % (self.d[k], bargraph(self.d[k],0,300,50,"'"))
                    else: strout= 'Not stuck!'
                elif k == 'rpm':
                    g= self.d['gear']
                    if g < 0:
                        g= 'R'
                    else:
                        g= '%1d'% g
                    strout= bargraph(self.d[k],0,10000,50,g)
                elif k == 'angle':
                    asyms= [
                        "  !  ", ".|'  ", "./'  ", "_.-  ", ".--  ", "..-  ",
                        "---  ", ".__  ", "-._  ", "'-.  ", "'\.  ", "'|.  ",
                        "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                        "  ---", "  --.", "  -._", "  -..", "  '\.", "  '|."  ]
                    rad= self.d[k]
                    deg= int(rad*180/PI)
                    symno= int(.5+ (rad+PI) / (PI/12) )
                    symno= symno % (len(asyms)-1)
                    strout= '%5.2f %3d (%s)' % (rad,deg,asyms[symno])
                elif k == 'skid': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    skid= 0
                    if frontwheelradpersec:
                        skid= .5555555555*self.d['speedX']/frontwheelradpersec - .66124
                    strout= bargraph(skid,-.05,.4,50,'*')
                elif k == 'slip': # A sensible interpretation of wheel spin.
                    frontwheelradpersec= self.d['wheelSpinVel'][0]
                    slip= 0
                    if frontwheelradpersec:
                        slip= ((self.d['wheelSpinVel'][2]+self.d['wheelSpinVel'][3]) -
                            (self.d['wheelSpinVel'][0]+self.d['wheelSpinVel'][1]))
                    strout= bargraph(slip,-5,150,50,'@')
                else:
                    strout= str(self.d[k])
            out+= "%s: %s\n" % (k,strout)
        return out

class DriverAction():
    def __init__(self):
        self.actionstr= str()
        self.d= { 'accel':0.5,  # Higher initial throttle
                    'brake':0,
                    'clutch':0.5, # Add some initial clutch to prevent stalling
                    'gear':1, 
                    'steer':0,
                    'focus':[-90,-45,0,45,90],
                    'meta':0
                    }

    def clip_to_limits(self):
        self.d['steer']= clip(self.d['steer'], -1, 1)
        self.d['brake']= clip(self.d['brake'], 0, 1)
        self.d['accel']= clip(self.d['accel'], 0, 1)
        self.d['clutch']= clip(self.d['clutch'], 0, 1)
        
        # Change the fallback from 0 to 1 so the car stays in gear
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear']= 1 
        if self.d['meta'] not in [0,1]:
            self.d['meta']= 0

    def __repr__(self):
        self.clip_to_limits()
        out= str()
        for k in self.d:
            out+= '('+k+' '
            v= self.d[k]
            if not type(v) is list:
                out+= '%.3f' % v
            else:
                out+= ' '.join([str(x) for x in v])
            out+= ')'
        return out

    def fancyout(self):
        '''Specialty output for useful monitoring of bot's effectors.'''
        out= str()
        od= self.d.copy()
        od.pop('gear','') # Not interesting.
        od.pop('meta','') # Not interesting.
        od.pop('focus','') # Not interesting. Yet.
        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                strout=''
                strout= '%6.3f %s' % (od[k], bargraph(od[k],0,1,50,k[0].upper()))
            elif k == 'steer': # Reverse the graph to make sense.
                strout= '%6.3f %s' % (od[k], bargraph(od[k]*-1,-1,1,50,'S'))
            else:
                strout= str(od[k])
            out+= "%s: %s\n" % (k,strout)
        return out

def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]

# ================= USER CONFIGURABLE PARAMETERS =================
BASE_TARGET_SPEED = 200
MAX_TARGET_SPEED = 280
MIN_TARGET_SPEED = 45.0
DAMPING_FACTOR = 100.0
STEER_LIMIT = 0.12
STEER_GAIN = 0.55    
CENTERING_GAIN = 1.2  
BRAKE_THRESHOLD = 0.8  
ENABLE_TRACTION_CONTROL = True 
LAST_STEER = 0.0
CENTER_DEADZONE = 0.05

# ================= HELPER FUNCTIONS =================

def classify_corner(track):
    far_min = min(track[0], track[18])
    inner_min = min(track[3], track[15])
    far_score = clip(1.0 - far_min / 80.0, 0.0, 1.0)
    inner_score = clip(1.0 - inner_min / 40.0, 0.0, 1.0)
    severity = far_score * 0.7 + inner_score * 0.3
    return 0.0 if severity < 0.25 else severity

def calculate_steering(S):
    global LAST_STEER
    angle = S.get('angle', 0)
    track_pos = S.get('trackPos', 0)
    speed = S.get('speedX', 0)
    
    # On skinny tracks, we need to react faster to trackPos 
    # but slower at high speeds to prevent spinning out.
    steer_sensitivity = 0.8 / (1.0 + (speed / 150.0))
    
    # High weight on track_pos (1.0) to force the car back to the middle
    target_steer = (angle - track_pos * 1.0) * steer_sensitivity
    
    # Narrower clip for smoothing to prevent jerky movements
    steer_diff = clip(target_steer - LAST_STEER, -0.15, 0.15)
    LAST_STEER += steer_diff
    return LAST_STEER

def calculate_throttle(S, R):
    speed = S.get('speedX', 0)
    
    # Maximize exit speed: If we are below 250, give it 100% throttle
    if speed < 250:
        return 1.0
        
    # Maintain top speed logic
    if speed > MAX_TARGET_SPEED:
        return 0.0
        
    return 1.0 

def apply_brakes(S):
    track = S.get('track', [200]*19)
    speed = S.get('speedX', 0)
    
    dist_ahead = track[9] 
    # If we see more than 100m of track, don't use the 'hard' braking function
    if dist_ahead > 100: 
        return 0.0

    # Calculate a simple required braking force
    # If speed is 200 and dist is 50, we need to shed speed fast.
    danger_factor = speed / (dist_ahead + 0.1)
    if danger_factor > 1.5:
        return clip(danger_factor * 0.2, 0.0, 1.0)
    return 0.0

def shift_gears(S):
    rpm = S.get('rpm', 0)
    gear = S.get('gear', 1)
    
    # Simple, high-rev shifting to keep torque high
    if gear < 6 and rpm > 8000:
        return gear + 1
    if gear > 1 and rpm < 3500:
        return gear - 1
    return gear

def traction_control(S, accel):
    if not ENABLE_TRACTION_CONTROL: return accel
    w = S.get('wheelSpinVel', [0,0,0,0])
    
    slip = (w[2] + w[3]) - (w[0] + w[1])

    if slip > 40.0:
        reduction = clip(slip / 200.0, 0, 0.5)
        accel -= reduction
        
    return clip(accel, 0, 1)

# ================= MACHINE LEARNING HELPERS =================

def get_state(S):
    # Increased from 5 bins to 11 bins for finer center-line control
    # trackPos ranges from -1 to 1; this maps it to integers 0-10
    t_pos = int(clip(S['trackPos'] * 5 + 5, 0, 10)) 
    
    # Angle (5 bins instead of 3 for better orientation awareness)
    raw_angle = S['angle']
    if raw_angle < -0.2: angle = 0
    elif raw_angle < -0.05: angle = 1
    elif raw_angle < 0.05: angle = 2
    elif raw_angle < 0.2: angle = 3
    else: angle = 4
    
    # Speed
    sp = S['speedX']
    speed_bin = 0 if sp < 30 else 1 if sp < 70 else 2 if sp < 120 else 3 if sp < 180 else 4
    
    # Curvature
    track = S.get('track', [200]*19)
    curve = 0 if (track[0] - track[18]) > 10 else 2 if (track[18] - track[0]) > 10 else 1
    
    return (t_pos, angle, speed_bin, curve)

def choose_action(state):
    if random.random() < EPSILON:
        return random.randint(0, 2)
    return Q[state].index(max(Q[state]))

def action_to_steer(a):
    return [-0.3, 0.0, 0.3][a]

def get_reward(S):
    if abs(S['trackPos']) > 1.0:
        return -50.0 # Increased penalty from -20

    progress = S['speedX'] * math.cos(S['angle'])
    # Square the trackPos to punish being near the edge more than being near the center
    center_penalty = (S['trackPos'] ** 2) * 2.0 

    return (progress * 0.1) - center_penalty

def drive_modular(c):
    global EPSILON, LAST_STEER
    S, R = c.S.d, c.R.d

    if 'speedX' not in S: return

    # --- 1. SENSOR DATA ---
    speed = S.get('speedX', 0)
    track = S.get('track', [200]*19)
    angle = S.get('angle', 0)
    t_pos = S.get('trackPos', 0)
    stuck = S.get('stucktimer', 0)
    rpm = S.get('rpm', 0)
    
    # --- 2. STUCK RECOVERY ---
    if stuck > 100 or abs(angle) > 1.5:
        R['gear'] = -1 if speed > -5 else 1 
        R['steer'] = -angle if abs(angle) < 1.5 else (1.0 if t_pos > 0 else -1.0)
        R['accel'] = 0.5
        R['brake'] = 0.0
        return 

    # --- 3. DYNAMIC TARGET SPEED ---
    dist_ahead = track[9]
    wall_proximity = min(track[0], track[18]) 
    target_speed = math.sqrt(max(0, dist_ahead) * 320) + (wall_proximity * 0.6)
    target_speed = clip(target_speed, 45.0, 260.0)

    # --- 4. CENTER-SEEKING STEERING ---
    damping_factor = 100.0
    # Sensitivity decreases with speed to maintain stability
    steer_sensitivity = 0.55 / (1.0 + (speed / damping_factor))
    
    # Increase CENTERING_FORCE to pull the car back to the middle (0.0)
    # We use (t_pos * 0.75) instead of 0.45 for a stronger center-line hold
    centering_force = t_pos * 0.75
    
    # Curvature detection (look-ahead)
    curvature = (track[18] - track[0]) / 200.0
    
    # Combine everything: Orientation + Stronger Centering + Curve Prediction
    physics_steer = (angle - centering_force + curvature) * steer_sensitivity

    # Smooth steering transitions
    steer_limit = 0.12
    diff = clip(physics_steer - LAST_STEER, -steer_limit, steer_limit)
    R['steer'] = LAST_STEER + diff
    LAST_STEER = R['steer']

    # --- 5. CONTROLLED THROTTLE & BRAKES ---
    traction_limit = 1.0 - (abs(R['steer']) * 0.7)
    
    if speed < target_speed:
        R['accel'] = clip(1.0 * traction_limit, 0.1, 1.0)
        R['brake'] = 0.0
    else:
        R['accel'] = 0.0
        R['brake'] = clip((speed - target_speed) / 15.0, 0.0, 0.8)

    # --- 6. GEARS & RL UPDATES ---
    if R['gear'] < 6 and rpm > 7500: R['gear'] += 1
    elif R['gear'] > 1 and rpm < 2500: R['gear'] -= 1

    current_state = get_state(S)
    action_idx = choose_action(current_state)
    
    reward = get_reward(S) # Note: your get_reward already punishes trackPos**2
    next_state = get_state(S)
    old_value = Q[current_state][action_idx]
    next_max = max(Q[next_state])
    Q[current_state][action_idx] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

# ================= MAIN LOOP =================
if __name__ == "__main__":
    C = Client(p=3001)

    for step in range(C.maxSteps):
        has_input = C.get_servers_input()
        if not has_input:
            break
        drive_modular(C)
        C.respond_to_server()

        # Periodic Save
        if step % 5000 == 0:
            with open(Q_FILE, "wb") as f:
                pickle.dump(dict(Q), f)

    # Final Save
    with open(Q_FILE, "wb") as f:
        pickle.dump(dict(Q), f)

    if C.so:
        C.shutdown()