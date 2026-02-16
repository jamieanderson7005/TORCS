import socket
import sys
import getopt
import os
import time

PI= 3.14159265359
data_size = 2**17

def clip(v,lo,hi):
    if v<lo: return lo
    elif v>hi: return hi
    else: return v

class Client():
    def __init__(self,H=None,p=None,i=None,e=None,t=None,s=None,d=None,vision=False):
        self.vision = vision
        self.host= 'localhost'
        self.port= 3001
        self.sid= 'SCR'
        self.maxEpisodes=1 
        self.trackname= 'unknown'
        self.stage= 3 
        self.debug= False
        self.maxSteps= 100000
        
        if H: self.host= H
        if p: self.port= p
        if i: self.sid= i
        self.S= ServerState()
        self.R= DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
            
        # INCREASED TIMEOUT: Gives you 30 seconds to navigate menus
        self.so.settimeout(30.0) 

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
                print(f"Waiting for TORCS on port {self.port}... (Navigate to Quick Race now!)")
                continue

            if '***identified***' in sockdata:
                print("Client connected! Race starting...")
                break

    def get_servers_input(self):
        if not self.so: return
        sockdata= str()
        while True:
            try:
                sockdata,addr= self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                continue
            
            if '***identified***' in sockdata:
                continue
            elif '***shutdown***' in sockdata:
                print("Server stopped the race.")
                self.shutdown()
                return
            else:
                self.S.parse_server_str(sockdata)
                break

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server")

    def shutdown(self):
        if not self.so: return
        self.so.close()
        self.so = None

class ServerState():
    def __init__(self):
        self.servstr= str()
        self.d= dict()

    def parse_server_str(self, server_string):
        self.servstr= server_string.strip()[:-1]
        sslisted= self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w= i.split(' ')
            self.d[w[0]]= destringify(w[1:])

class DriverAction():
    def __init__(self):
       self.d= { 'accel':0.2, 'brake':0, 'clutch':0, 'gear':1, 'steer':0, 'focus':0, 'meta':0 }

    def clip_to_limits(self):
        self.d['steer']= clip(self.d['steer'], -1, 1)
        self.d['brake']= clip(self.d['brake'], 0, 1)
        self.d['accel']= clip(self.d['accel'], 0, 1)

    def __repr__(self):
        self.clip_to_limits()
        out= str()
        for k in self.d:
            out+= '('+k+' '
            v= self.d[k]
            out+= '%.3f' % v if not type(v) is list else ' '.join([str(x) for x in v])
            out+= ')'
        return out

def destringify(s):
    if not s: return s
    if type(s) is str:
        try: return float(s)
        except ValueError: return s
    elif type(s) is list:
        if len(s) < 2: return destringify(s[0])
        else: return [destringify(i) for i in s]