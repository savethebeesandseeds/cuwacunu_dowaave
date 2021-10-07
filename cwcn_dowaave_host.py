# --- --- --- 
__author__='waajacu'
# --- --- --- 
import sys
import time
import datetime
# --- --- --- 
import cwcn_klmr_host_config as kc
from cwcn_klmr_host_config import CWCN_CURSOR as kccr
from cwcn_klmr_host_config import CWCN_COLORS as kccc
# --- --- --- 
# tail -f KALAMAR.__klmr_file_path
# --- --- --- 
KLMR_CLEAR_CARRIER = '{}{}'.format(kccr.CLEAR_LINE,kccr.CARRIER_RETURN)
assert(kc.KALAMAR_RENDER_MODE in ['terminal','gui','terminal/gui','gui/terminal']), 'WRONG CONFIGURATION, ONLY terminal is implemented as KALAMAR_RENDER_MODE'
# --- --- --- 
import os
# Windows
if os.name == 'nt':
    import msvcrt
# Posix (Linux, OS X)
else:
    import sys
    import termios
    import atexit
    from select import select
import uuid
import urllib.request
import socket
import ast
import tkinter as tk
# --- --- --- --- --- --- 
class TK_RENDER:
    def __init__(self):
        self.window = tk.Tk()
        self.window.configure(background='black')
        self.window.attributes('-fullscreen',True)
        self.ctx_aux_waka=0
        self.c_render={}
        self._init_render_graphs_()
        # self.window.mainloop()
    def _init_render_graphs_(self):
        klmr_state={}
        klmr_state['x_vals']=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        klmr_state['y_vals']=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for _bao in kc.PLOT_RENDER_BAO:
            self.c_render[_bao['ID']] = {'window':self.window}
            for __bao_key in list(_bao.keys()):
                if('update' not in __bao_key):
                    self.c_render[_bao['ID']][__bao_key] = _bao[__bao_key](klmr_state,self.c_render[_bao['ID']])
        self.ctx_aux_waka+=1
    def _update_render_graphs_(self):
        klmr_state={}
        klmr_state['x_vals']=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        klmr_state['y_vals']=[1*self.ctx_aux_waka, 2*self.ctx_aux_waka, 3*self.ctx_aux_waka, 4*self.ctx_aux_waka, 5*self.ctx_aux_waka, 6*self.ctx_aux_waka, 7*self.ctx_aux_waka, 8*self.ctx_aux_waka, 9*self.ctx_aux_waka, 10]
        for _bao in kc.PLOT_RENDER_BAO:
            for __bao_key in list(_bao.keys()):
                if('update' in __bao_key):
                    self.c_render[_bao['ID']][__bao_key] = _bao[__bao_key](klmr_state,self.c_render[_bao['ID']])
        self.ctx_aux_waka+=1
class KBHit:
    # references to: http://home.wlu.edu/~levys/software/kbhit.py
    def __init__(self):
        '''Creates a KBHit object that you can call to do various keyboard things.'''
        if os.name == 'nt':
            pass
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)
            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)
    def set_normal_term(self):
        ''' Resets to normal terminal.  On Windows this is a no-op.
        '''
        if os.name == 'nt':
            pass
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)
    def getch(self):
        ''' Returns a keyboard character after kbhit() has been called.
            Should not be called in the same program as getarrow().
        '''
        s = ''
        if os.name == 'nt':
            return msvcrt.getch().decode('utf-8')
        else:
            return sys.stdin.read(1)
    def getarrow(self):
        ''' Returns an arrow-key code after kbhit() has been called. Codes are
        0 : up  1 : right 2 : down 3 : 
        left Should not be called in the same program as getch().
        '''
        if os.name == 'nt':
            msvcrt.getch() # skip 0xE0
            c = msvcrt.getch()
            vals = [72, 77, 80, 75]
        else:
            c = sys.stdin.read(3)[2]
            vals = [65, 67, 66, 68]
        # return vals.index(ord(c.decode('utf-8')))
        return vals.index(ord(c))
    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.
        '''
        if os.name == 'nt':
            return msvcrt.kbhit()
        else:
            dr,dw,de = select([sys.stdin], [], [], 0)
            return dr != []
def RCsi_CRYPT(key, data):
    S = list(range(256))
    j = 0
    for i in list(range(256)):
        j = (j + S[i] + ord(key[i % len(key)])) % 256
        S[i], S[j] = S[j], S[i]
    j = 0
    y = 0
    out = []
    for char in data:
        j = (j + 1) % 256
        y = (y + S[j]) % 256
        S[j], S[y] = S[y], S[j]
        out.append(chr(ord(char) ^ S[(S[j] + S[y]) % 256]))
    return ''.join(out)
class KALAMAR:
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def __init__(self,__file=None):
        self.__klmr_state=dict([(__,None) for __ in kc.ACTIVE_ADHO_FIELD])
        self.__klmr_file_path = __file
        self.__last_pressed_backtime=None
        if(self.__klmr_file_path is None):
            self.__klmr_out = sys.stdout
        else:
            self.__klmr_out = open(
                self.__klmr_file_path,
                mode='w+',
                buffering=kc.KALAMAR_BUFFER_SIZE
            )
        self.__host=socket.gethostbyname(socket.gethostname())
        self.__node='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self.__host,':'.join(hex(uuid.getnode()).replace('0x', '')[i : i + 2] for i in range(0, 11, 2))))
        self.__symb='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self.__host,kc.SYMBOL_INSTRUMENT))
        self._init_capture_()
        self.__comand=''
        self._init_render_()
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def _init_capture_(self):
        self.__last_pressed_kb = 'None'
        self.__act_pressed_kb = 'None'
        self.__kb = KBHit()
    def _close_capture_(self):
        self.__kb.set_normal_term()
    def kalamar_capture(self):
        if self.__kb.kbhit():
            self.__last_pressed_kb = self.__kb.getch()
            self.__act_pressed_kb=self.__last_pressed_kb
            self.__last_pressed_backtime=time.time()
            self.__klmr_out.write('{}'.format(self.__last_pressed_kb))
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def kalamar_update(self):
        if(kc.KALAMAR_COMMAND_BAO[self.__last_pressed_kb] != kc.KALAMAR_COMMAND_BAO.default_factory()):
            if('MESSAGE' in kc.KALAMAR_COMMAND_BAO[self.__last_pressed_kb]):
                __cmd='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self.__host[::-1],kc.KALAMAR_COMMAND_BAO[self.__last_pressed_kb]))
                __url="{}?symbol={},node={},command={}".format(kc.CLIENT_URL,self.__symb,self.__node,__cmd)
                local_flx, local_headers = urllib.request.urlretrieve(__url)
                # print("\n\n\n")
                # print(local_headers)
                with open(local_flx,encoding='utf-8') as __flx:
                    ujcamei_msg=[RCsi_CRYPT(__url,__) for __ in __flx.read().split("%20") if __!='']
                    # print(ujcamei_msg)
                if('\x00\x84]H' in kc.KALAMAR_COMMAND_BAO[self.__last_pressed_kb]):
                    self.__klmr_state.update(dict([(__.split(":")[0],__.split(":")[1]) for __ in ujcamei_msg if __.split(":")[0] in kc.ACTIVE_ADHO_FIELD]))
                else:
                    print("ERROR!")
                    assert(False), "implement!"
                # print(self.__klmr_state)
                # request.FILES
            else:
                __cmd='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self.__host[::-1],kc.KALAMAR_COMMAND_BAO[self.__last_pressed_kb]))
                __url="{}?symbol={},node={},command={}".format(kc.CLIENT_URL,self.__symb,self.__node,__cmd)
                contents = urllib.request.urlopen(__url).read().decode('ascii') # utf-8
                # self.__klmr_state=ast.literal_eval(RCsi_CRYPT(__url,''.join([chr(int(__,2)) for __ in contents.split('%20')])))
            self.__last_pressed_kb='None'
            self.__last_pressed_backtime="{:.2f} [s]".format(time.time()-self.__last_pressed_backtime)
            # print("contents (reg): {}".format(contents))
            # print("contents (spl): {}".format(contents.split('%20')))
            # print("contents (dec): {}".format(RCsi_CRYPT(__url,''.join([chr(int(__,2)) for __ in contents.split('%20')]))))
        if('gui' in kc.KALAMAR_RENDER_MODE):
            self.tk_render.window.update()
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def _init_render_(self,r_mode=kc.KALAMAR_RENDER_MODE):
        # self.__klmr_out.trucante(0) # clear out
        if('terminal' in r_mode):
            self.__carrier_coord=[0,0] # x,y
            for y_c in range(int(kc.KALAMAR_RESOLUTION[1])):
                self.__klmr_out.write(kccr.NEW_LINE)
                self.__carrier_coord[1]+=kc.SCREEN_TO_KALAMAR_SCALER[1]
        if('gui' in r_mode):
            self.tk_render=TK_RENDER()
    def _close_render_(self):
        if('terminal' in kc.KALAMAR_RENDER_MODE):
            self.__klmr_out.write('{}'.format(kccc.REGULAR))
            self.__klmr_out.flush()
    def kalamar_render(self):
        # self.__klmr_out.write('{}'.format())
        if('terminal' in kc.KALAMAR_RENDER_MODE):
            def goto_zero_coord(other):
                # other.__klmr_out.write('{}'.format(kccr.LEFT*other.__carrier_coord[0]))
                other.__klmr_out.write('{}'.format(kccr.CARRIER_RETURN))
                other.__klmr_out.write('{}'.format(kccr.UP*other.__carrier_coord[1]))
                other.__carrier_coord=[0,0]
            def goto_klmr_coord(other,x,y):
                goto_zero_coord(other)
                other.__klmr_out.write('{}'.format(kccr.RIGHT*x*kc.KALAMAR_TO_SCREEN_SCALER[0]))
                other.__klmr_out.write('{}'.format(kccr.DOWN*y*kc.KALAMAR_TO_SCREEN_SCALER[1]))
                other.__carrier_coord=[x*kc.KALAMAR_TO_SCREEN_SCALER[0],y*kc.KALAMAR_TO_SCREEN_SCALER[1]]
                other.__klmr_out.flush()
            def clear_screen_and_go_to_zero_coord(other):
                other.__carrier_coord=[0,0]
                if os.name == 'nt':
                    os.system("cls")
                else:
                    os.system("clear")
                other._init_render_(r_mode='terminal')
                goto_klmr_coord(other,kc.KALAMAR_RESOLUTION[0],kc.KALAMAR_RESOLUTION[1])
                other.__klmr_out.write('{}'.format("{}{}".format(KLMR_CLEAR_CARRIER,kccr.UP)*other.__carrier_coord[1]))
            # self.__klmr_out.write(KLMR_CLEAR_CARRIER)
            clear_screen_and_go_to_zero_coord(self)
            for render_bao_key in list(kc.KALAMAR_RENDER_BAO.keys()):
                [x_coord,y_coord]=render_bao_key.split(',')
                # for x_ctx in x_coord:
                #     self.__klmr_out.write()
                goto_klmr_coord(self,x=int(x_coord),y=int(y_coord))
                self.__klmr_out.write('{}{}'.format(
                    kc.KALAMAR_RENDER_BAO[render_bao_key]['color'](self),
                    kc.KALAMAR_RENDER_BAO[render_bao_key]['lam'](self)
                ))
            self.__klmr_out.flush()
        if('gui' in kc.KALAMAR_RENDER_MODE):
            self.tk_render._update_render_graphs_()
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

if __name__=='__main__':
    # import ctypes
    # print(ctypes.__dict__)
    # lib=ctypes.cdll.LoadLibrary('linux.so')
    # print(lib)
    # ctypes.windll.user32.MessageBox(0,'waka',1)
    # print(repr(RCsi_CRYPT('','')))
    # input()
    # # c_klmr = KALAMAR('output.log')
    c_klmr = KALAMAR()
    c_klmr.kalamar_capture()
    c_klmr.kalamar_update()
    c_klmr.kalamar_render()
    while(True):
        time.sleep(1/kc.FRAMES_PER_SECOND)
        c_klmr.kalamar_capture()
        c_klmr.kalamar_update()
        c_klmr.kalamar_render()
    try:
        pass
    except:
        c_klmr._close_render_()
        c_klmr._close_capture_()
