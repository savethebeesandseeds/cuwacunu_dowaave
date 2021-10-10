# --- --- --- 
__author__='waajacu'
# --- --- --- 
import sys
import time
import datetime
import os
import copy
if os.name == 'nt': # Windows
    import msvcrt
else: # Posix (Linux, OS X)
    import sys
    import termios
    import atexit
    from select import select
import uuid
import urllib.request
import socket
import ast
import tkinter as tk
import io
from PIL import Image
# --- --- --- 
import cwcn_dwve_host_config as kc
from cwcn_dwve_host_config import CWCN_CURSOR as kccr
from cwcn_dwve_host_config import CWCN_COLORS as kccc
# --- --- --- 
# tail -f DOWAAVE._dwve_file_path
# --- --- --- 
KLMR_CLEAR_CARRIER = '{}{}'.format(kccr.CLEAR_LINE,kccr.CARRIER_RETURN)
assert(kc.DOWAAVE_RENDER_MODE in ['terminal','gui','terminal/gui','gui/terminal']), 'WRONG CONFIGURATION, ONLY terminal is implemented as DOWAAVE_RENDER_MODE'
# --- --- --- 
# --- --- --- --- --- --- 
class TK_RENDER:
    def __init__(self,_dowaave):
        self._dowaave=_dowaave
        self.window = tk.Tk()
        self.window.configure(background='black')
        self.window.attributes('-fullscreen',True)
        self.c_render={}
        # self.window.mainloop()
    def _init_render_graphs_(self):
        # self._dowaave._dwve_state={}
        # self._dowaave._dwve_state['x_vals']=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # self._dowaave._dwve_state['y_vals']=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for _bao in kc.PLOT_RENDER_BAO:
            self.c_render[_bao['ID']] = {'window':self.window}
            for __bao_key in list(_bao.keys()):
                try:
                    if('update' not in __bao_key and __bao_key not in ['ID']):
                        self.c_render[_bao['ID']][__bao_key] = _bao[__bao_key](self._dowaave._dwve_state,self.c_render[_bao['ID']])
                except Exception as e:
                    self._dowaave._add_to_meesage_buffer_("\n {} ERROR! : {} : {} {}".format(kccc.DANGER,__bao_key,e,kccc.REGULAR))
    def _update_render_graphs_(self):
        # self._dowaave._dwve_state={}
        # self._dowaave._dwve_state['x_vals']=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for _bao in kc.PLOT_RENDER_BAO:
            if(_bao['ID'] in list(self._dowaave._dwve_state.keys()) and self._dowaave._past_dwve_state[_bao['ID']] != self._dowaave._dwve_state[_bao['ID']]):
                for __bao_key in list(_bao.keys()):
                    try:
                        if('update' in __bao_key and __bao_key not in ['ID']):
                            self.c_render[_bao['ID']][__bao_key] = _bao[__bao_key](self._dowaave._dwve_state,self.c_render[_bao['ID']])
                    except Exception as e:
                        self._dowaave._add_to_meesage_buffer_("\n {} ERROR! : {} : {} {}".format(kccc.DANGER,__bao_key,e,kccc.REGULAR))
                self._dowaave._past_dwve_state[_bao['ID']]=copy.deepcopy(self._dowaave._dwve_state[_bao['ID']])
                # print(self._dowaave._past_dwve_state[_bao['ID']] != self._dowaave._dwve_state[_bao['ID']])
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
class DOWAAVE:
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def __init__(self,__file=None):
        # --- ---     
        self.daotime=time.time()
        self.taotime=time.time()
        # --- ---     
        self._message_buffer=''
        self._dwve_state=dict([(__,None) for __ in kc.ACTIVE_ADHO_FIELD])
        self._past_dwve_state=dict([(__,'') for __ in kc.ACTIVE_ADHO_FIELD])
        self._dwve_file_path = __file
        self._last_pressed_backtime=None
        if(self._dwve_file_path is None):
            self._dwve_out = sys.stdout
        else:
            self._dwve_out = open(
                self._dwve_file_path,
                mode='w+',
                buffering=kc.DOWAAVE_BUFFER_SIZE
            )
        self._host=socket.gethostbyname(socket.gethostname())
        self._node='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self._host,':'.join(hex(uuid.getnode()).replace('0x', '')[i : i + 2] for i in range(0, 11, 2))))
        self._symb='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self._host,kc.SYMBOL_INSTRUMENT))
        self._init_capture_()
        self._command=''
        self._init_render_()
    # --- --- --- --- --- --- --- --- --- --- --- --- 
    def _add_to_meesage_buffer_(self,msg):
        self._message_buffer="-> {}\n{}\n".format(msg,'\n'.join([_ for _ in self._message_buffer.split('\n')[:5]]))
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def _init_capture_(self):
        self._last_pressed_kb = 'None'
        self._act_pressed_kb = 'None'
        self._kb = KBHit()
    def _close_capture_(self):
        self._kb.set_normal_term()
    def dowaave_capture(self):
        if self._kb.kbhit():
            self._last_pressed_kb = self._kb.getch()
            self._act_pressed_kb=self._last_pressed_kb
            self._last_pressed_backtime=time.time()
            self._dwve_out.write('{}'.format(self._last_pressed_kb))
            self._add_to_meesage_buffer_('user command : {}'.format(repr(self._last_pressed_kb)))
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def decode_cmd(self,v_key=None):
        if(v_key is None):
            __cmd='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self._host[::-1],kc.DOWAAVE_COMMAND_BAO[self._last_pressed_kb]))
        else:
            __cmd='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self._host[::-1],kc.DOWAAVE_COMMAND_BAO[v_key]))
        __url="{}?symbol={},node={},command={}".format(kc.CLIENT_URL,self._symb,self._node,__cmd)
        return __cmd,__url
    # --- --- --- --- --- 
    def state_terminal_update(self,v_key=None):
        __cmd,__url=self.decode_cmd(v_key)
        local_flx, local_headers = urllib.request.urlretrieve(__url)
        with open(local_flx,encoding='utf-8') as __flx:
            ujcamei_msg=[RCsi_CRYPT(__url,__) for __ in __flx.read().split("%20") if __!='']
        self._dwve_state.update(dict([(__.split(":")[0],__.split(":")[1]) for __ in ujcamei_msg if __.split(":")[0] in kc.ACTIVE_ADHO_FIELD]))
    def state_gui_update(self,v_key=None):
        __cmd,__url = self.decode_cmd(v_key)
        local_flx, local_headers = urllib.request.urlretrieve(__url)
        with open(local_flx,'rb') as __flx:
            readed_load=__flx.read()
        ujcamei_msg=[__ for __ in readed_load.decode('utf-8').split("%20") if __!='']
        # print("ok! pass")
        # print(len(ujcamei_msg))
        img={}
        for _c in ujcamei_msg:
            __k = _c.split(':::::')[0]
            __c = _c.split(':::::')[1]
            c_decoded=str(RCsi_CRYPT(__url,"{}".format(__c)))
            c_decoded=[chr(int(__)) for __ in c_decoded.split(',')]
            c_decoded=''.join(c_decoded).encode('iso-8859-1')
            c_stream = io.BytesIO(c_decoded)
            img[__k]=Image.open(c_stream)
            # img[__k].show()
        self._dwve_state.update(img)
    def proceed(self,v_key=None):
        # ... #FIXME encode
        __cmd,__url=self.decode_cmd(v_key)
        local_flx, local_headers = urllib.request.urlretrieve(__url)
    # --- --- --- --- --- 
    def dowaave_update(self):
        if(kc.DOWAAVE_COMMAND_BAO[self._last_pressed_kb] != kc.DOWAAVE_COMMAND_BAO.default_factory()):
            if('\x00\x84]H' in kc.DOWAAVE_COMMAND_BAO[self._last_pressed_kb]):
                self.state_terminal_update()
            elif('ÄZ¬\x8e\\\x88$Æ' in kc.DOWAAVE_COMMAND_BAO[self._last_pressed_kb]):
                self.state_gui_update()
            elif('¸QÐIø5' in kc.DOWAAVE_COMMAND_BAO[self._last_pressed_kb]):
                self.proceed()
            else:
                __cmd='%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(self._host[::-1],kc.DOWAAVE_COMMAND_BAO[self._last_pressed_kb]))
                __url="{}?symbol={},node={},command={}".format(kc.CLIENT_URL,self._symb,self._node,__cmd)
                contents = urllib.request.urlopen(__url).read().decode('utf-8')
                # self._dwve_state=ast.literal_eval(RCsi_CRYPT(__url,''.join([chr(int(__,2)) for __ in contents.split('%20')])))
            self._last_pressed_kb='None'
            self._last_pressed_backtime="{:.2f} [s]".format(time.time()-self._last_pressed_backtime)
            # print("contents (reg): {}".format(contents))
            # print("contents (spl): {}".format(contents.split('%20')))
            # print("contents (dec): {}".format(RCsi_CRYPT(__url,''.join([chr(int(__,2)) for __ in contents.split('%20')]))))
        if('terminal' in kc.DOWAAVE_RENDER_MODE):
            self.state_terminal_update(v_key='w')
        if('gui' in kc.DOWAAVE_RENDER_MODE):
            if(self.tk_render.window.focus_get() is not None):
                self.state_gui_update(v_key='r')
                # self._add_to_meesage_buffer_("{} {}".format(self.tk_render.window.focus_get(),self.tk_render.window.focus_get() is not None))
            self.tk_render.window.update()

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def _init_render_(self,r_mode=kc.DOWAAVE_RENDER_MODE):
        # self._dwve_out.trucante(0) # clear out
        if('terminal' in r_mode):
            self._carrier_coord=[0,0] # x,y
            for y_c in range(int(kc.DOWAAVE_RESOLUTION[1])):
                self._dwve_out.write(kccr.NEW_LINE)
                self._carrier_coord[1]+=kc.SCREEN_TO_DOWAAVE_SCALER[1]
        if('gui' in r_mode):
            self.tk_render=TK_RENDER(self)
            self.tk_render._init_render_graphs_()
            self.tk_render._update_render_graphs_()
    def _close_render_(self):
        if('terminal' in kc.DOWAAVE_RENDER_MODE):
            self._dwve_out.write('{}'.format(kccc.REGULAR))
            self._dwve_out.flush()
    def dowaave_render(self):
        # self._dwve_out.write('{}'.format())
        if('terminal' in kc.DOWAAVE_RENDER_MODE):
            def goto_zero_coord(other):
                # other._dwve_out.write('{}'.format(kccr.LEFT*other._carrier_coord[0]))
                other._dwve_out.write('{}'.format(kccr.CARRIER_RETURN))
                other._dwve_out.write('{}'.format(kccr.UP*other._carrier_coord[1]))
                other._carrier_coord=[0,0]
            def goto_dwve_coord(other,x,y):
                goto_zero_coord(other)
                other._dwve_out.write('{}'.format(kccr.RIGHT*x*kc.DOWAAVE_TO_SCREEN_SCALER[0]))
                other._dwve_out.write('{}'.format(kccr.DOWN*y*kc.DOWAAVE_TO_SCREEN_SCALER[1]))
                other._carrier_coord=[x*kc.DOWAAVE_TO_SCREEN_SCALER[0],y*kc.DOWAAVE_TO_SCREEN_SCALER[1]]
                other._dwve_out.flush()
            def clear_screen_and_go_to_zero_coord(other):
                other._carrier_coord=[0,0]
                if os.name == 'nt':
                    os.system("cls")
                else:
                    os.system("clear")
                other._init_render_(r_mode='terminal')
                goto_dwve_coord(other,kc.DOWAAVE_RESOLUTION[0],kc.DOWAAVE_RESOLUTION[1])
                other._dwve_out.write('{}'.format("{}{}".format(KLMR_CLEAR_CARRIER,kccr.UP)*other._carrier_coord[1]))
            # self._dwve_out.write(KLMR_CLEAR_CARRIER)
            clear_screen_and_go_to_zero_coord(self)
            for render_bao_key in list(kc.DOWAAVE_RENDER_BAO.keys()):
                [x_coord,y_coord]=render_bao_key.split(',')
                # for x_ctx in x_coord:
                #     self._dwve_out.write()
                goto_dwve_coord(self,x=int(x_coord),y=int(y_coord))
                self._dwve_out.write('{}{}'.format(
                    kc.DOWAAVE_RENDER_BAO[render_bao_key]['color'](self),
                    kc.DOWAAVE_RENDER_BAO[render_bao_key]['lam'](self)
                ))
            self._dwve_out.flush()
        if('gui' in kc.DOWAAVE_RENDER_MODE):
            # if(self.tk_render.window.focus_get() is not None):
            # stime=time.time()
            self.tk_render._update_render_graphs_()
            # print("v render time: {}".format(time.time()-stime))
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    def central_loop(self):
        while True:
            try:
                self.daotime=self.taotime
                time.sleep(1/kc.FRAMES_PER_SECOND)
                c_dwve.dowaave_capture()
                c_dwve.dowaave_update()
                self.taotime=time.time()
                c_dwve.dowaave_render()
            except Exception as e:
                print("{}{}{}".format(kccc.RED,e,kccc.REGULAR))
                c_dwve._close_render_()
                c_dwve._close_capture_()
if __name__=='__main__':
    # import ctypes
    # print(ctypes.__dict__)
    # lib=ctypes.cdll.LoadLibrary('linux.so')
    # print(lib)
    # print(repr(RCsi_CRYPT('','')))
    # input()
    # # c_dwve = DOWAAVE('output.log')
    c_dwve = DOWAAVE()
    c_dwve.central_loop()
    # c_dwve.dowaave_capture()
    # c_dwve.dowaave_update()
    # c_dwve.dowaave_render()
