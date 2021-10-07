# --- --- --- --- 
# cwcn_config
# --- --- --- --- 
import os
import random
import time
from datetime import datetime, timedelta
from collections import defaultdict
# --- --- --- --- 
# --- --- --- --- 

# --- --- 
# --- --- --- --- 
# --- --- --- --- --- --- --- ---  
PAPER_INSTRUMENT = True #FIXME # < --- --- --- --- FAKE / REAL ; (bool) flag
# --- --- --- --- 
# --- --- --- 
SYMBOL_INSTRUMENT = 'BTCUSDTPERP' #'BCHUSDTPERP' #'BTCUSDTPERP' #'SINE-100'#'BTCUSDTPERP' #'ADAUSDTPERP'/'BTCUSDTPERP'
# --- --- 
ACTIVE_ADHO_FIELD = [
    # 'unrealisedPNL',
    'currency',
    'accountEquity',
    'availableBalance',
    'symbol',
    'unrealisedPnl',
    'realisedPnl',
    'currentQty',
    # --- --- --- 
    'Alliu',
    'sequence',
    'ts',
]
# --- --- --- --- 
# --- --- --- 
# --- --- 
CLIENT_PROTOL = 'http'
CLIENT_DIR = 'cuwacunu'
CLIENT_PORT = 8080
CLIENT_URL = '{}://{}:{}/'.format(CLIENT_PROTOL,CLIENT_DIR,CLIENT_PORT)
# --- --- 
# --- --- --- --- --- --- --- ---  
# --- --- --- --- 
# --- --- --- RENDER
# --- --- 
FRAMES_PER_SECOND = 2
KALAMAR_RENDER_MODE = 'terminal/gui' #'terminal'/'gui'
KALAMAR_BUFFER_SIZE = 1000
SCREEN_RESOLUTION = [120,20] # x_size,y_size
KALAMAR_RESOLUTION = [6,20] # x_size,y_size
# --- --- 
SCREEN_TO_KALAMAR_SCALER = [KALAMAR_RESOLUTION[0]//SCREEN_RESOLUTION[0],KALAMAR_RESOLUTION[1]//SCREEN_RESOLUTION[1]]
KALAMAR_TO_SCREEN_SCALER = [SCREEN_RESOLUTION[0]//KALAMAR_RESOLUTION[0],SCREEN_RESOLUTION[1]//KALAMAR_RESOLUTION[1]]
# --- --- 
if(KALAMAR_RENDER_MODE):
    #FIXME add all render asserts
    assert(KALAMAR_TO_SCREEN_SCALER[1]==1 and SCREEN_TO_KALAMAR_SCALER[1]==1), "WRONG KALAMAR RESOLUTION" #FIXME this is not true
# --- --- 
# --- --- 
class CWCN_COLORS:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    REGULAR = '\033[0m'
    GROSS = '\033[1m'
    DANGER = '\033[41m'
    RED = '\033[31m'
    UNDERLINE = '\033[4m'
    PRICE = '\033[0;32m'
    YELLOW = '\033[1;33m'
    DARKGRAY = '\033[1;30m'
    GRAY = '\033[0;37m'
    WHITE = '\033[1;37m'
class CWCN_CURSOR: #FIXME not in use
    UP='\033[A'
    DOWN='\033[B'
    LEFT='\033[D'
    RIGHT='\033[C'
    CLEAR_LINE='\033[K'
    CARRIER_RETURN='\r'
    NEW_LINE='\n'
# --- --- 

# --- --- 
KALAMAR_COMMAND_BAO = defaultdict(lambda : 'None')
KALAMAR_COMMAND_BAO['+']='BUY'
KALAMAR_COMMAND_BAO['-']='SELL'
KALAMAR_COMMAND_BAO['0']='CLOSE'
KALAMAR_COMMAND_BAO['R']='MESSAGE:ÄZ¬\x8e\\\x88$Æ' # 2x TEHDUJCO
KALAMAR_COMMAND_BAO['W']='MESSAGE:\x00\x84]H' # 2x AHDO
# --- --- 

KALAMAR_RENDER_BAO = {
    '0,0' : {'color':(lambda klmr : CWCN_COLORS.GREEN)      ,'lam':(lambda klmr : 'CUWACUNU')},
    
    # '0,2' : {'color':(lambda klmr : CWCN_COLORS.GREEN)    ,'lam':(lambda klmr : 'unrealisedPNL:')},
    # '1,2' : {'color':(lambda klmr : CWCN_COLORS.GREEN)    ,'lam':(lambda klmr : klmr._KALAMAR__klmr_state['unrealisedPNL'])},
    # '0,4' : {'color':(lambda klmr : CWCN_COLORS.GREEN)    ,'lam':(lambda klmr : 'accountEquity:')},
    # '1,4' : {'color':(lambda klmr : CWCN_COLORS.GREEN)    ,'lam':(lambda klmr : klmr._KALAMAR__klmr_state['accountEquity'])},
    '0,6' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)    ,'lam':(lambda klmr : 'availableBalance:')},
    '1,6' : {'color':(lambda klmr : CWCN_COLORS.GREEN)      ,'lam':(lambda klmr : klmr._KALAMAR__klmr_state['availableBalance'])},
    '0,7' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)    ,'lam':(lambda klmr : 'realisedPnl:')},
    '1,7' : {'color':(lambda klmr : CWCN_COLORS.GREEN)      ,'lam':(lambda klmr : klmr._KALAMAR__klmr_state['realisedPnl'])},
    '0,8' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)    ,'lam':(lambda klmr : 'currency:')},
    '1,8' : {'color':(lambda klmr : CWCN_COLORS.GREEN)      ,'lam':(lambda klmr : klmr._KALAMAR__klmr_state['currency'])},
    
    '0,10' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)   ,'lam':(lambda klmr : 'symbol:')},
    '1,10' : {'color':(lambda klmr : CWCN_COLORS.GREEN)     ,'lam':(lambda klmr : klmr._KALAMAR__klmr_state['symbol'])},
    '0,11' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)    ,'lam':(lambda klmr : 'unrealisedPnl:')},
    '1,11' : {'color':(lambda klmr : CWCN_COLORS.GREEN)      ,'lam':(lambda klmr : klmr._KALAMAR__klmr_state['unrealisedPnl'])},
    '0,12' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)   ,'lam':(lambda klmr : 'currentQty:')},
    '1,12' : {'color':(lambda klmr : CWCN_COLORS.WARNING)   ,'lam':(lambda klmr : klmr._KALAMAR__klmr_state['currentQty'])},
    

    '5,18' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)   ,'lam':(lambda klmr : 'response time:')},
    '6,18' : {'color':(lambda klmr : CWCN_COLORS.GREEN)     ,'lam':(lambda klmr : klmr._KALAMAR__last_pressed_backtime if isinstance(klmr._KALAMAR__last_pressed_backtime,str) else '')},
    '5,19' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)   ,'lam':(lambda klmr : 'command')},
    '6,19' : {'color':(lambda klmr : CWCN_COLORS.GREEN if KALAMAR_COMMAND_BAO[klmr._KALAMAR__act_pressed_kb] != KALAMAR_COMMAND_BAO.default_factory() else CWCN_COLORS.RED), 'lam':(lambda klmr : klmr._KALAMAR__act_pressed_kb)},
    '5,20' : {'color':(lambda klmr : CWCN_COLORS.REGULAR)   ,'lam':(lambda klmr : 'TIME:')},
    '6,20' : {'color':(lambda klmr : CWCN_COLORS.BLUE)      ,'lam':(lambda klmr : datetime.now())},
}

# --- --- --- --- --- --- --- --- --- --- --- --- 
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# --- --- --- --- --- --- 
PLOT_RENDER_BAO = [
    {
        'ID'    :(lambda klmr_state, render : 'waka'),
        'type'  :(lambda klmr_state, render : 'plot'),
        'figure':(lambda klmr_state, render : Figure(figsize=(4,2),facecolor="black")),
        'axis'  :(lambda klmr_state, render : render['figure'].add_subplot(111)),
        'facecolor':(lambda klmr_state, render : render['axis'].set_facecolor((0,0,0))),
        'title' :(lambda klmr_state, render : render['axis'].set_title("Estimation Grid", fontsize=8,color=(1,1,1))),
        'ylabel':(lambda klmr_state, render : render['axis'].set_ylabel("Y",fontsize=8)),
        'xlabel':(lambda klmr_state, render : render['axis'].set_xlabel("X",fontsize=8)),
        'plot'  :(lambda klmr_state, render : render['axis'].plot(klmr_state['x_vals'],klmr_state['y_vals'],color='red')),
        # 'plot2'  :(lambda klmr_state, render : render['axis'].plot(klmr_state['x_vals'],-klmr_state['y_vals'],color='blue')),
        'grid'  :(lambda klmr_state, render : render['axis'].grid(which='major',color='white',linestyle='-',linewidth=0.2)),
        'tick'  :(lambda klmr_state, render : render['axis'].tick_params(colors='red',which='both')),
        'canvas':(lambda klmr_state, render : FigureCanvasTkAgg(render['figure'],master=render['window'])),
        'pack'  :(lambda klmr_state, render : render['canvas'].get_tk_widget().pack()),
        'pos'   :(lambda klmr_state, render : render['canvas'].get_tk_widget().place(x=100, y=50)),
        'draw'  :(lambda klmr_state, render : render['canvas'].draw()),
        # 'update_clear'  :(lambda klmr_state, render : render['axis'].clear()),
        'update_x':(lambda klmr_state, render : render['plot'][0].set_xdata(klmr_state['x_vals'])),
        'update_y':(lambda klmr_state, render : render['plot'][0].set_ydata(klmr_state['y_vals'])),
        'update_relim':(lambda klmr_state, render : render['axis'].relim()),
        'update_autoscale':(lambda klmr_state, render : render['axis'].autoscale_view()),
        'update_draw'  :(lambda klmr_state, render : render['canvas'].draw()),
        'update_flush'  :(lambda klmr_state, render : render['canvas'].flush_events()),
    }
]
# --- --- --- --- --- --- 


