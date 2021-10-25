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
# --- --- --- SYMBOL INSTRUMENT MUST MATCH WHAT IS CONFIGURED ON CLIENT
# SYMBOL_INSTRUMENT = 'ETHUSDTPERP' #'BCHUSDTPERP' #'BTCUSDTPERP' #'SINE-100'#'BTCUSDTPERP' #'ADAUSDTPERP'/'BTCUSDTPERP'
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
    # --- --- --- 
    'DOWAAVE_GSS_F1',
    'DOWAAVE_GSS_F2',
    'DOWAAVE_TFT_SHORT_F1',
    'DOWAAVE_TFT_SHORT_F2',
    'DOWAAVE_HRZ_F1',
    # --- --- --- 
    'price',
    'markPrice',
    'realLeverage',
    # --- --- --- 
    'activeStrategy',
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
DOWAAVE_RENDER_MODE = 'terminal/gui' #'terminal'/'gui'
DOWAAVE_BUFFER_SIZE = 1000
SCREEN_RESOLUTION = [120,20] # x_size,y_size
DOWAAVE_RESOLUTION = [6,20] # x_size,y_size
# --- --- 
SCREEN_TO_DOWAAVE_SCALER = [DOWAAVE_RESOLUTION[0]//SCREEN_RESOLUTION[0],DOWAAVE_RESOLUTION[1]//SCREEN_RESOLUTION[1]]
DOWAAVE_TO_SCREEN_SCALER = [SCREEN_RESOLUTION[0]//DOWAAVE_RESOLUTION[0],SCREEN_RESOLUTION[1]//DOWAAVE_RESOLUTION[1]]
# --- --- 
if(DOWAAVE_RENDER_MODE):
    #FIXME add all render asserts
    assert(DOWAAVE_TO_SCREEN_SCALER[1]==1 and SCREEN_TO_DOWAAVE_SCALER[1]==1), "WRONG DOWAAVE RESOLUTION" #FIXME this is not true
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
DOWAAVE_COMMAND_BAO = defaultdict(lambda : 'None')
DOWAAVE_COMMAND_BAO['R']='HEAVYLOAD:ÄZ¬\x8e\\\x88$Æ' # 2x TEHDUJCO
DOWAAVE_COMMAND_BAO['W']='HEAVYLOAD:\x00\x84]H' # 2x AHDO
DOWAAVE_COMMAND_BAO['r']='KUAILOAD:ÄZ¬\x8e\\\x88$Æ' # 2x TEHDUJCO
DOWAAVE_COMMAND_BAO['w']='KUAILOAD:\x00\x84]H' # 2x AHDO
DOWAAVE_COMMAND_BAO['+']='PROCEED:¸QÐIø5:BUY' # 2x avyeta 
DOWAAVE_COMMAND_BAO['-']='PROCEED:¸QÐIø5:SELL' # 2x avyeta 
DOWAAVE_COMMAND_BAO['0']='PROCEED:¸QÐIø5:CLOSE' # 2x avyeta 
DOWAAVE_COMMAND_BAO['.']='PROCEED:¸QÐIø5:CHANGE_INSTRUMENT' # 2x avyeta 
DOWAAVE_COMMAND_BAO['*']='PROCEED:¸QÐIø5:CHANGE_STRATEGY' # 2x avyeta
# --- --- 

DOWAAVE_RENDER_BAO = {
    '0,0' : {'color':(lambda dwve : CWCN_COLORS.GREEN)      ,'lam':(lambda dwve : 'CUWACUNU')},
    
    # '0,2' : {'color':(lambda dwve : CWCN_COLORS.GREEN)    ,'lam':(lambda dwve : 'unrealisedPNL:')},
    # '1,2' : {'color':(lambda dwve : CWCN_COLORS.GREEN)    ,'lam':(lambda dwve : dwve._dwve_state['unrealisedPNL'])},
    # '0,4' : {'color':(lambda dwve : CWCN_COLORS.GREEN)    ,'lam':(lambda dwve : 'accountEquity:')},
    # '1,4' : {'color':(lambda dwve : CWCN_COLORS.GREEN)    ,'lam':(lambda dwve : dwve._dwve_state['accountEquity'])},
    '0,7' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)    ,'lam':(lambda dwve : 'availableBalance:')},
    '1,7' : {'color':(lambda dwve : CWCN_COLORS.GREEN)      ,'lam':(lambda dwve : dwve._dwve_state['availableBalance'])},
    '0,8' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)    ,'lam':(lambda dwve : 'currency:')},
    '1,8' : {'color':(lambda dwve : CWCN_COLORS.GREEN)      ,'lam':(lambda dwve : dwve._dwve_state['currency'])},
    
    '0,9' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'symbol:')},
    '1,9' : {'color':(lambda dwve : CWCN_COLORS.GREEN)     ,'lam':(lambda dwve : dwve._dwve_state['symbol'])},
    '0,10' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'price')},
    '1,10' : {'color':(lambda dwve : CWCN_COLORS.GREEN)     ,'lam':(lambda dwve : '{} [{}]'.format(dwve._dwve_state['price'],dwve._dwve_state['currency']))},
    '0,11' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'markPrice')},
    '1,11' : {'color':(lambda dwve : CWCN_COLORS.GREEN)     ,'lam':(lambda dwve : '{} [{}]'.format(dwve._dwve_state['markPrice'],dwve._dwve_state['currency']))},
    '0,12' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'unrealisedPnl:')},
    '1,12' : {'color':(lambda dwve : CWCN_COLORS.GREEN)     ,'lam':(lambda dwve : "{}{}{}".format(CWCN_COLORS.RED if float(dwve._dwve_state['unrealisedPnl'])<0 else CWCN_COLORS.GREEN if float(dwve._dwve_state['unrealisedPnl'])>0 else CWCN_COLORS.WARNING,dwve._dwve_state['unrealisedPnl'],CWCN_COLORS.REGULAR) if dwve._dwve_state['unrealisedPnl'] is not None else CWCN_COLORS.WARNING)},
    '0,13' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)    ,'lam':(lambda dwve : 'realisedPnl:')},
    '1,13' : {'color':(lambda dwve : CWCN_COLORS.GREEN)     ,'lam':(lambda dwve : "{}{}{}".format(CWCN_COLORS.RED if float(dwve._dwve_state['realisedPnl'])<0 else CWCN_COLORS.GREEN if float(dwve._dwve_state['realisedPnl'])>0 else CWCN_COLORS.WARNING,dwve._dwve_state['realisedPnl'],CWCN_COLORS.REGULAR) if dwve._dwve_state['realisedPnl'] is not None else CWCN_COLORS.WARNING)},
    '0,14' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'currentQty:')},
    '1,14' : {'color':(lambda dwve : CWCN_COLORS.WARNING)   ,'lam':(lambda dwve : dwve._dwve_state['currentQty'])},
    '0,15' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'realLeverage:')},
    '1,15' : {'color':(lambda dwve : CWCN_COLORS.WARNING)   ,'lam':(lambda dwve : dwve._dwve_state['realLeverage'])},
    

    '5,3' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'activeStrategy:')},
    '6,3' : {'color':(lambda dwve : CWCN_COLORS.DANGER)     ,'lam':(lambda dwve : dwve._dwve_state['activeStrategy'])},

    '5,17' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'response time:')},
    '6,17' : {'color':(lambda dwve : CWCN_COLORS.GREEN)     ,'lam':(lambda dwve : dwve._last_pressed_backtime if isinstance(dwve._last_pressed_backtime,str) else '')},
    '5,18' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : "heart beat rate:")},
    '6,18' : {'color':(lambda dwve : CWCN_COLORS.BLUE)      ,'lam':(lambda dwve : "{:.4} [Hz]".format(1/(dwve.taotime-dwve.daotime)))},
    '5,19' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'command')},
    '6,19' : {'color':(lambda dwve : CWCN_COLORS.GREEN if DOWAAVE_COMMAND_BAO[dwve._act_pressed_kb] != DOWAAVE_COMMAND_BAO.default_factory() else CWCN_COLORS.RED), 'lam':(lambda dwve : "{}".format(repr(dwve._act_pressed_kb)) if DOWAAVE_COMMAND_BAO[dwve._act_pressed_kb] != DOWAAVE_COMMAND_BAO.default_factory() else "{} [no action]".format(repr(dwve._act_pressed_kb)))},
    '5,20' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : 'TIME:')},
    '6,20' : {'color':(lambda dwve : CWCN_COLORS.BLUE)      ,'lam':(lambda dwve : datetime.now())},
    '0,21' : {'color':(lambda dwve : CWCN_COLORS.REGULAR)   ,'lam':(lambda dwve : "--------------------------------------------------------------------------------------\n{}".format(dwve._message_buffer))},
}

# --- --- --- --- --- --- --- --- --- --- --- --- 
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# --- --- --- --- --- --- 
PLOT_RENDER_BAO = [
    # {
    #     'ID'    :(lambda dwve_state, render : 'waka'),
    #     'type'  :(lambda dwve_state, render : 'plot'),
    #     'figure':(lambda dwve_state, render : Figure(figsize=(4,2),facecolor="black")),
    #     'axis'  :(lambda dwve_state, render : render['figure'].add_subplot(111)),
    #     'facecolor':(lambda dwve_state, render : render['axis'].set_facecolor((0,0,0))),
    #     'title' :(lambda dwve_state, render : render['axis'].set_title("Estimation Grid", fontsize=8,color=(1,1,1))),
    #     'ylabel':(lambda dwve_state, render : render['axis'].set_ylabel("Y",fontsize=8)),
    #     'xlabel':(lambda dwve_state, render : render['axis'].set_xlabel("X",fontsize=8)),
    #     'plot'  :(lambda dwve_state, render : render['axis'].plot(dwve_state['x_vals'],dwve_state['y_vals'],color='red')),
    #     # 'plot2'  :(lambda dwve_state, render : render['axis'].plot(dwve_state['x_vals'],-dwve_state['y_vals'],color='blue')),
            # 'grid'  :(lambda dwve_state, render : render['axis'].grid(which='major',color='white',linestyle='-',linewidth=0.2)),
    #     'tick'  :(lambda dwve_state, render : render['axis'].tick_params(colors='red',which='both')),
    #     'canvas':(lambda dwve_state, render : FigureCanvasTkAgg(render['figure'],master=render['window'])),
    #     'pack'  :(lambda dwve_state, render : render['canvas'].get_tk_widget().pack()),
    #     'pos'   :(lambda dwve_state, render : render['canvas'].get_tk_widget().place(x=100, y=50)),
    #     'draw'  :(lambda dwve_state, render : render['canvas'].draw()),
    #     # 'update_clear'  :(lambda dwve_state, render : render['axis'].clear()),
    #     'update_x':(lambda dwve_state, render : render['plot'][0].set_xdata(dwve_state['x_vals'])),
    #     'update_y':(lambda dwve_state, render : render['plot'][0].set_ydata(dwve_state['y_vals'])),
    #     'update_relim':(lambda dwve_state, render : render['axis'].relim()),
    #     'update_autoscale':(lambda dwve_state, render : render['axis'].autoscale_view()),
    #     'update_draw'  :(lambda dwve_state, render : render['canvas'].draw()),
    #     'update_flush'  :(lambda dwve_state, render : render['canvas'].flush_events()),
    # },
    {
        'ID'    :'DOWAAVE_HRZ_F1',
        'type'  :(lambda dwve_state, render : 'plot'),
        'figure':(lambda dwve_state, render : Figure(facecolor="black")),
        'axis'  :(lambda dwve_state, render : render['figure'].add_subplot(111)),
        'facecolor':(lambda dwve_state, render : render['axis'].set_facecolor((0,0,0))),
        'title' :(lambda dwve_state, render : render['axis'].set_title("Estimation Grid", fontsize=8,color=(0,0,0))),
        # 'ylabel':(lambda dwve_state, render : render['axis'].set_ylabel("Y",fontsize=8)),
        # 'xlabel':(lambda dwve_state, render : render['axis'].set_xlabel("X",fontsize=8)),
        # 'plot2'  :(lambda dwve_state, render : render['axis'].plot(dwve_state['x_vals'],-dwve_state['y_vals'],color='blue')),
        # 'grid'  :(lambda dwve_state, render : render['axis'].grid(which='major',color='white',linestyle='-',linewidth=0.2)),
        'tick'  :(lambda dwve_state, render : render['axis'].tick_params(colors='black',which='both')),
        'canvas':(lambda dwve_state, render : FigureCanvasTkAgg(render['figure'],master=render['window'])),
        'pack'  :(lambda dwve_state, render : render['canvas'].get_tk_widget().pack()),
        'pos'   :(lambda dwve_state, render : render['canvas'].get_tk_widget().place(x=-50, y=-50)),
        'draw'  :(lambda dwve_state, render : render['canvas'].draw()),
        # 'update_clear'  :(lambda dwve_state, render : render['axis'].clear()),
        # 'update_x':(lambda dwve_state, render : render['plot'][0].set_xdata(dwve_state['x_vals'])),
        # 'update_y':(lambda dwve_state, render : render['plot'][0].set_ydata(dwve_state['y_vals'])),
        'update_img':(lambda dwve_state, render : render['axis'].imshow(dwve_state['DOWAAVE_HRZ_F1']) if dwve_state['DOWAAVE_HRZ_F1'] is not None else None),
        'update_autoscale':(lambda dwve_state, render : render['axis'].autoscale_view()),
        'update_draw'  :(lambda dwve_state, render : render['canvas'].draw()),
        'update_flush'  :(lambda dwve_state, render : render['canvas'].flush_events()),
    },
    # {
    #     'ID'    :'DOWAAVE_GSS_F1',
    #     'type'  :(lambda dwve_state, render : 'plot'),
    #     'figure':(lambda dwve_state, render : Figure(facecolor="black")),
    #     'axis'  :(lambda dwve_state, render : render['figure'].add_subplot(111)),
    #     'facecolor':(lambda dwve_state, render : render['axis'].set_facecolor((0,0,0))),
    #     'title' :(lambda dwve_state, render : render['axis'].set_title("Estimation Grid", fontsize=8,color=(0,0,0))),
    #     # 'ylabel':(lambda dwve_state, render : render['axis'].set_ylabel("Y",fontsize=8)),
    #     # 'xlabel':(lambda dwve_state, render : render['axis'].set_xlabel("X",fontsize=8)),
    #     # 'plot2'  :(lambda dwve_state, render : render['axis'].plot(dwve_state['x_vals'],-dwve_state['y_vals'],color='blue')),
    #     # 'grid'  :(lambda dwve_state, render : render['axis'].grid(which='major',color='white',linestyle='-',linewidth=0.2)),
    #     'tick'  :(lambda dwve_state, render : render['axis'].tick_params(colors='black',which='both')),
    #     'canvas':(lambda dwve_state, render : FigureCanvasTkAgg(render['figure'],master=render['window'])),
    #     'pack'  :(lambda dwve_state, render : render['canvas'].get_tk_widget().pack()),
    #     'pos'   :(lambda dwve_state, render : render['canvas'].get_tk_widget().place(x=-50, y=-50)),
    #     'draw'  :(lambda dwve_state, render : render['canvas'].draw()),
    #     # 'update_clear'  :(lambda dwve_state, render : render['axis'].clear()),
    #     # 'update_x':(lambda dwve_state, render : render['plot'][0].set_xdata(dwve_state['x_vals'])),
    #     # 'update_y':(lambda dwve_state, render : render['plot'][0].set_ydata(dwve_state['y_vals'])),
    #     'update_img':(lambda dwve_state, render : render['axis'].imshow(dwve_state['DOWAAVE_GSS_F1']) if dwve_state['DOWAAVE_GSS_F1'] is not None else None),
    #     'update_autoscale':(lambda dwve_state, render : render['axis'].autoscale_view()),
    #     'update_draw'  :(lambda dwve_state, render : render['canvas'].draw()),
    #     'update_flush'  :(lambda dwve_state, render : render['canvas'].flush_events()),
    # },
    {
        'ID'    :'DOWAAVE_GSS_F2',
        'type'  :(lambda dwve_state, render : 'plot'),
        'figure':(lambda dwve_state, render : Figure(facecolor="black")),
        'axis'  :(lambda dwve_state, render : render['figure'].add_subplot(111)),
        'facecolor':(lambda dwve_state, render : render['axis'].set_facecolor((0,0,0))),
        'title' :(lambda dwve_state, render : render['axis'].set_title("Estimation Grid", fontsize=8,color=(0,0,0))),
        # 'ylabel':(lambda dwve_state, render : render['axis'].set_ylabel("Y",fontsize=8)),
        # 'xlabel':(lambda dwve_state, render : render['axis'].set_xlabel("X",fontsize=8)),
        # 'plot2'  :(lambda dwve_state, render : render['axis'].plot(dwve_state['x_vals'],-dwve_state['y_vals'],color='blue')),
        # 'grid'  :(lambda dwve_state, render : render['axis'].grid(which='major',color='white',linestyle='-',linewidth=0.2)),
        'tick'  :(lambda dwve_state, render : render['axis'].tick_params(colors='black',which='both')),
        'canvas':(lambda dwve_state, render : FigureCanvasTkAgg(render['figure'],master=render['window'])),
        'pack'  :(lambda dwve_state, render : render['canvas'].get_tk_widget().pack()),
        'pos'   :(lambda dwve_state, render : render['canvas'].get_tk_widget().place(x=-50, y=335)),
        'draw'  :(lambda dwve_state, render : render['canvas'].draw()),
        # 'update_clear'  :(lambda dwve_state, render : render['axis'].clear()),
        # 'update_x':(lambda dwve_state, render : render['plot'][0].set_xdata(dwve_state['x_vals'])),
        # 'update_y':(lambda dwve_state, render : render['plot'][0].set_ydata(dwve_state['y_vals'])),
        'update_img':(lambda dwve_state, render : render['axis'].imshow(dwve_state['DOWAAVE_GSS_F2']) if dwve_state['DOWAAVE_GSS_F2'] is not None else None),
        'update_autoscale':(lambda dwve_state, render : render['axis'].autoscale_view()),
        'update_draw'  :(lambda dwve_state, render : render['canvas'].draw()),
        'update_flush'  :(lambda dwve_state, render : render['canvas'].flush_events()),
    },
    {
        'ID'    :'DOWAAVE_TFT_SHORT_F1',
        'type'  :(lambda dwve_state, render : 'plot'),
        'figure':(lambda dwve_state, render : Figure(facecolor="black")),
        'axis'  :(lambda dwve_state, render : render['figure'].add_subplot(111)),
        'facecolor':(lambda dwve_state, render : render['axis'].set_facecolor((0,0,0))),
        'title' :(lambda dwve_state, render : render['axis'].set_title("Estimation Grid", fontsize=8,color=(0,0,0))),
        # 'ylabel':(lambda dwve_state, render : render['axis'].set_ylabel("Y",fontsize=8)),
        # 'xlabel':(lambda dwve_state, render : render['axis'].set_xlabel("X",fontsize=8)),
        # 'plot2'  :(lambda dwve_state, render : render['axis'].plot(dwve_state['x_vals'],-dwve_state['y_vals'],color='blue')),
        # 'grid'  :(lambda dwve_state, render : render['axis'].grid(which='major',color='white',linestyle='-',linewidth=0.2)),
        'tick'  :(lambda dwve_state, render : render['axis'].tick_params(colors='black',which='both')),
        'canvas':(lambda dwve_state, render : FigureCanvasTkAgg(render['figure'],master=render['window'])),
        'pack'  :(lambda dwve_state, render : render['canvas'].get_tk_widget().pack()),
        'pos'   :(lambda dwve_state, render : render['canvas'].get_tk_widget().place(x=525, y=-50)),
        'draw'  :(lambda dwve_state, render : render['canvas'].draw()),
        # 'update_clear'  :(lambda dwve_state, render : render['axis'].clear()),
        # 'update_x':(lambda dwve_state, render : render['plot'][0].set_xdata(dwve_state['x_vals'])),
        # 'update_y':(lambda dwve_state, render : render['plot'][0].set_ydata(dwve_state['y_vals'])),
        'update_img':(lambda dwve_state, render : render['axis'].imshow(dwve_state['DOWAAVE_TFT_SHORT_F1']) if dwve_state['DOWAAVE_TFT_SHORT_F1'] is not None else None),
        'update_autoscale':(lambda dwve_state, render : render['axis'].autoscale_view()),
        'update_draw'  :(lambda dwve_state, render : render['canvas'].draw()),
        'update_flush'  :(lambda dwve_state, render : render['canvas'].flush_events()),
    },
    {
        'ID'    :'DOWAAVE_TFT_SHORT_F2',
        'type'  :(lambda dwve_state, render : 'plot'),
        'figure':(lambda dwve_state, render : Figure(facecolor="black")),
        'axis'  :(lambda dwve_state, render : render['figure'].add_subplot(111)),
        'facecolor':(lambda dwve_state, render : render['axis'].set_facecolor((0,0,0))),
        'title' :(lambda dwve_state, render : render['axis'].set_title("Estimation Grid", fontsize=8,color=(0,0,0))),
        # 'ylabel':(lambda dwve_state, render : render['axis'].set_ylabel("Y",fontsize=8)),
        # 'xlabel':(lambda dwve_state, render : render['axis'].set_xlabel("X",fontsize=8)),
        # 'plot2'  :(lambda dwve_state, render : render['axis'].plot(dwve_state['x_vals'],-dwve_state['y_vals'],color='blue')),
        'grid'  :(lambda dwve_state, render : render['axis'].grid(which='major',color='white',linestyle='-',linewidth=0.2)),
        'tick'  :(lambda dwve_state, render : render['axis'].tick_params(colors='black',which='both')),
        'canvas':(lambda dwve_state, render : FigureCanvasTkAgg(render['figure'],master=render['window'])),
        'pack'  :(lambda dwve_state, render : render['canvas'].get_tk_widget().pack()),
        'pos'   :(lambda dwve_state, render : render['canvas'].get_tk_widget().place(x=525, y=335)),
        'draw'  :(lambda dwve_state, render : render['canvas'].draw()),
        # 'update_clear'  :(lambda dwve_state, render : render['axis'].clear()),
        # 'update_x':(lambda dwve_state, render : render['plot'][0].set_xdata(dwve_state['x_vals'])),
        # 'update_y':(lambda dwve_state, render : render['plot'][0].set_ydata(dwve_state['y_vals'])),
        'update_img':(lambda dwve_state, render : render['axis'].imshow(dwve_state['DOWAAVE_TFT_SHORT_F2']) if dwve_state['DOWAAVE_TFT_SHORT_F2'] is not None else None),
        'update_autoscale':(lambda dwve_state, render : render['axis'].autoscale_view()),
        'update_draw'  :(lambda dwve_state, render : render['canvas'].draw()),
        'update_flush'  :(lambda dwve_state, render : render['canvas'].flush_events()),
    },

]
# --- --- --- --- --- --- 


