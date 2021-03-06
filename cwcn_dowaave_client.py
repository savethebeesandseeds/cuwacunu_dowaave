# cwcn_dowaave_client, by waajacu.
# --- --- --- --- --- --- ---
# --- --- ---
import os
from flask import Flask
from flask import request
from flask import jsonify
from flask import send_from_directory
import time
import threading
import ast
import sys
sys.path.append('./communications/')
import json
# --- --- --- --- --- --- ---
# --- --- ---
import poloniex_api
import cwcn_dwve_client_config as dwvcc
import cwcn_dowaave_memeenune as dwvmm
os.environ['FLASK_ENV']='development'
os.environ['FLASK_APP']='flaskr'
app = Flask(__name__)
# --- --- --- --- --- --- ---
# --- --- --- --- --- --- ---
def RCsi_CRYPT(key, data): # tehdujco, !
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
# --- --- --- --- --- --- ---
class DOWAAVE_CLIENT:
    def __init__(self):
        # --- --- --- --- ---
        self.actual_strategy_index=dict([(dwvcc.VALID_NODES[__],dict([(___,0) for ___ in dwvcc.ACTIVE_SYMBOLS])) for __ in list(dwvcc.VALID_NODES.keys())])
        self.actual_strategy=dict([(dwvcc.VALID_NODES[__],dict([(___,dwvcc.ACTIVE_STRATEGIES[self.actual_strategy_index[dwvcc.VALID_NODES[__]][___]]) for ___ in dwvcc.ACTIVE_SYMBOLS])) for __ in list(dwvcc.VALID_NODES.keys())])
        # --- --- --- --- ---
        self.actual_instrument_index=dict([(dwvcc.VALID_NODES[__],0) for __ in list(dwvcc.VALID_NODES.keys())])
        self.actual_instrument=dict([(dwvcc.VALID_NODES[__],dwvcc.ACTIVE_SYMBOLS[self.actual_instrument_index[dwvcc.VALID_NODES[__]]]) for __ in list(dwvcc.VALID_NODES.keys())])
        # --- --- --- --- ---
        self.actual_leverage=dict([(dwvcc.VALID_NODES[__],dwvcc.dwve_instrument_configuration.__dict__[self.actual_instrument[dwvcc.VALID_NODES[__]]].LEVERAGE) for __ in list(dwvcc.VALID_NODES.keys())])
        self.actual_order_size=dict([(dwvcc.VALID_NODES[__],dwvcc.dwve_instrument_configuration.__dict__[self.actual_instrument[dwvcc.VALID_NODES[__]]].ORDER_SIZE) for __ in list(dwvcc.VALID_NODES.keys())])
        # --- --- --- --- ---

        # --- --- --- --- ---
        self.c_memeenune = dwvmm.MEMEENUNE()
        try:
            self.c_memeenune.launch_uwaabo(force=True)
        except Exception as e:
            print("ERROR : unable to launch initial uwaabo, {}".format(e))
        # --- --- --- --- ---
        self.l2_market=None
        self.dowaave_user_data = {}
        self.dowaave_instruments_data = {}
        # --- --- --- --- ---
        self.dowaave_instruments_data=dict([(__,{}) for __ in dwvcc.ACTIVE_SYMBOLS])
        self.dowaave_user_data=dict([(dwvcc.VALID_NODES[__],dict([(___,{'activeStrategy':self.actual_strategy[dwvcc.VALID_NODES[__]][___]}) for ___ in dwvcc.ACTIVE_SYMBOLS])) for __ in list(dwvcc.VALID_NODES.keys())])
        self.dowaave_user_tian=dict([(dwvcc.VALID_NODES[__],{'daotime':time.time(),'taotime':time.time()}) for __ in list(dwvcc.VALID_NODES.keys())])
        # --- --- --- --- ---
        self.app=FlaskAppWrapper('DOWAAVE')
        self.app.add_endpoint(
            endpoint='/',#/<path:path>',
            endpoint_name='dowaave_home',
            handler=self.dowaave_home
        )
        # --- --- --- --- ---
        def start_server():
            return self.app.run(
                host=dwvcc.CLIENT_DIR,
                port=dwvcc.CLIENT_PORT,
                debug=False,
                use_reloader=False,
                # load_dotenv=True
            )
        threading.Thread(target=start_server).start()
        # --- --- --- --- ---
        # print(dwvcc.WEB_SOCKET_SUBS)
        def start_echance_instrument():
            self.exchange_instrument = poloniex_api.EXCHANGE_INSTRUMENT(_is_farm=False)
        threading.Thread(target=start_echance_instrument).start()
        # --- --- --- --- ---
        # --- --- --- --- --- --- ---
        # --- --- --- --- ---
        # import time
        # time.sleep(30)
        # import asyncio
        # async def wait():
        #     await asyncio.sleep(30000)
        # loop=asyncio.get_event_loop()
        # loop.run_until_complete(wait())
    # --- --- --- --- --- --- --- --- --- --- ---
    def _change_instrument(self,req_node):
        self.actual_instrument_index[dwvcc.VALID_NODES[req_node]]+=1
        if(self.actual_instrument_index[dwvcc.VALID_NODES[req_node]]>=len(dwvcc.ACTIVE_SYMBOLS)):
            self.actual_instrument_index[dwvcc.VALID_NODES[req_node]]=0
        self.actual_instrument[dwvcc.VALID_NODES[req_node]]=dwvcc.ACTIVE_SYMBOLS[self.actual_instrument_index[dwvcc.VALID_NODES[req_node]]]
        self.actual_leverage[dwvcc.VALID_NODES[req_node]]=dwvcc.dwve_instrument_configuration.__dict__[self.actual_instrument[dwvcc.VALID_NODES[req_node]]].LEVERAGE
        self.actual_order_size[dwvcc.VALID_NODES[req_node]]=dwvcc.dwve_instrument_configuration.__dict__[self.actual_instrument[dwvcc.VALID_NODES[req_node]]].ORDER_SIZE
        self.dowaave_user_data[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]['activeStrategy']=self.actual_strategy[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]
    def _change_strategy(self,req_node):
        print("change 1")
        self.actual_strategy_index[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]+=1
        print("change 2")
        if(self.actual_strategy_index[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]>=len(dwvcc.ACTIVE_STRATEGIES)):
            print("change 3")
            self.actual_strategy_index[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]=0
        print("change 4")
        self.actual_strategy[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]=dwvcc.ACTIVE_STRATEGIES[self.actual_strategy_index[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]]
        print("change 5")
        print(self.actual_strategy)
        print("change 6")
        self.dowaave_user_data[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]['activeStrategy']=self.actual_strategy[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]]
        print("change 7")
    # --- --- --- --- --- --- --- --- --- --- ---
    def update_terminal_bao(self,req_command,req_node):
        print("update_terminal_bao: {}:[{}] {}".format(req_node,self.actual_instrument[dwvcc.VALID_NODES[req_node]],repr(req_command)))
        c_file = "{}/WALLET.{}.poloniex_wallet_data".format(dwvcc.CWCN_FRONT_CONFIG.FRONT_WALLET_FOLDER,self.actual_instrument[dwvcc.VALID_NODES[req_node]])
        try:
            if('HEAVYLOAD' in req_command):
                prime_1 = self.exchange_instrument.user_instrument.get_account_overview() # get_account_overview
                prime_2 = self.exchange_instrument.trade_instrument.get_position_details(self.actual_instrument[dwvcc.VALID_NODES[req_node]]) # get_position_details
            else:
                prime_1  = {}
            if(os.path.isfile(c_file)):
                c_wall={}
                with open(c_file,"r",encoding='utf-8') as _F:
                    # c_wall = RCsi_CRYPT('shallowsecurewallet',_F.read())
                    c_decoded=[chr(int(__)) for __ in _F.read().split(',')]
                    c_decoded=''.join(c_decoded)
                    c_wall = RCsi_CRYPT('shallowsecurewallet',c_decoded)
                    c_wall = c_wall[c_wall.find("{"):c_wall.rfind("}")+1]
                # print(repr(c_wall))
                c_wall = ast.literal_eval(c_wall)
                prime_1.update(c_wall)
            if('HEAVYLOAD' in req_command):
                prime_1.update(prime_2)
            prime_1.update({'price':self.c_memeenune.c_data_kijtyu.get_c_tick(symbol=self.actual_instrument[dwvcc.VALID_NODES[req_node]])['price']})
            with open(c_file,"w+",encoding='utf-8') as _F:
                # _F.write("{}".format(RCsi_CRYPT('shallowsecurewallet','{}'.format(prime_1))))
                c_encoded = "{}".format(RCsi_CRYPT('shallowsecurewallet','{}'.format(prime_1)))
                c_encoded = ','.join([str(ord(__)) for __ in c_encoded])
                _F.write(c_encoded)
            prime_1={k:v for k,v in prime_1.items() if k in dwvcc.AHDO_FIELDS}
            self.dowaave_user_data[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]].update(prime_1)
            # print(json.dumps(self.dowaave_user_data,indent=4))
        except Exception as e:
            print("problem updating terminal bao: {}".format(e))
        # --- --- --- --- --- --- --- --- --- --- --- --- ---
        try:
            crypt_resp = {}
            for _yield_key in list(self.dowaave_user_data[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]].keys()):
                crypt_resp[_yield_key]=str(RCsi_CRYPT(request.url,"{}:{}".format(_yield_key,self.dowaave_user_data[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]][_yield_key]))+"%20").encode('utf-8')
            def generate():
                for _yield_key in list(self.dowaave_user_data[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]].keys()):
                    yield crypt_resp[_yield_key]
            return app.response_class(generate(), content_type='text/encoded; charset=utf-8')
            # --- --- --- --- --- --- --- --- --- --- --- --- ---
        except Exception as e:
            print("problem responding updated terminal bao: {}".format(e))

    def update_plot_bao(self,req_command,req_node):
        print("update_plot_bao: {}:[{}] {}".format(req_node,self.actual_instrument[dwvcc.VALID_NODES[req_node]],repr(req_command)))
        os.environ['DOWAAVE_GSS_F1']="./gauss_dumps/{}/{}.0.png".format(self.actual_instrument[dwvcc.VALID_NODES[req_node]],self.actual_instrument[dwvcc.VALID_NODES[req_node]])
        os.environ['DOWAAVE_GSS_F2']="./gauss_dumps/{}/{}.0.png".format(self.actual_instrument[dwvcc.VALID_NODES[req_node]],self.actual_instrument[dwvcc.VALID_NODES[req_node]])
        os.environ['DOWAAVE_TFT_SHORT_F1']="./tft_dumps/{}/{}.s0.png".format(self.actual_instrument[dwvcc.VALID_NODES[req_node]],self.actual_instrument[dwvcc.VALID_NODES[req_node]])
        os.environ['DOWAAVE_TFT_SHORT_F2']="./tft_dumps/{}/{}.s1.png".format(self.actual_instrument[dwvcc.VALID_NODES[req_node]],self.actual_instrument[dwvcc.VALID_NODES[req_node]])
        os.environ['DOWAAVE_TFT_LONG_F1']="./tft_dumps/{}/{}.l0.png".format(self.actual_instrument[dwvcc.VALID_NODES[req_node]],self.actual_instrument[dwvcc.VALID_NODES[req_node]])
        os.environ['DOWAAVE_TFT_LONG_F2']="./tft_dumps/{}/{}.l1.png".format(self.actual_instrument[dwvcc.VALID_NODES[req_node]],self.actual_instrument[dwvcc.VALID_NODES[req_node]])
        os.environ['DOWAAVE_HRZ_F1']="./hrz_dumps/{}/{}.0.png".format(self.actual_instrument[dwvcc.VALID_NODES[req_node]],self.actual_instrument[dwvcc.VALID_NODES[req_node]])
        try:
            if('HEAVYLOAD' in req_command):
                ret_flag=self.c_memeenune.launch_uwaabo(
                    symbol=self.actual_instrument[dwvcc.VALID_NODES[req_node]],
                    force=True,
                    gss_flag=dwvcc.launch_uwaabo_gss,
                    tft_flag=dwvcc.launch_uwaabo_tft)
            else:
                ret_flag=self.c_memeenune.launch_uwaabo(
                    symbol=self.actual_instrument[dwvcc.VALID_NODES[req_node]],
                    force=False,
                    gss_flag=dwvcc.launch_uwaabo_gss,
                    tft_flag=dwvcc.launch_uwaabo_tft)
        except Exception as e:
            print("Problem launching uwaabo : (data might not be ready) : {}".format(e))
            raise Exception(str(e))
        crypt_resp={}
        if(True): # #FIXME add ret_flag
            for _yield_key in [_k for _k in list(os.environ.keys()) if _k in ['DOWAAVE_GSS_F1','DOWAAVE_GSS_F2','DOWAAVE_TFT_SHORT_F1','DOWAAVE_TFT_SHORT_F2','DOWAAVE_HRZ_F1']]:
                try:
                    with open(os.environ[_yield_key],'rb') as _F:
                        c_loaded_file=_F.read()
                    # print(repr(c_loaded_file))
                    c_encoded = ','.join([str(ord(__)) for __ in c_loaded_file.decode('iso-8859-1')])
                    c_encoded='{}:::::'.format(_yield_key)+str(RCsi_CRYPT(request.url,"{}".format(c_encoded)))+"%20"
                    # c_encoded=str(RCsi_CRYPT(request.url,"{}".format(c_loaded_file.decode('iso-8859-1'))))+"%20"
                    crypt_resp[_yield_key]=c_encoded
                except Exception as e:
                    print("problem updating <{}> plot bao: {}".format(_yield_key,e))
        # --- --- --- --- --- --- --- --- --- --- --- --- ---
        try:
            if(len(list(crypt_resp.keys()))>0):
                delta_x = 64
                def generate():
                    for _yield_key in list(crypt_resp.keys()):
                        for _yield_beat in [crypt_resp[_yield_key][x:x+delta_x] for x in range(0,len(crypt_resp[_yield_key]),delta_x)]:
                            yield _yield_beat
                return app.response_class(generate(), content_type='text/encoded; charset=utf-8')
            else:
                return app.response_class("None", content_type='text/encoded; charset=utf-8')
        except Exception as e:
            print("problem responding updated plot bao: {}".format(e))

    def proceed_bao(self,req_command,req_node):
        print("proceed_bao: {}:[{}] {}".format(req_node,self.actual_instrument[dwvcc.VALID_NODES[req_node]],repr(req_command)))
        if('BUY' in req_command and not dwvcc.PAPER_INSTRUMENT):
            order_data=self.exchange_instrument.trade_instrument.create_market_order(
                symbol=self.actual_instrument[dwvcc.VALID_NODES[req_node]],
                side='buy',
                size=self.actual_order_size[dwvcc.VALID_NODES[req_node]],
                leverage=self.actual_leverage[dwvcc.VALID_NODES[req_node]])
            print(json.dumps(order_data,indent=4))
        if('SELL' in req_command and not dwvcc.PAPER_INSTRUMENT):
            order_data=self.exchange_instrument.trade_instrument.create_market_order(
                symbol=self.actual_instrument[dwvcc.VALID_NODES[req_node]],
                side='sell',
                size=self.actual_order_size[dwvcc.VALID_NODES[req_node]],
                leverage=self.actual_leverage[dwvcc.VALID_NODES[req_node]])
            print(json.dumps(order_data,indent=4))
        if('CLOSE' in req_command):
            self.exchange_instrument.trade_instrument.clear_positions(
                LEVERAGE=self.actual_leverage[dwvcc.VALID_NODES[req_node]],
                SYMBOL=self.actual_instrument[dwvcc.VALID_NODES[req_node]])
            print("Closed all positions for symbol: {} with leverage: {}".format(self.actual_instrument[dwvcc.VALID_NODES[req_node]],self.actual_leverage[dwvcc.VALID_NODES[req_node]]))
        if('CHANGE_INSTRUMENT' in req_command):
            self._change_instrument(req_node=req_node)
            self.update_terminal_bao(req_command='HEAVYLOAD',req_node=req_node)
            self.update_plot_bao(req_command='KUAILOAD',req_node=req_node)
        if('CHANGE_STRATEGY' in req_command):
            self._change_strategy(req_node=req_node)
            if(self.actual_strategy[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]] == 'Lu'):
                self.l2_market_bao(req_node=req_node)
                self.dowaave_strategy_lu(req_node=req_node)
        return "completed!", 200

    def l2_market_bao(self,req_node):
        try:
            c_wall={}
            c_file = "{}/MRKT.{}.poloniex_l2_mrkt_data".format(dwvcc.CWCN_FRONT_CONFIG.FRONT_MARKET_FOLDER,self.actual_instrument[dwvcc.VALID_NODES[req_node]])
            if(os.path.isfile(c_file)):
                with open(c_file,"r",encoding='utf-8') as _F:
                    # c_wall = RCsi_CRYPT('shallowsecuremarket',_F.read())
                    c_decoded=[chr(int(__)) for __ in _F.read().split(',')]
                    c_decoded=''.join(c_decoded)
                    c_wall = RCsi_CRYPT('shallowsecuremarket',c_decoded)
                    c_wall = c_wall[c_wall.find("{"):c_wall.rfind("}")+1]
            # print(repr(c_wall))
            self.l2_market = ast.literal_eval(c_wall)
            # print(self.l2_market)
        except Exception as e:
            print("problem responding updated l2_market bao: {}".format(e))

    def dowaave_home(self,**args):
        # --- --- --- ---
        try:
            if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
                req_addr=request.environ['REMOTE_ADDR']
            else: # if behind a proxy
                req_addr=request.environ['HTTP_X_FORWARDED_FOR']
            req_split=request.environ['QUERY_STRING'].split(',')
            # req_symbol=''.join([chr(int(__,2)) for __ in [_ for _ in req_split if 'symbol' in _][0].split('=')[1].split('%20')])
            # req_symbol=RCsi_CRYPT(req_addr,req_symbol)
            req_node=''.join([chr(int(__,2)) for __ in [_ for _ in req_split if 'node' in _][0].split('=')[1].split('%20')])
            req_node=RCsi_CRYPT(req_addr,req_node)
            req_command=''.join([chr(int(__,2)) for __ in [_ for _ in req_split if 'command' in _][0].split('=')[1].split('%20')])
            req_command=RCsi_CRYPT(req_addr[::-1],req_command)
            print("symbol   : {}".format(repr(self.actual_instrument[dwvcc.VALID_NODES[req_node]])))
            print("command  : {}".format(repr(req_command)))
            print("node     : {} -> {}".format(repr(req_node),dwvcc.VALID_NODES[req_node]))
            if(req_node in list(dwvcc.VALID_NODES.keys())):
                if(self.actual_strategy[dwvcc.VALID_NODES[req_node]][self.actual_instrument[dwvcc.VALID_NODES[req_node]]] == 'Lu'):
                    self.l2_market_bao(req_node=req_node)
                    self.dowaave_strategy_lu(req_node=req_node)
                # --- --- --- --- --- --- --- 
                if('\x00\x84]H' in req_command): # update terminal by direct request (slow)
                    c_return=self.update_terminal_bao(req_command,req_node)
                elif('??Z??\x8e\\\x88$??' in req_command): # update gui graphs by direct request (slow)
                    c_return=self.update_plot_bao(req_command,req_node)
                elif('??Q??I??5' in req_command): # procedure action
                    c_return=self.proceed_bao(req_command,req_node)
                else:
                    return '%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(request.url,'COMMAND ERROR'))
                return c_return
            else:
                return 'DENIED!', 402
        except Exception as e:
            print("Error on dowaave_home : {}".format(e))
            return 'ERROR!', 400
    # def dowaave_transfomations(self):
    #     self.dowaave_instruments_data[...]['date']=datetime.datetime.fromtimestamp(self.dowaave_instruments_data['ts']/1000000000)
    def dowaave_ujcamei(self,msg): #FIXME not in use
        # {'data': {'symbol': '1000SHIBUSDTPERP', 'sequence': 1621480496894, 'side': 'sell', 'size': 2, 'price': 0.006996, 'bestBidSize': 3, 'bestBidPrice': '0.006996', 'bestAskPrice': '0.007008', 'tradeId': '6154ed2ed8d9f225c859eae3', 'ts': 1632955698027907792, 'bestAskSize': 150}, 'subject': 'ticker', 'topic': '/contractMarket/ticker:1000SHIBUSDTPERP', 'type': 'message'}
        print("dowaave_ujcamei:\t",msg)
        if('topic' in list(msg.keys()) and '/contractMarket/ticker' in msg['topic']):
            aux_1={k:v for k,v in msg['data'].items() if k in dwvcc.AHDO_FIELDS}
            self.dowaave_instruments_data[msg['data']['symbol']].update(aux_1)
            print("updated dowaave_instruments_data for symbol instrument : {} -> {}".format(msg['data']['symbol'], self.dowaave_instruments_data))
    
    def dowaave_strategy_lu(self,req_node):
        print("PLACE orders FIXME implement")
# --- --- --- --- --- --- ---
# references to : https://stackoverflow.com/a/40466535/13654027
class EndpointAction(object):
    def __init__(self,action):
        self.action=action
    def __call__(self,**args):
        return self.action(**args)
class FlaskAppWrapper(object):
    def __init__(self, name):
        self.app=Flask(name)
    def run(self,**args):
        self.app.run(**args)
    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
        self.app.add_url_rule(endpoint,endpoint_name, EndpointAction(handler), methods=['GET'])
# --- --- --- --- --- --- ---
if __name__ == '__main__':
    klrm_ctl=DOWAAVE_CLIENT()
# --- --- --- --- --- --- ---