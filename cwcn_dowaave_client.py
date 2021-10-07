# cwcn_dowaave_client, by waajacu.
# --- --- --- --- --- --- ---
# --- --- --- 
import os
from flask import Flask
from flask import request
from flask import jsonify
from flask import send_from_directory
# --- --- --- --- --- --- ---
# --- --- --- 
import threading
import sys
sys.path.append('./communications/')
import poloniex_api
# --- --- --- --- --- --- ---
# --- --- --- 
import cwcn_dwve_client_config as dwvcc
os.environ['FLASK_ENV']='development'
os.environ['FLASK_APP']='flaskr'
app = Flask(__name__)
# --- --- --- --- --- --- ---
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
class KALAMAR_CLIENT:
    def __init__(self):
        # --- --- --- --- --- 
        self.dowaave_user_data = {}
        # --- --- --- --- --- 
        self.dowaave_instruments_data=dict([(__,{}) for __ in dwvcc.ACTIVE_SYMBOLS])
        self.dowaave_user_data=dict([(__,{}) for __ in list(dwvcc.VALID_NODES.keys())])
        # --- --- --- --- --- 
        self.app=FlaskAppWrapper('KALAMAR')
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
        print(dwvcc.WEB_SOCKET_SUBS)
        def start_echance_instrument():
            self.exchange_instrument = poloniex_api.EXCHANGE_INSTRUMENT(
                _message_wrapper_=self.dowaave_ujcamei, 
                _websocket_subs=dwvcc.WEB_SOCKET_SUBS,
                _is_farm=False)
        threading.Thread(target=start_echance_instrument).start()
        # --- --- --- --- --- 
        
        # --- --- --- --- --- 
        # import time
        # time.sleep(30)
        # import asyncio
        # async def wait():
        #     await asyncio.sleep(30000)
        # loop=asyncio.get_event_loop()
        # loop.run_until_complete(wait())
    # --- --- --- --- --- --- --- --- --- --- --- 
    def dowaave_home(self,**args):
        try:
            if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
                req_addr=request.environ['REMOTE_ADDR']
            else: # if behind a proxy
                req_addr=request.environ['HTTP_X_FORWARDED_FOR']
            req_split=request.environ['QUERY_STRING'].split(',')
            req_symbol=''.join([chr(int(__,2)) for __ in [_ for _ in req_split if 'symbol' in _][0].split('=')[1].split('%20')])
            req_node=''.join([chr(int(__,2)) for __ in [_ for _ in req_split if 'node' in _][0].split('=')[1].split('%20')])
            req_command=''.join([chr(int(__,2)) for __ in [_ for _ in req_split if 'command' in _][0].split('=')[1].split('%20')])
            req_symbol=RCsi_CRYPT(req_addr,req_symbol)
            req_node=RCsi_CRYPT(req_addr,req_node)
            req_command=RCsi_CRYPT(req_addr[::-1],req_command)
            print("symbol   : {}".format(repr(req_symbol)))
            print("node     : {}".format(repr(req_node)))
            print("command  : {}".format(repr(req_command)))
            if(req_node in list(dwvcc.VALID_NODES.keys())):
                if('MESSAGE' in req_command):
                    if('\x00\x84]H' in req_command): # update terminal by direct request (slow)
                        aux_1 = self.exchange_instrument.user_instrument.get_account_overview() # get_account_overview
                        aux_2 = self.exchange_instrument.trade_instrument.get_position_details(req_symbol) # get_position_details
                        aux_1.update(aux_2)
                        aux_1={k:v for k,v in aux_1.items() if k in dwvcc.AHDO_FIELDS}
                        self.dowaave_user_data[dwvcc.VALID_NODES[req_node]].update(aux_1)
                        # import json
                        # print(json.dumps(self.dowaave_user_data,indent=4))
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- 
                        crypt_resp = {}
                        for _yield_key in list(self.dowaave_user_data.keys()):
                            crypt_resp[_yield_key]=str(RCsi_CRYPT(request.url,"{}:{}".format(_yield_key,self.dowaave_user_data[dwvcc.VALID_NODES[req_node]][_yield_key]))+"%20").encode('utf-8')
                        def generate():
                            for _yield_key in list(self.dowaave_user_data.keys()):
                                yield crypt_resp[_yield_key]
                        return app.response_class(generate(), content_type='text/encoded; charset=utf-8')
                        # --- --- --- --- --- --- --- --- --- --- --- --- --- 
                    elif('ÄZ¬\x8e\\\x88$Æ' in req_command): # update gui graphs by direct request
                        def generate():
                            # for row in iter_all_rows():
                            yield 1 # ... 
                        return app.response_class(generate(), content_type='text/encoded; charset=utf-8')
                return '%20'.join(format(ord(XXX), 'b') for XXX in RCsi_CRYPT(request.url,'\"Hello World!\"'))
            else:
                return 'DENIED!', 402
        except:
            return 'ERROR!', 400
    def dowaave_transfomations(self):
        self.dowaave_instruments_data['date']=datetime.datetime.fromtimestamp(self.dowaave_instruments_data['ts']/1000000000)
    def dowaave_ujcamei(self,msg):
        # {'data': {'symbol': '1000SHIBUSDTPERP', 'sequence': 1621480496894, 'side': 'sell', 'size': 2, 'price': 0.006996, 'bestBidSize': 3, 'bestBidPrice': '0.006996', 'bestAskPrice': '0.007008', 'tradeId': '6154ed2ed8d9f225c859eae3', 'ts': 1632955698027907792, 'bestAskSize': 150}, 'subject': 'ticker', 'topic': '/contractMarket/ticker:1000SHIBUSDTPERP', 'type': 'message'}
        print("dowaave_ujcamei:\t",msg)
        if('topic' in list(msg.keys()) and '/contractMarket/ticker' in msg['topic']):
            aux_1={k:v for k,v in msg['data'].items() if k in dwvcc.AHDO_FIELDS}
            self.dowaave_instruments_data[msg['data']['symbol']].update(aux_1)
            print("updated dowaave_instruments_data for symbol instrument : {} -> {}".format(msg['data']['symbol'], self.dowaave_instruments_data))

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
    klrm_ctl=KALAMAR_CLIENT()
# --- --- --- --- --- --- --- 