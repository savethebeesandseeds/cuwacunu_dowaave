import sys
sys.path.append('./communications')
import asyncio
# --- --- --- --- 
import os
import ast
import poloniex_api
import cwcn_dwve_client_config as dwvcc
# --- --- --- --- --- --- --- 
# --- --- --- --- # The front has the data update, the front has the wallet updates
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
def assert_folder(_f_path):
    if(not os.path.isdir(_f_path)):
        os.mkdir(_f_path)
assert_folder(os.path.dirname(dwvcc.CWCN_FARM_CONFIG.FRONT_WALLET_FILE))
# --- --- --- --- --- --- --- 
if __name__ == '__main__':
    # --- --- --- --- --- --- --- 
    async def wait_forever(): # async wait 
        while True:
            await asyncio.sleep(30000)
    # --- --- --- --- --- --- --- 
    def _front_on_message_(self,msg):
        if("/contractAccount/wallet" in msg['topic'] or "/contract/position" in msg['topic']):
            try:
                c_wall={}
                if(os.path.isfile(dwvcc.CWCN_FARM_CONFIG.FRONT_WALLET_FILE)):
                    with open(dwvcc.CWCN_FARM_CONFIG.FRONT_WALLET_FILE,"r",encoding='utf-8') as _F:
                        # c_wall = RCsi_CRYPT('shallowsecurewallet',_F.read())
                        c_decoded=[chr(int(__)) for __ in _F.read().split(',')]
                        c_decoded=''.join(c_decoded)
                        c_wall = RCsi_CRYPT('shallowsecurewallet',c_decoded)
                    c_wall = ast.literal_eval(c_wall)
                c_wall.update(msg['data'])
                print("[UPDATE:] c_wall : {}".format(c_wall))
                with open(dwvcc.CWCN_FARM_CONFIG.FRONT_WALLET_FILE,"w+",encoding='utf-8') as _F:
                    # _F.write("{}".format(RCsi_CRYPT('shallowsecurewallet','{}'.format(c_wall))))
                    c_encoded = "{}".format(RCsi_CRYPT('shallowsecurewallet','{}'.format(c_wall)))
                    c_encoded = ','.join([str(ord(__)) for __ in c_encoded])
                    _F.write(c_encoded)
                # print_method(f'Get {msg["data"]["symbol"]} Ticket :{msg["data"]} : unix time : {time.time()}')
                sys.stdout.write(dwvcc.CWCN_CURSOR.CARRIER_RETURN)
                sys.stdout.write(dwvcc.CWCN_CURSOR.CLEAR_LINE)
                sys.stdout.write('WALLET UPDATE : {}'.format(msg['data']))
                sys.stdout.write(dwvcc.CWCN_CURSOR.CARRIER_RETURN)
                sys.stdout.flush()
            except Exception as e:
                print("FORNT ERROR! {}".format(e))
    # --- --- --- --- --- --- --- 
    c_trade_instrument = poloniex_api.EXCHANGE_INSTRUMENT(
        _message_wrapper_=_front_on_message_, 
        _websocket_subs=['/contractAccount/wallet', '/contract/position:{}'.format(dwvcc.dwve_instrument_configuration.SYMBOL)],
        _is_farm=True)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    loop=asyncio.get_event_loop()
    loop.run_until_complete(wait_forever())