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
# --- --- --- --- --- --- --- 
if __name__ == '__main__':
    # --- --- --- --- --- --- --- 
    async def wait_forever(): # async wait 
        while True:
            await asyncio.sleep(30000)
    print("[ERROR:] farm does not delate the files once backup is made?")        
    # --- --- --- --- --- --- --- 
    # --- --- --- --- --- --- --- 
    c_trade_instrument = poloniex_api.EXCHANGE_INSTRUMENT(
        # _message_wrapper_=_farm_on_message_, 
        # _websocket_subs=None, # not needed when is only farm
        _is_farm=True,
        _is_front=False)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    loop=asyncio.get_event_loop()
    loop.run_until_complete(wait_forever())