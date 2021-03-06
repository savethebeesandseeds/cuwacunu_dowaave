import sys
sys.path.append('./communications')
import asyncio
# --- --- --- --- 
import os
import ast
import json
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
assert_folder(dwvcc.CWCN_FRONT_CONFIG.FRONT_WALLET_FOLDER)
# --- --- --- --- --- --- --- 
if __name__ == '__main__':
    # --- --- --- --- --- --- --- 
    async def wait_forever(): # async wait 
        while True:
            await asyncio.sleep(30000)
    # --- --- --- --- --- --- --- 
    # --- --- --- --- --- --- --- 
    wallet_subs=[]
    instrument_subs=[]
    instrument_subs+=['/contract/position:{}'.format(__) for __ in dwvcc.ACTIVE_SYMBOLS]
    instrument_subs+=['/contractMarket/level2:{}'.format(__) for __ in dwvcc.ACTIVE_SYMBOLS]
    wallet_subs+=['/contractAccount/wallet']
    c_trade_instrument = poloniex_api.EXCHANGE_INSTRUMENT(
        # _message_wrapper_=front_meesage_wrapper, 
        _websocket_subs=wallet_subs+instrument_subs,
        _is_farm=False,
        _is_front=True)
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    # print(
    #     json.dumps(c_trade_instrument.market_instrument.get_l2_order_book('BTCUSDTPERP'),indent=4)
    # )
    # print(c_trade_instrument.market_instrument.get_l2_order_book('BTCUSDTPERP'))
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
    loop=asyncio.get_event_loop()
    loop.run_until_complete(wait_forever())