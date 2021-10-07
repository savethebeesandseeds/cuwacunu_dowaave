import sys
sys.path.append('./communications')
import asyncio
# --- --- --- --- 
import poloniex_api
# --- --- --- --- 
if __name__ == '__main__':
    async def wait():
        while True:
            await asyncio.sleep(30000)
    
    c_trade_instrument = poloniex_api.EXCHANGE_INSTRUMENT(_is_farm=True)
    loop=asyncio.get_event_loop()
    loop.run_until_complete(wait())