# --- --- --- 
# close_all_positions.py
# --- --- ---  
import sys
sys.path.append('./communications')
import poloniex_api
# --- --- ---  
import cwcn_dwve_client_config
# --- --- --- --- --- 
if(__name__=='__main__'):
    # --- --- --- --- s
    c_trade_instrument = poloniex_api.EXCHANGE_INSTRUMENT(_is_farm=False)
    print("clear_positions:")
    c_trade_instrument.trade_instrument.clear_positions(
        cwcn_dwve_client_config.dwve_instrument_configuration.LEVERAGE,cwcn_dwve_client_config.dwve_instrument_configuration.SYMBOL)