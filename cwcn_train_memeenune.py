# --- --- --- --- 
import cwcn_dowaave_memeenune as dwvmm
from cwcn_dwve_client_config import dwve_instrument_configuration as dwvc
# --- --- --- --- 
if __name__=='__main__':
    print("[WARNING] lower the horizont of prediction")
    c_memeenune = dwvmm.MEMEENUNE()
    working_dataframe = c_memeenune.c_data_kijtyu.load_and_get_dataframe(
        dwvc.train_load_n_seq
    )
    c_memeenune.train_tff(
        working_dataframe,
        reset=False, # meaning to reset the model or True is one wants to train the current model
    )
    # input()
    # working_dataframe = c_memeenune.c_data_kijtyu.load_and_get_dataframe(
    #     max(dwvc.gss_c_seq_size,dwvc.tft_c_seq_size)
    # )
    # c_memeenune.launch_uwaabo(working_dataframe)