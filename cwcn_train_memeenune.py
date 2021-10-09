# --- --- --- --- 
import cwcn_dowaave_memeenune as dwvmm
# --- --- --- --- 
if __name__=='__main__':
    print("[WARNING] lower the horizont of prediction")
    c_memeenune = dwvmm.MEMEENUNE()
    working_dataframe = c_memeenune.c_data_kijtyu.load_and_get_dataframe(
        dwvc.train_load_n_seq
    )
    c_memeenune.train_tff(
        working_dataframe,
        ACTUAL_MODEL_PATH='lightning_logs/always_saving_tft.ckpt', # set to None for fresh start
    )
    # input()
    # working_dataframe = c_memeenune.c_data_kijtyu.load_and_get_dataframe(
    #     max(dwvc.gss_c_seq_size,dwvc.tft_c_seq_size)
    # )
    # c_memeenune.launch_uwaabo(working_dataframe)