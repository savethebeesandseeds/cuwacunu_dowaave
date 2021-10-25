import os
import ast
import pandas as pd
# --- --- --- --- 
import cwcn_dowaave_memeenune as dwvmm
import cwcn_dwve_client_config as dwvcc
from cwcn_dwve_client_config import dwve_instrument_configuration as dwvic
# --- --- --- --- 
if __name__=='__main__':
    print("[WARNING] lower the horizont of prediction")
    symbol='BTCUSDTPERP'
    symbol='ETHUSDTPERP'
    c_memeenune = dwvmm.MEMEENUNE()
    if(dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['FIX_BKUP_STANDAR']):
        assert(dwvcc.CWCN_DUURUVA_CONFIG.DUURUVA_MEMEENUNE_TYPE == 'mean'), "method for fix bkup requires dowaave duuruva to be configure as mean"
        if(os.path.isfile(dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['TEMP_DATA_FILE'])):
            print("[PURGING:] file {}".format(dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['TEMP_DATA_FILE']))
            os.system('rm {}'.format(dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['TEMP_DATA_FILE']))
        with open(dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['BKUP_DATA_FILE'],'r') as _F:
            loaded_data=_F.read()
        loaded_data=loaded_data.split(dwvcc.CWCN_FARM_CONFIG.BKUP_MARKER)
        loaded_data=[__ for __ in loaded_data if __!='']
        for c_index,_d in enumerate(loaded_data):
            first_item=_d[:1000].split('\n')[0]+'\n'
            loaded_data[c_index]=first_item*dwvcc.CWCN_DUURUVA_CONFIG.DUURUVA_READY_COUNT+_d
            seq_aux=ast.literal_eval('[{}]'.format(loaded_data[c_index].replace('\n','')))
            loaded_dataframe=pd.DataFrame.from_dict(seq_aux,orient='columns')
            loaded_dataframe=c_memeenune.c_data_kijtyu.ujcamei_transform(symbol=symbol,working_dataframe=loaded_dataframe)
            # import matplotlib.pyplot as plt
            # loaded_dataframe['price'].plot()
            # plt.show()
            if(len(loaded_dataframe.index)>250):
                del loaded_dataframe['INDEX']
                loaded_data[c_index]=loaded_dataframe.to_dict('records')
                with open(dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['TEMP_DATA_FILE'],'a+',encoding='utf-8') as _F:
                    _F.write(',\n'.join([str(__) for __ in loaded_data[c_index]])+',\n')
        del loaded_data
        del loaded_dataframe
        del seq_aux
        del first_item
        aux_training_file_path=dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['TEMP_DATA_FILE']
    else:
        aux_training_file_path=dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['DATA_FILE']
    
    working_dataframe = c_memeenune.c_data_kijtyu.load_and_get_dataframe(
        data_file=aux_training_file_path,
        symbol=symbol,
        seq_size=dwvic.__dict__[symbol].train_load_n_seq
    )

    if(dwvic.__dict__[symbol].TRAINING_TRANSFORM_CONFIG['FIX_BKUP_STANDAR']):
        os.system('rm {}'.format(aux_training_file_path))

    c_memeenune.train_tff(
        working_dataframe,
        symbol=symbol,
        reset=False, # meaning to reset the model or True is one wants to train the current model
        train_short=True,
        train_long=False,
    )
    # input()
    # working_dataframe = c_memeenune.c_data_kijtyu.load_and_get_dataframe(
    #     max(dwvic.__dict__[symbol].gss_c_seq_size,dwvic.__dict__[symbol].tft_c_seq_size)
    # )
    # c_memeenune.launch_uwaabo(working_dataframe)