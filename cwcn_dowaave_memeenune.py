# --- --- --- ---- 
# from os import system
import os
import time
import subprocess
import ast
import pandas as pd
# --- --- --- ---- 
os.environ['DOWAAVE_GSS_F1']=""
os.environ['DOWAAVE_GSS_F2']=""
os.environ['DOWAAVE_TFT_S_F1']=""
os.environ['DOWAAVE_TFT_S_F2']=""
os.environ['DOWAAVE_TFT_L_F1']=""
os.environ['DOWAAVE_TFT_L_F2']=""
os.environ['DOWAAVE_HRZ_F1']=""
# --- --- --- ---- 
import torch_dowaave_gauss_nebajke
import torch_dowaave_tft_nebajke
import cwcn_dowaave_hrz_nebajke
import cwcn_duuruva_piaabo
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# EXPECT cwcn_dowaave_front.py be active
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
import cwcn_dwve_client_config as dwvcc
# --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
def logging_fun(msg):
    print("[dowaave_nebajke:] \n{}".format(msg))
# --- --- --- --- --- 
class MEMEENUNE_DATA_KIJTYU:
    def __init__(self,dwvic):
        # --- --- --- --- --- SET UP THE PRICE DUURUVA
        self.dwvic=dwvic
        self.loaded_dataframe=dict([(__,None) for __ in dwvcc.ACTIVE_SYMBOLS])
        # --- --- --- --- --- 
        self.price_duuruva=dict([(__,
            cwcn_duuruva_piaabo.DUURUVA(
                _duuruva_vector_size=1,
                _wrapper_duuruva_normalize=dwvcc.CWCN_DUURUVA_CONFIG.DUURUVA_MEMEENUNE_TYPE,
                _d_name='{}_price_duuruva'.format(__)
            )
        ) for __ in dwvcc.ACTIVE_SYMBOLS])
        # --- --- --- --- --- 
        self.load_data()
        # --- --- --- --- --- 
    def load_data(self):
        # --- --- --- --- --- READ ORIGINAL FILE
        # self.loaded_dataframe[symbol]=pd.read_csv(
        #     self.dwvic[symbol].DATA_FILE, 
        #     usecols=self.dwvic[symbol].DATA_USED_COLUMNS,
        #     # nrows=200,
        #     )
        for symbol in dwvcc.ACTIVE_SYMBOLS:
            self.loaded_dataframe[symbol]=self.load_and_get_dataframe(
                symbol=symbol, 
                seq_size=max(
                    self.dwvic[symbol].gss_c_seq_size,
                    self.dwvic[symbol].tft_short_c_seq_size,
                    self.dwvic[symbol].tft_long_c_seq_size,
                    self.dwvic[symbol].holding_n_seq
                )
            )
        # --- --- --- ---
        # --- --- --- ---|
    def ujcamei_transform(self,symbol,working_dataframe=None):
        if(working_dataframe is None):
            ujcamei_data=self.loaded_dataframe[symbol]
        else:
            ujcamei_data=working_dataframe
        # --- --- --- --- --- CONTRACT
        ujcamei_data=ujcamei_data[self.dwvic[symbol].DATA_USED_COLUMNS]
        ujcamei_data[['price','ts','sequence']]=ujcamei_data[['price','ts','sequence']].apply(pd.to_numeric)
        # --- --- --- --- --- EXPAND 
        ujcamei_data['delta_sequence'] = ujcamei_data['sequence'].diff().fillna(0)
        ujcamei_data['delta_price'] = ujcamei_data['price'].diff().fillna(0)
        ujcamei_data['delta_ts'] = ujcamei_data['ts'].diff().fillna(0)
        ujcamei_data['INDEX'] = ujcamei_data.index # ujcamei_data.reset_index(level=0,inplace=True) # add index column
        ujcamei_data['real_price'] = ujcamei_data['price'].copy()
        # plt.figure()
        # ujcamei_data['delta_sequence'].plot()
        # plt.figure()
        # ujcamei_data['price'].plot()
        # plt.figure()
        # ujcamei_data['delta_price'].plot()
        # plt.figure()
        # ujcamei_data['delta_ts'].plot()
        # plt.show()
        # --- --- --- --- --- SPLIT DATAFRAME INTO VALID FRAGMENTS
        # logging_fun("--- --- sequence --- ---")
        # logging_fun(ujcamei_data[ujcamei_data.apply(lambda x: x['delta_sequence']!=1.0,axis=1)])
        # logging_fun(ujcamei_data[ujcamei_data.apply(lambda x: x['delta_sequence']!=1.0,axis=1)].index)
        # logging_fun("--- --- price --- ---")
        # logging_fun(ujcamei_data['delta_price'].max())
        # --- --- --- --- --- SPLIT DATAFRAME INTO VALID FRAGMENTS
        if(self.dwvic[symbol].TRANSFORM_CANDLE['candle_flag']):
            def get_applied_item(df,_index):
                # return df.iloc[_index][self.dwvic[symbol].TRANSFORM_CANDLE['candle_step'][3]]
                return df.iloc[_index][self.dwvic[symbol].TRANSFORM_CANDLE['candle_item']]
            idx_list=[]
            candle_stp=self.dwvic[symbol].TRANSFORM_CANDLE['candle_step']
            hold_state=get_applied_item(ujcamei_data,0)
            for c_index,row in ujcamei_data.iterrows():
                # print(hold_state,get_applied_item(ujcamei_data,c_index),candle_stp)
                if(abs(hold_state-get_applied_item(ujcamei_data,c_index))>=candle_stp):
                    idx_list.append(c_index)
                    hold_state=get_applied_item(ujcamei_data,c_index)
            logging_fun("candle data length : {}".format(len(idx_list)))
            ujcamei_data=ujcamei_data.iloc[idx_list]
            ujcamei_data.reset_index(inplace=True)
            ujcamei_data['delta_sequence'] = ujcamei_data['sequence'].diff().fillna(0)
            ujcamei_data['delta_price'] = ujcamei_data['price'].diff().fillna(0)
            ujcamei_data['delta_ts'] = ujcamei_data['ts'].diff().fillna(0)
            ujcamei_data['INDEX'] = ujcamei_data.index # ujcamei_data.reset_index(level=0,inplace=True) # add index column
            del(ujcamei_data['index'])
            logging_fun("--- --- data candeled --- ---")
            # logging_fun(ujcamei_data.describe())
            # logging_fun(ujcamei_data.head())
        # --- --- --- --- --- DUURUVA TRANSFORM THE PRICE
        if(dwvcc.CWCN_DUURUVA_CONFIG.DUURUVA_MEMEENUNE_TYPE == 'mean'):
            # c_mean=ujcamei_data['price'].mean()
            # ujcamei_data['price']=ujcamei_data['price'].apply(lambda x: x-c_mean)
            self.price_duuruva[symbol]._reset_duuruva_()
            ujcamei_data['price']=ujcamei_data['price'].apply(
                lambda x: self.price_duuruva[symbol]._duuruva_value_wrapper_(x))
        else:
            assert(False), "dwvcc.CWCN_DUURUVA_CONFIG.DUURUVA_MEMEENUNE_TYPE not implemented nor or stable for type not 'mean'"
        # plt
        # ujcamei_data['price'].plot()
        # for c_index,row in ujcamei_data.iterrows():
        #     aux_v=self.price_duuruva[symbol]._duuruva_value_wrapper_(row['price'])
        #     print(row['price'],"->",aux_v)
        #     ujcamei_data.ilo[c_index,'price']=aux_v
        #     print(ujcamei_data.ilo[c_index,'price'])
        #     input()
        return ujcamei_data
    def get_c_tick(self,symbol):
        seq_aux=subprocess.check_output(['tail', '-{}'.format(1), self.dwvic[symbol].DATA_FILE]).decode('ascii').replace('\n','')
        seq_aux=ast.literal_eval('[{}]'.format(seq_aux))
        return seq_aux[0]
    def check_if_data_update(self,symbol):
        seq_aux=subprocess.check_output(['tail', '-{}'.format(1), self.dwvic[symbol].DATA_FILE]).decode('ascii').replace('\n','')
        seq_aux=ast.literal_eval('[{}]'.format(seq_aux))
        if(self.dwvic[symbol].TRANSFORM_CANDLE['candle_flag']):
            def get_candle_step(df):
                assert(False), "not in use!"
            hold_state=self.c_last_tk[self.dwvic[symbol].TRANSFORM_CANDLE['candle_item']].item()
            actual_state=seq_aux[0][self.dwvic[symbol].TRANSFORM_CANDLE['candle_item']]
            candle_stp=self.dwvic[symbol].TRANSFORM_CANDLE['candle_step']
            if(abs(hold_state-actual_state)>=candle_stp):
                return True
            else:
                return False
        else:
            if(self.c_last_tk['sequence'].item()!=seq_aux[0]['sequence']):
                return True
            else:
                return False
    def load_and_get_dataframe(self, symbol, seq_size, data_file=None):
        # print('tail -{} {}'.format(seq_size,self.dwvic[symbol].DATA_FILE))
        seq_size=int(seq_size)
        load_seq_size=int(self.dwvic[symbol].TRANSFORM_CANDLE['prom_candle_step_multiplier']*seq_size)
        if(data_file is None):
            seq_aux=subprocess.check_output(['tail', '-{}'.format(load_seq_size), self.dwvic[symbol].DATA_FILE]).decode('ascii').replace('\n','')
        else:
            seq_aux=subprocess.check_output(['tail', '-{}'.format(load_seq_size), data_file]).decode('ascii').replace('\n','')
        seq_aux=ast.literal_eval('[{}]'.format(seq_aux))
        self.loaded_dataframe[symbol]=pd.DataFrame.from_dict(seq_aux,orient='columns')
        self.c_last_tk=self.loaded_dataframe[symbol].tail(1)
        self.loaded_dataframe[symbol]=self.ujcamei_transform(symbol=symbol)
        logging_fun("--- --- ujcamei data --- ---")
        logging_fun(self.loaded_dataframe[symbol].describe())
        logging_fun(self.loaded_dataframe[symbol].head()) 
        self.loaded_dataframe[symbol]=self.loaded_dataframe[symbol].tail(seq_size)
        if(len(self.loaded_dataframe[symbol].index)<seq_size):
            print("[WARNING] loading dataframe, seq_size requested to be {}, but loaded only {}. Is recommended to configure a higher value for self.dwvic[symbol].TRANSFORM_CANDLE['prom_candle_step_multiplier'].".format(seq_size,len(self.loaded_dataframe[symbol].index)))
        return self.loaded_dataframe[symbol]


class MEMEENUNE:
    def __init__(self):
        self.dwvic={}
        self.working_dataframe=None
        self.dwvmoldes={}
        for symbol in dwvcc.ACTIVE_SYMBOLS:
            self.dwvic[symbol]=dwvcc.dwve_instrument_configuration.__dict__[symbol]
            self.dwvmoldes[symbol]={'s':None,'l':None}
        self.c_data_kijtyu = MEMEENUNE_DATA_KIJTYU(dwvic=self.dwvic)
        for symbol in dwvcc.ACTIVE_SYMBOLS:
            self.dwvmoldes[symbol]['s'] = torch_dowaave_tft_nebajke.load_tft(
                model_path=self.dwvic[symbol].tft_short_ALWAYS_SAVING_MODEL)
            # tft_model=self.dwvmoldes[symbol]['l'] = torch_dowaave_tft_nebajke.load_tft(
            #     model_path=self.dwvic[symbol].tft_long_ALWAYS_SAVING_MODEL)


        # self.launch_uwaabo(True)
    def train_tff(self,working_dataframe,symbol,reset=False,train_long=False,train_short=False):
        if(train_short):
            print("[MEMEENUNE,] training short horizont tft model...")
            torch_dowaave_tft_nebajke.train_tft(working_dataframe,
                ALWAYS_SAVING_MODEL=self.dwvic[symbol].tft_short_ALWAYS_SAVING_MODEL, # reduntant saving
                ACTUAL_MODEL_PATH=None if reset or not(os.path.isfile(self.dwvic[symbol].tft_short_ACTUAL_MODEL_PATH)) else self.dwvic[symbol].tft_short_ACTUAL_MODEL_PATH,
                JUST_LOAD=False,
                DO_TUNNIN=self.dwvic[symbol].tft_short_DO_TUNNIN,
                FIND_OPTMAL_LR=self.dwvic[symbol].tft_short_FIND_OPTMAL_LR,
                LEARNING_RATE=self.dwvic[symbol].tft_short_LEARNING_RATE,
                max_prediction_length=self.dwvic[symbol].tft_short_max_prediction_length,
                max_encoder_length=self.dwvic[symbol].tft_short_max_encoder_length,
                validation_porcentaje=self.dwvic[symbol].tft_short_validation_porcentaje,
                n_epochs=self.dwvic[symbol].tft_short_n_epochs,
                batch_size=self.dwvic[symbol].tft_short_batch_size)
        else:
            print("[MEMEENUNE,] skipping training short horizont tft model...")
        if(train_long):
            print("[MEMEENUNE,] training long horizont tft model...")
            torch_dowaave_tft_nebajke.train_tft(working_dataframe,
                ALWAYS_SAVING_MODEL=self.dwvic[symbol].tft_long_ALWAYS_SAVING_MODEL, # reduntant saving
                ACTUAL_MODEL_PATH=None if reset or not(os.path.isfile(self.dwvic[symbol].tft_long_ACTUAL_MODEL_PATH)) else self.dwvic[symbol].tft_long_ACTUAL_MODEL_PATH,
                JUST_LOAD=False,
                DO_TUNNIN=self.dwvic[symbol].tft_long_DO_TUNNIN,
                FIND_OPTMAL_LR=self.dwvic[symbol].tft_long_FIND_OPTMAL_LR,
                LEARNING_RATE=self.dwvic[symbol].tft_long_LEARNING_RATE,
                max_prediction_length=self.dwvic[symbol].tft_long_max_prediction_length,
                max_encoder_length=self.dwvic[symbol].tft_long_max_encoder_length,
                validation_porcentaje=self.dwvic[symbol].tft_long_validation_porcentaje,
                n_epochs=self.dwvic[symbol].tft_long_n_epochs,
                batch_size=self.dwvic[symbol].tft_long_batch_size)
        else:
            print("[MEMEENUNE,] skipping training long horizont tft model...")

    def launch_uwaabo(self,symbol=None,force=True,gss_flag=True,tft_flag=True,hrz_flag=True):
        # --- --- --- --- 
        # --- --- --- --- 
        ret_flag=False
        iter_symbols = dwvcc.ACTIVE_SYMBOLS if symbol is None else [symbol]
        for symbol in iter_symbols:
            if(force or self.c_data_kijtyu.check_if_data_update(symbol=symbol)):
                # --- --- --- --- 
                self.working_dataframe = self.c_data_kijtyu.load_and_get_dataframe(
                    symbol=symbol, 
                    seq_size=max(self.dwvic[symbol].gss_c_seq_size,self.dwvic[symbol].tft_short_c_seq_size,self.dwvic[symbol].tft_long_c_seq_size,self.dwvic[symbol].holding_n_seq)
                )
                # --- --- --- --- 
                if(force or hrz_flag):
                    hrz_figname=cwcn_dowaave_hrz_nebajke.relife_hrz(
                        working_dataframe=self.working_dataframe,
                        symbol=symbol,
                        active_dimension='real_price',
                    )
                    os.environ['DOWAAVE_HRZ_F1']=hrz_figname
                # --- --- --- --- 
                if(force or gss_flag):
                    s_time=time.time()
                    gss_figname=torch_dowaave_gauss_nebajke.relife_gauss(
                        self.working_dataframe.tail(self.dwvic[symbol].gss_c_seq_size).copy(),
                        active_dimension='price',
                        active_coin=symbol,
                        training_iter =self.dwvic[symbol].gss_training_iter,
                        learning_rate =self.dwvic[symbol].gss_learning_rate,
                        c_horizon=self.dwvic[symbol].gss_c_horizon,
                        c_horizon_delta=self.dwvic[symbol].gss_c_horizon_delta,
                        c_iterations=self.dwvic[symbol].gss_c_iterations,
                        c_backlash=self.dwvic[symbol].gss_c_backlash)
                    os.environ['DOWAAVE_GSS_F1']=gss_figname[0]
                    os.environ['DOWAAVE_GSS_F2']=gss_figname[1]
                    logging_fun("memeenune update - [{}] gauss_fig: {}".format(symbol,gss_figname))
                    logging_fun("memeenune update - [{}] gauss exe in : {} [s]".format(symbol,time.time()-s_time))
                # --- --- --- --- 
                if(force or tft_flag):
                    s_time=time.time()
                    tft_short_figname=torch_dowaave_tft_nebajke.relife_tft(
                        self.working_dataframe.tail(self.dwvic[symbol].tft_short_c_seq_size).copy(), # self.dwvic[symbol].tft_short_c_seq_size
                        tft_model=self.dwvmoldes[symbol]['s'],
                        # model_path=self.dwvic[symbol].tft_short_ALWAYS_SAVING_MODEL, # not needed due to parsing model #FIXME when updating model online
                        max_encoder_length=self.dwvic[symbol].tft_short_max_encoder_length,
                        max_prediction_length=self.dwvic[symbol].tft_short_max_prediction_length,
                        indicator='s')
                    logging_fun("memeenune update - [{}] tft_short_fig: {}".format(symbol,tft_short_figname))
                    logging_fun("memeenune update - [{}] tft exe in : {} [s]".format(symbol,time.time()-s_time))
                    os.environ['DOWAAVE_TFT_S_F1']=tft_short_figname[0]
                    os.environ['DOWAAVE_TFT_S_F2']=tft_short_figname[1]
                # --- --- --- --- 
                # s_time=time.time()
                # tft_long_figname=torch_dowaave_tft_nebajke.relife_tft(
                #     self.working_dataframe.tail(self.dwvic[symbol].tft_long_c_seq_size).copy(), # self.dwvic[symbol].tft_long_c_seq_size
                #     model_path=self.dwvic[symbol].tft_long_ALWAYS_SAVING_MODEL,
                #     max_encoder_length=self.dwvic[symbol].tft_long_max_encoder_length,
                #     max_prediction_length=self.dwvic[symbol].tft_long_max_prediction_length,
                #     indicator='l')
                # logging_fun("tft_long_fig: {}".format(tft_long_figname))
                # logging_fun("tft exe in : {} [s]".format(time.time()-s_time))
                # os.environ['DOWAAVE_TFT_L_F1']=tft_long_figname[0]
                # os.environ['DOWAAVE_TFT_L_F2']=tft_long_figname[1]
                # --- --- --- --- 
                ret_flag=True
        # --- --- --- --- 
        return ret_flag
        # --- --- --- --- 
            # s_time=time.time()
            # gss_figname1, gss_figname2=torch_dowaave_gauss_nebajke.relife_gauss(
            #     working_dataframe.tail(self.dwvic[symbol].gss_c_seq_size).copy(),
            #     active_dimension='price',
            #     active_coin=self.dwvic[symbol].SYMBOL,
            #     training_iter =self.dwvic[symbol].gss_training_iter,
            #     learning_rate =self.dwvic[symbol].gss_learning_rate,
            #     c_horizon=self.dwvic[symbol].gss_c_horizon,
            #     c_horizon_delta=self.dwvic[symbol].gss_c_horizon_delta,
            #     c_iterations=self.dwvic[symbol].gss_c_iterations,
            #     c_backlash=self.dwvic[symbol].gss_c_backlash)
            # logging_fun("gauss_fig1: {}, gauss_fig2:{}".format(gss_figname1,gss_figname2))
            # logging_fun("gauss exe in : {} [s]".format(time.time()-s_time))
            # import random
            # --- --- --- --- 
            # for _ in range(1):
            #     idx_rand=random.randint(0,len(working_dataframe.index)-self.dwvic[symbol].tft_short_c_seq_size)
            #     s_time=time.time()
            #     gss_figname1, gss_figname2=torch_dowaave_gauss_nebajke.relife_gauss(
            #         working_dataframe.iloc[idx_rand:idx_rand+self.dwvic[symbol].tft_short_c_seq_size].copy(),
            #         active_dimension='price',
            #         active_coin=self.dwvic[symbol].SYMBOL,
            #         training_iter =self.dwvic[symbol].gss_training_iter,
            #         learning_rate =self.dwvic[symbol].gss_learning_rate,
            #         c_horizon=self.dwvic[symbol].gss_c_horizon,
            #         c_horizon_delta=self.dwvic[symbol].gss_c_horizon_delta,
            #         c_iterations=self.dwvic[symbol].gss_c_iterations,
            #         c_backlash=self.dwvic[symbol].gss_c_backlash)
            #     logging_fun("gauss_fig1: {}, gauss_fig2:{}".format(gss_figname1,gss_figname2))
            #     logging_fun("gauss exe in : {} [s]".format(time.time()-s_time))
            #     s_time=time.time()
            #     tft_short_figname=torch_dowaave_tft_nebajke.relife_tft(
            #         working_dataframe.iloc[idx_rand:idx_rand+self.dwvic[symbol].tft_short_c_seq_size].copy(), # self.dwvic[symbol].tft_short_c_seq_size
            #         model_path=self.dwvic[symbol].tft_short_ALWAYS_SAVING_MODEL,
            #         max_encoder_length=self.dwvic[symbol].tft_short_max_encoder_length,
            #         max_prediction_length=self.dwvic[symbol].tft_short_max_prediction_length,
            #         indicator='s')
            #     logging_fun("tft_short_fig1: {}".format(tft_short_figname))
            #     logging_fun("tft exe in : {} [s]".format(time.time()-s_time))
            # --- --- --- --- 
if __name__=="__main__":
    c_memeenune = MEMEENUNE()
    c_memeenune.launch_uwaabo(symbol=None)