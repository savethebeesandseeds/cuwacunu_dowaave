# --- --- --- ---- 
# from os import system
import os
import subprocess
import ast
import pandas as pd
# --- --- --- ---- 
os.environ['DOWAAVE_GSS_F1']=""
os.environ['DOWAAVE_GSS_F2']=""
os.environ['DOWAAVE_TFT_F1']=""
# --- --- --- ---- 
import torch_dowaave_gauss_nebajke
import torch_dowaave_tft_nebajke
import cwcn_duuruva_piaabo
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# EXPECT cwcn_dowaave_front.py be active
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
from cwcn_dwve_client_config import dwve_instrument_configuration as dwvc
# --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
def logging_fun(msg):
    print("[dowaave_nebajke:] \n{}".format(msg))
# --- --- --- --- --- 
class DATA_KIJTYU:
    def __init__(self):
        # --- --- --- --- --- SET UP THE PRICE DUURUVA
        self.price_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=1,_wrapper_duuruva_normalize='not', _d_name='price_duuruva')
        self.load_data()
    def load_data(self):
        # --- --- --- --- --- READ ORIGINAL FILE
        # self.loaded_dataframe=pd.read_csv(
        #     dwvc.DATA_FILE, 
        #     usecols=dwvc.DATA_USED_COLUMNS,
        #     # nrows=200,
        #     )
        self.loaded_dataframe=self.load_and_get_dataframe(max(dwvc.gss_c_seq_size,dwvc.tft_c_seq_size))
        # --- --- --- ---
        # --- --- --- ---
    def ujcamei_transform(self):
        # --- --- --- --- --- CONTRACT
        self.loaded_dataframe=self.loaded_dataframe[dwvc.DATA_USED_COLUMNS]
        self.loaded_dataframe[['price','ts','sequence']]=self.loaded_dataframe[['price','ts','sequence']].apply(pd.to_numeric)
        # --- --- --- --- --- EXPAND 
        self.loaded_dataframe['delta_sequence'] = self.loaded_dataframe['sequence'].diff().fillna(0)
        self.loaded_dataframe['delta_price'] = self.loaded_dataframe['price'].diff().fillna(0)
        self.loaded_dataframe['delta_ts'] = self.loaded_dataframe['ts'].diff().fillna(0)
        self.loaded_dataframe['INDEX'] = self.loaded_dataframe.index # self.loaded_dataframe.reset_index(level=0,inplace=True) # add index column
        # plt.figure()
        # self.loaded_dataframe['delta_sequence'].plot()
        # plt.figure()
        # self.loaded_dataframe['price'].plot()
        # plt.figure()
        # self.loaded_dataframe['delta_price'].plot()
        # plt.figure()
        # self.loaded_dataframe['delta_ts'].plot()
        # plt.show()
        # --- --- --- --- --- SPLIT DATAFRAME INTO VALID FRAGMENTS
        # logging_fun("--- --- sequence --- ---")
        # logging_fun(self.loaded_dataframe[self.loaded_dataframe.apply(lambda x: x['delta_sequence']!=1.0,axis=1)])
        # logging_fun(self.loaded_dataframe[self.loaded_dataframe.apply(lambda x: x['delta_sequence']!=1.0,axis=1)].index)
        # logging_fun("--- --- price --- ---")
        # logging_fun(self.loaded_dataframe['delta_price'].max())
        # --- --- --- --- --- SPLIT DATAFRAME INTO VALID FRAGMENTS
        if(dwvc.TRANSFORM_CANDLE['candle_flag']):
            def get_applied_item(df,_index):
                return df.iloc[_index][dwvc.TRANSFORM_CANDLE['candle_step'][3]]
            def get_candle_step(df):
                return dwvc.TRANSFORM_CANDLE['candle_step'][0]*dwvc.TRANSFORM_CANDLE['candle_step'][1](df.describe()[dwvc.TRANSFORM_CANDLE['candle_step'][2]])
            idx_list=[]
            candle_stp = get_candle_step(self.loaded_dataframe)
            hold_state=get_applied_item(self.loaded_dataframe,0)
            for c_index,row in self.loaded_dataframe.iterrows():
                # print(hold_state,get_applied_item(self.loaded_dataframe,c_index),candle_stp)
                if(abs(hold_state-get_applied_item(self.loaded_dataframe,c_index))>=candle_stp):
                    idx_list.append(c_index)
                    hold_state=get_applied_item(self.loaded_dataframe,c_index)
            logging_fun("candle data length : {}".format(len(idx_list)))
            self.loaded_dataframe=self.loaded_dataframe.iloc[idx_list]
            self.loaded_dataframe.reset_index(inplace=True)
            self.loaded_dataframe['delta_sequence'] = self.loaded_dataframe['sequence'].diff().fillna(0)
            self.loaded_dataframe['delta_price'] = self.loaded_dataframe['price'].diff().fillna(0)
            self.loaded_dataframe['delta_ts'] = self.loaded_dataframe['ts'].diff().fillna(0)
            self.loaded_dataframe['INDEX'] = self.loaded_dataframe.index # self.loaded_dataframe.reset_index(level=0,inplace=True) # add index column
            del(self.loaded_dataframe['index'])
            logging_fun("--- --- data candeled --- ---")
            # logging_fun(self.loaded_dataframe.describe())
            # logging_fun(self.loaded_dataframe.head())
        # --- --- --- --- --- DUURUVA TRANSFORM THE PRICE
        self.loaded_dataframe['price']=self.loaded_dataframe['price'].apply(lambda x: self.price_duuruva._duuruva_value_wrapper_(x)) #FIXME duuruva not used
        # plt
        # self.loaded_dataframe['price'].plot()
        # for c_index,row in self.loaded_dataframe.iterrows():
        #     aux_v=self.price_duuruva._duuruva_value_wrapper_(row['price'])
        #     print(row['price'],"->",aux_v)
        #     self.loaded_dataframe.ilo[c_index,'price']=aux_v
        #     print(self.loaded_dataframe.ilo[c_index,'price'])
        #     input()
    def get_c_tick(self):
        seq_aux=subprocess.check_output(['tail', '-{}'.format(1), dwvc.DATA_FILE]).decode('ascii').replace('\n','')
        seq_aux=ast.literal_eval('[{}]'.format(seq_aux))
        return seq_aux[0]
    def check_if_data_update(self):
        seq_aux=subprocess.check_output(['tail', '-{}'.format(1), dwvc.DATA_FILE]).decode('ascii').replace('\n','')
        seq_aux=ast.literal_eval('[{}]'.format(seq_aux))
        print(self.c_last_tk['sequence'].item(),seq_aux[0]['sequence'])
        if(self.c_last_tk['sequence'].item()!=seq_aux[0]['sequence']):
            return True
        else:
            return False
    def load_and_get_dataframe(self, seq_size):
        # print('tail -{} {}'.format(seq_size,dwvc.DATA_FILE))
        seq_aux=subprocess.check_output(['tail', '-{}'.format(seq_size), dwvc.DATA_FILE]).decode('ascii').replace('\n','')
        seq_aux=ast.literal_eval('[{}]'.format(seq_aux))
        self.loaded_dataframe=pd.DataFrame.from_dict(seq_aux,orient='columns')
        logging_fun("--- --- ujcamei data --- ---")
        logging_fun(self.loaded_dataframe.describe())
        logging_fun(self.loaded_dataframe.head()) 
        self.ujcamei_transform()
        self.c_last_tk=self.loaded_dataframe.tail(1)
        return self.loaded_dataframe
        
        
class MEMEENUNE:
    def __init__(self):
        self.c_data_kijtyu = DATA_KIJTYU()
        self.launch_uwaabo(True)
    def train_tff(self,working_dataframe):
        torch_dowaave_tft_nebajke.train_tft(working_dataframe,
            ALWAYS_SAVING_MODEL=dwvc.tft_ALWAYS_SAVING_MODEL, # reduntant saving
            ACTUAL_MODEL_PATH=dwvc.tft_ACTUAL_MODEL_PATH,
            JUST_LOAD=False,
            DO_TUNNIN=dwvc.tft_DO_TUNNIN,
            FIND_OPTMAL_LR=dwvc.tft_FIND_OPTMAL_LR,
            LEARNING_RATE=dwvc.tft_LEARNING_RATE,
            max_prediction_length=dwvc.tft_max_prediction_length,
            max_encoder_length=dwvc.tft_max_encoder_length,
            validation_porcentaje=dwvc.tft_validation_porcentaje,
            n_epochs=dwvc.tft_n_epochs,
            batch_size=dwvc.tft_batch_size)

    def launch_uwaabo(self,force=False):
        # --- --- --- --- 
        import time
        s_time=time.time()
        # --- --- --- --- 
        if(force or self.c_data_kijtyu.check_if_data_update()):
            self.working_dataframe = self.c_data_kijtyu.load_and_get_dataframe(max(dwvc.gss_c_seq_size,dwvc.tft_c_seq_size))
            # --- --- --- --- 
            gss_figname1, gss_figname2=torch_dowaave_gauss_nebajke.relife_gauss(
                self.working_dataframe.tail(dwvc.gss_c_seq_size).copy(),
                active_dimension='price',
                active_coin=dwvc.SYMBOL,
                training_iter =dwvc.gss_training_iter,
                learning_rate =dwvc.gss_learning_rate,
                c_horizon=dwvc.gss_c_horizon,
                c_horizon_delta=dwvc.gss_c_horizon_delta,
                c_iterations=dwvc.gss_c_iterations,
                c_backlash=dwvc.gss_c_backlash)
            os.environ['DOWAAVE_GSS_F1']=gss_figname1
            os.environ['DOWAAVE_GSS_F2']=gss_figname2
            logging_fun("gauss_fig1: {}, gauss_fig2:{}".format(gss_figname1,gss_figname2))
            logging_fun("gauss exe in : {} [s]".format(time.time()-s_time))
            s_time=time.time()
            tft_figname=torch_dowaave_tft_nebajke.relife_tft(
                self.working_dataframe.tail(dwvc.tft_c_seq_size).copy(), # dwvc.tft_c_seq_size
                model_path=dwvc.tft_ALWAYS_SAVING_MODEL,
                max_encoder_length=dwvc.tft_max_encoder_length,
                max_prediction_length=dwvc.tft_max_prediction_length)
            logging_fun("tft_fig1: {}".format(tft_figname))
            logging_fun("tft exe in : {} [s]".format(time.time()-s_time))
            os.environ['DOWAAVE_TFT_F1']=tft_figname
        # --- --- --- --- 
        
        # s_time=time.time()
        # gss_figname1, gss_figname2=torch_dowaave_gauss_nebajke.relife_gauss(
        #     working_dataframe.tail(dwvc.gss_c_seq_size).copy(),
        #     active_dimension='price',
        #     active_coin=dwvc.SYMBOL,
        #     training_iter =dwvc.gss_training_iter,
        #     learning_rate =dwvc.gss_learning_rate,
        #     c_horizon=dwvc.gss_c_horizon,
        #     c_horizon_delta=dwvc.gss_c_horizon_delta,
        #     c_iterations=dwvc.gss_c_iterations,
        #     c_backlash=dwvc.gss_c_backlash)
        # logging_fun("gauss_fig1: {}, gauss_fig2:{}".format(gss_figname1,gss_figname2))
        # logging_fun("gauss exe in : {} [s]".format(time.time()-s_time))
        # import random
        # --- --- --- --- 
        # for _ in range(1):
        #     idx_rand=random.randint(0,len(working_dataframe.index)-dwvc.tft_c_seq_size)
        #     s_time=time.time()
        #     gss_figname1, gss_figname2=torch_dowaave_gauss_nebajke.relife_gauss(
        #         working_dataframe.iloc[idx_rand:idx_rand+dwvc.tft_c_seq_size].copy(),
        #         active_dimension='price',
        #         active_coin=dwvc.SYMBOL,
        #         training_iter =dwvc.gss_training_iter,
        #         learning_rate =dwvc.gss_learning_rate,
        #         c_horizon=dwvc.gss_c_horizon,
        #         c_horizon_delta=dwvc.gss_c_horizon_delta,
        #         c_iterations=dwvc.gss_c_iterations,
        #         c_backlash=dwvc.gss_c_backlash)
        #     logging_fun("gauss_fig1: {}, gauss_fig2:{}".format(gss_figname1,gss_figname2))
        #     logging_fun("gauss exe in : {} [s]".format(time.time()-s_time))
        #     s_time=time.time()
        #     tft_figname=torch_dowaave_tft_nebajke.relife_tft(
        #         working_dataframe.iloc[idx_rand:idx_rand+dwvc.tft_c_seq_size].copy(), # dwvc.tft_c_seq_size
        #         model_path=dwvc.tft_ALWAYS_SAVING_MODEL,
        #         max_encoder_length=dwvc.tft_max_encoder_length,
        #         max_prediction_length=dwvc.tft_max_prediction_length)
        #     logging_fun("tft_fig1: {}".format(tft_figname))
        #     logging_fun("tft exe in : {} [s]".format(time.time()-s_time))
        # --- --- --- --- 

if __name__=="__main__":
    c_memeenune = MEMEENUNE()
    # c_data_kijtyu = DATA_KIJTYU()
    # working_dataframe = c_data_kijtyu.load_and_get_dataframe(
    #     dwvc.train_load_n_seq
    # )
    # train_tff(working_dataframe)
    # input()
    c_memeenune.launch_uwaabo()
    # while True:
    #     print(os.environ['DOWAAVE_GSS_F1'])