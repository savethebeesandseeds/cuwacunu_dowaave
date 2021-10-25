# --- --- --- --- --- 
# mayor references to Pytorch and https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html?highlight=covariates#Demand-forecasting-with-the-Temporal-Fusion-Transformer
# --- --- --- --- --- 
# powered by waajacu
# --- --- --- --- ---
import os
import time
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
# os.chdir("../../..")
import copy
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
# --- --- --- --- --- 
import cwcn_duuruva_piaabo
from cwcn_dwve_client_config import dwve_instrument_configuration as dwvc
# --- --- --- --- --- 
tft_dpi=120
ON_FILE_WLOT_FOLDER='./tft_dumps'
# --- --- --- --- --- 

def assert_folder(_f_path):
    if(not os.path.isdir(_f_path)):
        os.mkdir(_f_path)
assert_folder(ON_FILE_WLOT_FOLDER)
for _f in os.listdir(ON_FILE_WLOT_FOLDER):
    for _f2 in os.listdir(os.path.join(ON_FILE_WLOT_FOLDER,_f)):
        print("[PURGIN FILE:]",os.path.join(ON_FILE_WLOT_FOLDER,_f,_f2))
        os.remove(os.path.join(ON_FILE_WLOT_FOLDER,_f,_f2))
# --- --- ---
def logging_fun(msg): 
    print("[dowaave_nebajke:] \n{}".format(msg))
# --- --- --- --- --- 
    
    

# # --- --- --- --- --- SET UP THE PRICE DUURUVA
# price_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=1,_wrapper_duuruva_normalize='not', _d_name='price_duuruva')
# # --- --- --- --- --- READ ORIGINAL FILE
# loaded_dataframe=pd.read_csv(
#     dwvc.tft_DATA_FILE, 
#     usecols=dwvc.tft_DATA_USED_COLUMNS,
#     )
# # --- --- --- --- --- EXPAND delta_sequence
# loaded_dataframe['delta_sequence'] = loaded_dataframe['sequence'].diff().fillna(0)
# loaded_dataframe['delta_price'] = loaded_dataframe['price'].diff().fillna(0)
# loaded_dataframe['delta_ts'] = loaded_dataframe['ts'].diff().fillna(0)
# loaded_dataframe['INDEX'] = loaded_dataframe.index # loaded_dataframe.reset_index(level=0,inplace=True) # add index column
# # plt.figure()
# # loaded_dataframe['delta_sequence'].plot()
# # plt.figure()
# # loaded_dataframe['price'].plot()
# # plt.figure()
# # loaded_dataframe['delta_price'].plot()
# # plt.figure()
# # loaded_dataframe['delta_ts'].plot()
# # plt.show()
# # --- --- --- --- --- SPLIT DATAFRAME INTO VALID FRAGMENTS
# # logging_fun("--- --- sequence --- ---")
# # logging_fun(loaded_dataframe[loaded_dataframe.apply(lambda x: x['delta_sequence']!=1.0,axis=1)])
# # logging_fun(loaded_dataframe[loaded_dataframe.apply(lambda x: x['delta_sequence']!=1.0,axis=1)].index)
# # logging_fun("--- --- price --- ---")
# # logging_fun(loaded_dataframe['delta_price'].max())
# # --- --- --- --- --- SPLIT DATAFRAME INTO VALID FRAGMENTS
# if(dwvc.tft_TRANSFORM_CANDLE['candle_flag']):
#     def get_applied_item(df,_index):
#         return df.iloc[_index][dwvc.tft_TRANSFORM_CANDLE['candle_step'][3]]
#     def get_candle_step(df):
#         return dwvc.tft_TRANSFORM_CANDLE['candle_step'][0]*dwvc.tft_TRANSFORM_CANDLE['candle_step'][1](df.describe()[dwvc.tft_TRANSFORM_CANDLE['candle_step'][2]])
#     idx_list=[]
#     candle_stp = get_candle_step(loaded_dataframe)
#     hold_state=get_applied_item(loaded_dataframe,0)
#     for c_index,row in loaded_dataframe.iterrows():
#         # print(hold_state,get_applied_item(loaded_dataframe,c_index),candle_stp)
#         if(abs(hold_state-get_applied_item(loaded_dataframe,c_index))>=candle_stp):
#             idx_list.append(c_index)
#             hold_state=get_applied_item(loaded_dataframe,c_index)
#     logging_fun("candle data length : {}".format(len(idx_list)))
#     loaded_dataframe=loaded_dataframe.iloc[idx_list]
#     loaded_dataframe.reset_index(inplace=True)
#     loaded_dataframe['delta_sequence'] = loaded_dataframe['sequence'].diff().fillna(0)
#     loaded_dataframe['delta_price'] = loaded_dataframe['price'].diff().fillna(0)
#     loaded_dataframe['delta_ts'] = loaded_dataframe['ts'].diff().fillna(0)
#     loaded_dataframe['INDEX'] = loaded_dataframe.index # loaded_dataframe.reset_index(level=0,inplace=True) # add index column
#     del(loaded_dataframe['index'])
#     logging_fun("--- --- data candeled --- ---")
#     # logging_fun(loaded_dataframe.describe())
#     # logging_fun(loaded_dataframe.head())
# # plt.figure()
# # loaded_dataframe['price'].plot()
# # --- --- --- --- --- DUURUVA TRANSFORM THE PRICE
# # loaded_dataframe['price']=loaded_dataframe['price'].apply(lambda x: price_duuruva._duuruva_value_wrapper_(x))

# # price_mean=loaded_dataframe['price'].mean()
# # price_std=loaded_dataframe['price'].std()
# # loaded_dataframe['price']=loaded_dataframe['price'].apply(lambda x: (x-price_mean)/price_std)

# # for c_index,row in loaded_dataframe.iterrows():
# #     aux_v=price_duuruva._duuruva_value_wrapper_(row['price'])
# #     print(row['price'],"->",aux_v)
# #     loaded_dataframe.ilo[c_index,'price']=aux_v
# #     print(loaded_dataframe.ilo[c_index,'price'])
# #     input()
# logging_fun("--- --- data --- ---")
# logging_fun(loaded_dataframe.describe())
# logging_fun(loaded_dataframe.head())
# # plt.figure()
# # loaded_dataframe['price'].plot()
# # plt.show()
# # input("STOP")
# # logging_fun(loaded_dataframe[loaded_dataframe.apply(lambda x: x['delta_price']!=1.0,axis=1)])
# # logging_fun(loaded_dataframe[loaded_dataframe.apply(lambda x: x['delta_price']!=1.0,axis=1)].index)
# # logging_fun("--- --- ts --- ---")
# # logging_fun(loaded_dataframe[loaded_dataframe.apply(lambda x: x['delta_ts']!=1.0,axis=1)])
# # logging_fun(loaded_dataframe[loaded_dataframe.apply(lambda x: x['delta_ts']!=1.0,axis=1)].index)
# # for index, df_row in loaded_dataframe.iterrows():

# # --- --- --- --- --- 

# # --- --- --- --- --- EVALUATE DATA HEALT
# # --- --- --- --- --- 
# # tsane_composed_distribution=torch.distributions.Categorical(
# #     probs=dowaave_state
# # )
# # sample=tsane_composed_distribution.sample()
# # --- --- --- --- --- 

# # --- --- --- --- --- 

def train_tft(working_dataframe,
    ALWAYS_SAVING_MODEL,
    ACTUAL_MODEL_PATH,#'lightning_logs/default/version_23/checkpoints/epoch=0-step=29.ckpt',
    DO_TUNNIN,
    JUST_LOAD, # meaning False when training is needed,
    FIND_OPTMAL_LR,
    LEARNING_RATE,
    max_prediction_length,
    max_encoder_length,
    validation_porcentaje,
    n_epochs,
    batch_size): # from the data to the model .ckpt

    # --- --- --- --- --- BUILDING THE TEMPORAL FUSION TRANSFORMER
    validation_size = int((working_dataframe["INDEX"].max()-working_dataframe["INDEX"].min())*validation_porcentaje)
    validation_size = validation_size - validation_size%max_prediction_length
    assert(validation_size%max_prediction_length==0)
    # validation_size=5000
    training_cutoff = working_dataframe["INDEX"].max() - validation_size

    training = TimeSeriesDataSet(
        working_dataframe[lambda x: x.INDEX <= training_cutoff],
        time_idx="INDEX",
        target="price",
        group_ids=['symbol'], # much to do here
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length // max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=['symbol'],
        static_reals=[],
        # variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_categoricals=[],
        time_varying_known_reals=["INDEX"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["price"],
        target_normalizer=GroupNormalizer(
            groups=["symbol"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=False, # defaulto to True
        add_target_scales=False,# defaults to True,
        add_encoder_length=False,# defaults to True
    )
    validation = TimeSeriesDataSet(
        working_dataframe[lambda x: x.INDEX > training_cutoff], 
        time_idx="INDEX",
        target="price",
        group_ids=['symbol'], # much to do here
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length // max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=['symbol'],
        static_reals=[],
        # variable_groups={},  # group of categorical variables can be treated as one variable
        time_varying_known_categoricals=[],
        time_varying_known_reals=["INDEX"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["price"],
        target_normalizer=GroupNormalizer(
            groups=["symbol"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=False, # defaulto to True
        add_target_scales=False,# defaults to True,
        add_encoder_length=False,# defaults to True
    )

    # --- --- --- --- --- BUILD THE DATALOADERS
    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    # validation = TimeSeriesDataSet.from_dataset(
    #     dataset=training, 
    #     data=working_dataframe[lambda x: x.INDEX > training_cutoff], 
    #     predict=True, 
    #     stop_randomization=True)
    # validation = TimeSeriesDataSet(
    #     working_dataframe[lambda x: x.INDEX > training_cutoff],
    #     time_idx="INDEX",
    #     target="price",
    #     group_ids=['symbol'], # much to do here
    #     min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    #     max_encoder_length=max_encoder_length,
    #     min_prediction_length=max_prediction_length // max_prediction_length,
    #     max_prediction_length=max_prediction_length,
    #     static_categoricals=['symbol'],
    #     static_reals=[],
    #     # variable_groups={},  # group of categorical variables can be treated as one variable
    #     time_varying_known_categoricals=[],
    #     time_varying_known_reals=["INDEX"],
    #     time_varying_unknown_categoricals=[],
    #     time_varying_unknown_reals=["price"],
    #     target_normalizer=GroupNormalizer(
    #         groups=["symbol"], transformation="softplus"
    #     ),  # use softplus and normalize by group
    #     add_relative_time_idx=True,
    #     add_target_scales=True,
    #     add_encoder_length=True,
    # )
    # create dataloaders for model
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    # --- --- --- --- --- 
    if(JUST_LOAD):
        logging_fun("Using model path: {}!".format(ACTUAL_MODEL_PATH))
    else:
        # --- --- --- --- --- TRAIN THE MODEL
        # configure network and trainer
        pl.seed_everything(42)
        # --- --- --- --- --- FIND THE OPTIMAL LEARNING RATE
        if(FIND_OPTMAL_LR):
            trainer = pl.Trainer(
                gpus=0,
                # clipping gradients is a hyperparameter and important to prevent divergance
                # of the gradient for recurrent neural networks
                gradient_clip_val=0.07,
            )
            if(ACTUAL_MODEL_PATH is not None):
                tft = TemporalFusionTransformer.load_from_checkpoint(ACTUAL_MODEL_PATH)
            else:
                tft = TemporalFusionTransformer.from_dataset(
                    training,
                    # not meaningful for finding the learning rate but otherwise very important
                    learning_rate=0.03,#0.03,
                    hidden_size=108,  # most important hyperparameter apart from learning rate
                    # number of attention heads. Set to up to 4 for large datasets
                    attention_head_size=2,
                    dropout=0.1,  # between 0.1 and 0.3 are good values
                    hidden_continuous_size=8,  # set to <= hidden_size
                    output_size=21,  # 7 quantiles by default
                    loss=QuantileLoss(),
                    # reduce learning rate if no improvement in validation loss after x epochs
                    reduce_on_plateau_patience=4,
                )

            logging_fun(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
            res = trainer.tuner.lr_find(
                tft,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader,
                max_lr=10.0,
                min_lr=1e-6,
            )

            logging_fun(f"suggested learning rate: {res.suggestion()}")
            fig = res.plot(show=True, suggest=True)
            fig.show()
                # # SUGESTED TO BE: 0.12882495516931336
        # --- --- --- --- --- CONFIGURE NETWORK AND TRAINER
        # --- --- --- --- --- HIPERPARAMETER OPTIMIZATION
        if(DO_TUNNIN):
            from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
            # create study
            study = optimize_hyperparameters(
                train_dataloader,
                val_dataloader,
                model_path="optuna_test",
                n_trials=50,
                max_epochs=n_epochs,
                gradient_clip_val_range=(0.01, 1.0),
                hidden_size_range=(8, 128),
                hidden_continuous_size_range=(8, 128),
                attention_head_size_range=(1, 4),
                learning_rate_range=(0.001, 0.1),
                dropout_range=(0.05, 0.3),
                trainer_kwargs=dict(limit_train_batches=30),
                reduce_on_plateau_patience=4,
                use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
            )
            # --- --- --- --- --- LOAD BEST MODEL
            import pickle
            # save study results - also we can resume tuning at a later point in time
            with open("test_study.pkl", "wb") as fout:
                pickle.dump(study, fout)
            # show best hyperparameters
            logging_fun(study.best_trial.params)
            # {'gradient_clip_val': 0.07090602237711438, 'hidden_size': 108, 'dropout': 0.11158566880558941, 'hidden_continuous_size': 8, 'attention_head_size': 2, 'learning_rate': 0.02882430075946678}
            input("[TUNNING IS READY;] check the output params! STOP.")
        else: # simple regular training
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=10, verbose=False, mode="min")
            lr_logger = LearningRateMonitor()  # log the learning rate
            logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

            trainer = pl.Trainer(
                max_epochs=n_epochs,
                gpus=0,
                weights_summary="top",
                gradient_clip_val=0.07,
                limit_train_batches=30,  # coment in for training, running valiation every 30 batches
                # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
                callbacks=[lr_logger, early_stop_callback],
                logger=logger,
            )
            if(ACTUAL_MODEL_PATH is not None):
                tft = TemporalFusionTransformer.load_from_checkpoint(ACTUAL_MODEL_PATH)
            else:
                tft = TemporalFusionTransformer.from_dataset(
                    training,
                    learning_rate=LEARNING_RATE,
                    hidden_size=512,
                    attention_head_size=24,
                    dropout=0.1,
                    hidden_continuous_size=16,
                    output_size=21,  # 7 quantiles by default
                    loss=QuantileLoss(),#QuantileLoss(),
                    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
                    reduce_on_plateau_patience=4,
                )
            
            logging_fun(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
            # --- --- --- --- --- FIT THE NETWORK
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )        
        # --- --- --- --- --- EVALUATE PERFORMANCE
        # load the best model according to the validation loss
        # (given that we use early stopping, this is not necessarily the last epoch)
        ACTUAL_MODEL_PATH = trainer.checkpoint_callback.best_model_path
        logging_fun("best model path : {}".format(ACTUAL_MODEL_PATH))
        # --- --- --- --- --- 
        os.system('cp {} {}'.format(ACTUAL_MODEL_PATH,ALWAYS_SAVING_MODEL))
        print('saving a copy of the trained model into : {}'.format(ALWAYS_SAVING_MODEL))
        # --- --- --- --- --- 
    input("STOP! model is tranned!")
    tft_model = TemporalFusionTransformer.load_from_checkpoint(ALWAYS_SAVING_MODEL)
    # --- --- --- --- --- TEST THE MODEL
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    # # --- --- --- MAE
    # # calcualte mean absolute error on validation set
    # print("actuals : {}".format(actuals.shape))
    # predictions = tft_model.predict(val_dataloader)
    # print("predictions : {}".format(predictions.shape))
    # print("difference : {}".format(actuals - predictions))
    # logging_fun("Mean absolute error: {}".format((actuals - predictions).abs().mean()))
    # # --- --- --- 
    # # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions, x = tft_model.predict(val_dataloader, mode="raw", return_x=True)
    # --- --- --- --- ---
    import random
    for idx in range(30):  # plot 10    examples
        # print(x.keys())
        # print(raw_predictions.keys())
        # tft_model.plot_prediction(x, raw_predictions, idx=idx, plot_attention=True, add_loss_to_title=True)
        tft_model.plot_prediction(x, raw_predictions, idx=random.randint(0,actuals.size()[0]), plot_attention=True, add_loss_to_title=True)
    plt.show()
    # --- --- --- --- --- WORSE PREDICTIONS
    # calcualte metric by which to display
    predictions = tft_model.predict(val_dataloader)
    mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
    indices = mean_losses.argsort(descending=True)  # sort losses
    for idx in range(10):  # plot 10 examples
        tft_model.plot_prediction(
            x, raw_predictions, idx=indices[idx], plot_attention=True, add_loss_to_title=SMAPE(quantiles=tft_model.loss.quantiles)
        )
    plt.show()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- 
def load_tft(model_path):
    stime=time.time()
    tft_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    print("loaded file tft model in : {}".format(time.time()-stime))
    return tft_model
def relife_tft(
    working_dataframe,
    max_encoder_length,
    max_prediction_length,
    indicator, # s/l, indicating short or long (just for label porpousses)
    tft_model=None,
    model_path=None,
    ): # from the data to the image
    # --- --- --- --- --- LOAD THE MODEL
    assert(tft_model or model_path), "relife_tft function requires to set tft_model or model_path"
    if(tft_model is None):
        tft_model=load_tft(model_path)
    # --- --- --- --- --- 
    # --- --- --- --- --- TEST THE MODEL ON NEW DATA
    encoder_data = working_dataframe.tail(max_encoder_length)
    c_last_index = working_dataframe['INDEX'].iloc[-1] + 1
    decoder_data = pd.DataFrame([_c + c_last_index for _c in range(max_prediction_length)], columns=['INDEX'])
    decoder_data['symbol'] = working_dataframe['symbol'].iloc[-1]
    decoder_data['price'] = 0 # working_dataframe['price'].iloc[-1]
    predict_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    predict_data.reset_index(inplace=True)
    # print(encoder_data)
    # print(decoder_data)
    # print(predict_data.to_string())
    # input()
    new_raw_predictions, new_x = tft_model.predict(predict_data, mode="raw", return_x=True)
    # --- --- --- --- --- 
    fig_1, (ax1) = plt.subplots(1,1, gridspec_kw={'width_ratios': [1]})
    fig_2, (ax2,ax3) = plt.subplots(1,2, gridspec_kw={'width_ratios': [6,2]})
    # fig_1.legend().set_visible(False)
    # fig_2.legend().set_visible(False)
    # # --- --- 
    # print("new_x : {}".format(new_x.keys()))
    # print("new_x : {}".format([new_x[_].shape for _ in list(new_x.keys())]))
    # print("new_raw_predictions : {}".format(new_raw_predictions.keys()))
    # print("new_raw_predictions : {}".format([new_raw_predictions[_].shape for _ in list(new_raw_predictions.keys())]))
    # # --- --- 
    # print("new_raw_predictions : {}".format(new_raw_predictions['prediction'].shape))
    # print("new_raw_predictions : {}".format(new_raw_predictions['prediction']))
    # # --- --- 
    # # print("new_raw_predictions : {}".format(new_raw_predictions['decoder_variables'].shape))
    # # print("new_raw_predictions : {}".format(new_raw_predictions['decoder_variables']))
    # print("new_x : {}".format(new_x['decoder_lengths'].shape))
    # print("new_x : {}".format(new_x['decoder_lengths']))
    # --- --- 
    # input()
    # --- --- 
    # ax1.margins(x=0,y=-0.25)
    # --- --- 
    # --- ---
    f_ret=tft_model.plot_prediction(new_x, new_raw_predictions, idx=0, ax=ax1, show_future_observed=False)
    ax1.get_lines()[0].set_color("white")
    ax1.get_lines()[0].set_linewidth(0.8)
    ax1.get_lines()[0].set_alpha(0.8)
    handles, labels = ax1.get_legend_handles_labels()
    # print(labels)
    # handles[0].visible=False
    # handles[1].visible=False
    # print(f_ret)
    # print(handles)
    # f_ret.legend(labelcolor='white',frameon=False,framealpha=0.1,facecolor='black')
    # ax1.legend(labelcolor='red',frameon=False,framealpha=0.1,loc=10)
    # f_ret.legend().remove()
    # handles[1].remove()
    # print(handles)
    # input()
    # ax1.axis(xmin=0,xmax=new_x['decoder_cont'].size()[1])#,ymin=,ymax=)
    # ax1.margins(x=0,y=0.25)
    # ax1.axis(xmin=-new_x['encoder_cont'].size()[1],xmax=0,ymin=0.999*min(encoder_data['price'].to_list()),ymax=1.001*max(encoder_data['price'].to_list()))
    # ax1.get_lines()[1].set_color("white")
    # tft_model.plot_prediction(new_x, new_raw_predictions, idx=0, ax=ax2, show_future_observed=False)
    # --- ---
    # ax3.plot(new_raw_predictions['prediction'][0,:,0].detach().cpu())
    # print(new_raw_predictions['prediction'].mean(0).shape)
    # print(new_raw_predictions['prediction'].mean(1).shape)
    # print(new_raw_predictions['prediction'].shape)
    # --- --- --- --- 
    def defirst(inp):
        return (inp - inp[0])/(inp.max()-inp.min())
    def demean(inp):
        return (inp - inp.mean())/(inp.max()-inp.min())
    def delast(inp):
        return (inp - inp[-1])/(inp.max()-inp.min())
    # --- --- --- --- 
    c_delast=delast(np.array(encoder_data['price'].to_list()))
    p_defirst=defirst(ax1.get_lines()[1]._y)
    # --- --- --- --- 
    ax2.plot(c_delast,color='green',linewidth=0.45)
    # --- --- --- --- 
    ax3.plot(p_defirst,color='orange')
    for ctx_ in range(new_raw_predictions['prediction'].size()[2]):
        ax3.plot(defirst(new_raw_predictions['prediction'][0,:,ctx_].detach().cpu()),color='r',alpha=0.15)
    c_defirst=defirst(new_raw_predictions['prediction'].mean(2)[0].detach().cpu())
    ax3.plot(c_defirst,color='r',linewidth=1.25,alpha=1.0)
    ax3.plot([0]*len(c_defirst),color='w',linewidth=0.5,alpha=0.8)
    # --- --- --- --- 
    ax2.axis(ymin=min(-1.0,min(c_defirst),min(p_defirst)),ymax=max(1.0,max(c_defirst),max(p_defirst)))
    ax3.axis(ymin=min(-1.0,min(c_defirst),min(p_defirst)),ymax=max(1.0,max(c_defirst),max(p_defirst)))
    # --- --- --- --- 
    # ax3.plot(new_raw_predictions['prediction'][0,:,1].detach().cpu())
    # ax3.plot(new_raw_predictions['prediction'][0,:,2].detach().cpu())
    # ax3.plot(new_raw_predictions['prediction'][0,:,3].detach().cpu())
    # ax3.plot(new_raw_predictions['prediction'][0,:,4].detach().cpu())
    # ax3.plot(new_raw_predictions['prediction'][0,:,5].detach().cpu())
    # ax3.plot(new_raw_predictions['prediction'][0,:,6].detach().cpu())
    # --- --- 
    # plt.legend(frameon=False)
    fig_1.patch.set_facecolor((114/255,47/255,55/255))
    fig_2.patch.set_facecolor((0,0,0))
    ax1.set_facecolor((114/255,47/255,55/255))
    ax2.set_facecolor((0,0,0))
    ax3.set_facecolor((0,0,0))
    # ax1.legend().set_visible(False)
    # ax2.legend().set_visible(False)
    # ax3.legend().set_visible(False)
    # ax1.twinx().set_xticklabels([])
    # ax2.twinx().set_xticklabels([])
    # ax1.twinx().set_xticks([])
    # ax2.twinx().set_xticks([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    # ax2.legend().set_visible(False)
    # ax3.legend().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()
    # plt.show()
    # --- --- 
    import uuid
    wlot_path="{}/{}".format(ON_FILE_WLOT_FOLDER,working_dataframe['symbol'].iloc[-1])
    assert_folder(wlot_path)
    # figname=os.path.join(wlot_path,"{}.png".format(working_dataframe['symbol'].iloc[-1]))
    # figname=os.path.join(wlot_path,"{}-{}.png".format(working_dataframe['symbol'].iloc[-1],uuid.uuid4()))
    figname_1=os.path.join(wlot_path,"{}.{}0.png".format(working_dataframe['symbol'].iloc[-1],indicator))
    figname_2=os.path.join(wlot_path,"{}.{}1.png".format(working_dataframe['symbol'].iloc[-1],indicator))
    fig_1.savefig(figname_1, dpi=tft_dpi, facecolor='black', edgecolor='black',
        orientation='portrait', format=None, transparent=False, 
        bbox_inches='tight', pad_inches=0.0,metadata=None)
    fig_2.savefig(figname_2, dpi=tft_dpi, facecolor='black', edgecolor='black',
        orientation='portrait', format=None, transparent=False, 
        bbox_inches='tight', pad_inches=0.0,metadata=None)
    return figname_1,figname_2
    # --- --- --- --- --- 
    # --- --- --- --- --- 
if __name__=='__main__':
    print("[warning:] implement!")