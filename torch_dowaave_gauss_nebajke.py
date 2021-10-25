import os
import pandas as pd
import numpy as np
import logging
# --- --- ---
ON_FILE_WLOT_FOLDER='./gauss_dumps'
gss_dpi=120
# --- --- ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] :: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
# --- --- ---
def assert_folder(_f_path):
    if(not os.path.isdir(_f_path)):
        os.mkdir(_f_path)
assert_folder(ON_FILE_WLOT_FOLDER)
for _f in os.listdir(ON_FILE_WLOT_FOLDER):
    for _f2 in os.listdir(os.path.join(ON_FILE_WLOT_FOLDER,_f)):
        print("[PURGIN FILE:]",os.path.join(ON_FILE_WLOT_FOLDER,_f,_f2))
        os.remove(os.path.join(ON_FILE_WLOT_FOLDER,_f,_f2))
# --- --- ---
class DATA_KIJTYU:
    def __init__(self,_folder):
        logging.info("Loading data from : "+_folder)
        self.__exe_discriminator='csv'
        self.__folder=_folder
        self._load_folder_()
        self.__data={}
        self.__data_size={}
        self.__c_idc=None
    def _load_folder_(self):
        __files_list=[os.path.join(self.__folder,_) for _ in os.listdir(self.__folder) if os.path.isfile(os.path.join(self.__folder,_)) and _.split('.')[-1]==self.__exe_discriminator]
        self.__files_dict=dict([(_.split('.')[-2].split('/')[-1],_) for _ in __files_list])
        self.__items_list=list(self.__files_dict.keys())
        # logging.info(self.__items_list)
    def _c_data_size_(self):
        return self.__data_size[self.__c_idc]
    def _load_data_(self,_idc,_list_cols=None):
        try:
            self.__c_idc=_idc # index with respect to loaded folder files order
            # --- --- --- --- --- READ ORIGINAL FILE
            self.loaded_dataframe[_idc]=pd.read_csv(
                self.__files_dict[_idc],
                usecols=dwvc._list_cols)
            logging.info("Loaded <{}> data <{}>".format(self.__exe_discriminator,_idc))
            if(pd.DataFrame(self.loaded_dataframe[_idc]['Close']).isnull().any().any()):
                logging.warning("Data <{}> has None values".format(_idc))
            # logging.info(self.loaded_dataframe[_idc].head())
            logging.info(self.loaded_dataframe[_idc]['Close'])
        except Exception as e:
            logging.error("Problem loading data <{}> : {}".format(_idc,e))
        self.__data_size[_idc]=len(self.loaded_dataframe[_idc])
        return self.loaded_dataframe[_idc]
    def _transform_data_(self,_candle_config=None):
        # --- --- --- --- --- SET UP THE PRICE DUURUVA
        price_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=1,_wrapper_duuruva_normalize='not', _d_name='price_duuruva')
        # --- --- --- --- --- EXPAND delta_sequence
        self.loaded_dataframe['delta_sequence'] = self.loaded_dataframe['sequence'].diff().fillna(0)
        self.loaded_dataframe['delta_price'] = self.loaded_dataframe['price'].diff().fillna(0)
        self.loaded_dataframe['delta_ts'] = self.loaded_dataframe['ts'].diff().fillna(0)
        self.loaded_dataframe['INDEX'] = self.loaded_dataframe.index # self.loaded_dataframe.reset_index(level=0,inplace=True) # add index column

        if(_candle_configis is not None and _candle_config['candle_flag']):
            def get_applied_item(df,_index):
                return df.iloc[_index][_candle_config['candle_step'][3]]
            def get_candle_step(df):
                return _candle_config['candle_step'][0]*_candle_config['candle_step'][1](df.describe()[_candle_config['candle_step'][2]])
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
        # plt.figure()
        # self.loaded_dataframe['price'].plot()
        # --- --- --- --- --- DUURUVA TRANSFORM THE PRICE
        self.loaded_dataframe['price']=self.loaded_dataframe['price'].apply(lambda x: price_duuruva._duuruva_value_wrapper_(x))
        return self.loaded_dataframe
# --- --- ---
import math
import torch
import gpytorch
import matplotlib.pyplot as plt
# this is for running the notebook in our testing framework
import os
import imageio

# --- --- ---

# Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 10)
# True function is sin(2*pi*x) with Gaussian noise

# train_y = torch.pow(torch.sin(train_x) * (36 * math.pi),0.1)*torch.tanh(train_x * (16 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.0004)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.LinearMean(1)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GAUSSIAN_WIKIMYEI:
    def __init__(self):
        # initialize likelihood and model
        self.model = None
    def _set_jk_optimizer_(self,learning_rate):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters
    def _set_knowledge_base_(self,jkimyei_x,jkimyei_y,learning_rate):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(jkimyei_x, jkimyei_y, self.likelihood)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self._set_jk_optimizer_(learning_rate)
    def _jkimyei_(self,jkimyei_x,jkimyei_y,learning_rate,training_iter):
        # Exact GPModel
        # if(self.model is None):
        self._set_knowledge_base_(jkimyei_x,jkimyei_y,learning_rate)
        # Use the adam optimizer
        # "Loss" for GPs - the marginal log likelihood
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()
        for i_ in range(training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(jkimyei_x)
            # Calc loss and backprop gradients
            loss = -self.mll(output, jkimyei_y)
            loss.backward()
            if(i_%50==0):
                logging.info('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i_, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
            self.optimizer.step()
    def _uwaabo_hash_(self,uwaabo_x):
        # f_preds = model(uwaabo_x)
        # y_preds = likelihood(model(uwaabo_x))
        # f_mean = f_preds.mean
        # f_var = f_preds.variance
        # f_covar = f_preds.covariance_matrix
        # f_samples = f_preds.sample(sample_shape=torch.Size(1000,))
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        # print(uwaabo_x)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(uwaabo_x))
        return observed_pred
    
class GAUSSIAN_WLOTER:
    def __init__(self,wlot_itm):
        self.wlot_itm=wlot_itm
        self.out_wlot_folder_itm=os.path.join(ON_FILE_WLOT_FOLDER,self.wlot_itm)
        self.itx_ctx=0
        assert_folder(self.out_wlot_folder_itm)
        # --- --- ---
    def __purge_wlot_folder__(self):
        assert_folder(self.out_wlot_folder_itm)
        logging.warning("Purging [folder] : <{}>".format(self.out_wlot_folder_itm))
        for f_ in os.listdir(self.out_wlot_folder_itm):
            ff_=os.path.join(self.out_wlot_folder_itm,f_)
            logging.warning("\t - Purging [file] : <{}>".format(ff_))
            os.remove(ff_)
    def __wlot_gif__(self):
        imags=[]
        for f_ in os.listdir(self.out_wlot_folder_itm):
            ff_=os.path.join(self.out_wlot_folder_itm,f_)
            imags.append(imageio.imread(ff_))
            logging.info("\t - Wppending \t{}\t[GIF] : <{}>".format(self.wlot_itm,ff_))
        gf_path=os.path.join(self.out_wlot_folder_itm,"{}.{}".format(self.wlot_itm,"gif"))
        imageio.mimsave(gf_path,imags, format='GIF', duration=1)
        return gf_path
    def __wlot_graph__(self,uwaabo_x,uwaabo_y,jkimyei_x,jkimyei_y,truth_x,truth_y,ax_=None):
        with torch.no_grad():
            # Initialize plot
            ax_flag=False
            if(ax_ is None):
                ax_flag=True
                f_, ax_ = plt.subplots(1, 1)
                # f_.canvas.manager.full_screen_toggle()
                f_.patch.set_facecolor((0,0,0))
            # ax_.set_title("{} - {}".format(ax_.get_title(),self.wlot_itm),color=(0,0,0))
            ax_.set_facecolor((0,0,0))
            ax_.spines['bottom'].set_color('black')
            ax_.spines['top'].set_color('black')
            ax_.spines['right'].set_color('black')
            ax_.spines['left'].set_color('black')
            ax_.xaxis.label.set_color('black')
            ax_.yaxis.label.set_color('black')
            ax_.tick_params(colors='black',which='both')
            # Get upper and lower confidence bounds
            lower, upper = uwaabo_y.confidence_region()
            # Plot future data
            ax_.plot(truth_x.numpy(), truth_y.numpy(), 'g', linewidth=0.3)
            # ax_.plot(truth_2_x.numpy(), truth_2_y.numpy(), 'y')
            # Plot predictive means as blue line
            ax_.plot(uwaabo_x.numpy(), uwaabo_y.mean.numpy(), 'r', linewidth=0.6)
            # Plot training data as black stars
            ax_.plot(jkimyei_x.numpy(), jkimyei_y.numpy(), 'w', linewidth=0.8,alpha=0.6)
            # Shade between the lower and upper confidence bounds
            ax_.fill_between(uwaabo_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.2)
            # ax_.set_ylim([-3, 3])
            # ax_.legend(['Kijtiyu Alliu, Uwaabo Mean, Unknown Alliu, Uwaabo Confidence'])
            # import uuid
            # # plt.show()
            # ax_.set_title(self.wlot_itm,color="white")
            figname=os.path.join(self.out_wlot_folder_itm,"{}.{}.png".format(self.wlot_itm,self.itx_ctx))
        self.itx_ctx+=1
        if(ax_flag):
            return f_, ax_, figname
        else:
            return None,ax_, figname
# --- --- ---


# --- --- ---
def relife_gauss(
    working_dataframe,
    active_dimension,
    active_coin,
    training_iter = 100,
    learning_rate = 0.1,
    c_horizon=50,
    c_horizon_delta=0.5,
    c_iterations=0xFFFF,
    c_backlash=-0.85,
    ):
    # --- --- ---
    # ON_FILE_ACTIVE_COIN='BTCUSDTPERP'
    # ON_FILE_ACTIVE_DATA_FOLDER='../data_farm/FARM'
    # data_kijtyu=DATA_KIJTYU(_folder=ON_FILE_ACTIVE_DATA_FOLDER)
    # working_dataframe=data_kijtyu._load_data_(ON_FILE_ACTIVE_COIN,_list_cols=['symbol','price','ts','sequence'])
    # working_dataframe=data_kijtyu._transform_data_(_candle_config={
    #     'candle_flag':True,
    #     'candle_step':(200,(lambda x: abs(x['mean'])),'delta_price','price') # every 20 'std' of 'delta_price' aplied to 'price'
    # })
    # --- --- ---
    
    # --- --- ---
    
    # --- --- ---
    working_dataframe[active_dimension]=(working_dataframe[active_dimension]-working_dataframe[active_dimension].mean())/(working_dataframe[active_dimension].std()+0.00001)
    # --- --- ---
    
    seq_size=len(working_dataframe.index)
    c_future=int(c_horizon_delta*c_horizon-seq_size%c_horizon)
    # --- --- 
    truth_x=torch.linspace(0, seq_size+c_future, seq_size+c_future)

    truth_y=torch.FloatTensor(list(working_dataframe[active_dimension][:])+[np.nan]*c_future)
    # ---
    # --- --- ---
    gw=GAUSSIAN_WLOTER(wlot_itm=active_coin)
    # gw.__purge_wlot_folder__()
    # --- --- ---
    gwk=GAUSSIAN_WIKIMYEI()
    # --- --- ---
    BUILD_GIF = False
    if(BUILD_GIF):
        _idx_=0
        while(int(((c_backlash if _idx_!=0 else 0)+_idx_)*c_horizon)<seq_size):
            if(c_iterations<_idx_):
                break
            # --- 
            if(int((1+c_backlash+c_horizon_delta+_idx_)*c_horizon)<seq_size):
                c_index_init=int(((c_backlash if _idx_!=0 else 0)+_idx_)*c_horizon)
                c_index_final=int((1+c_backlash+c_horizon_delta+_idx_)*c_horizon)
                c_index_horizon=int((1+c_backlash+_idx_)*c_horizon)
                # --- 
                uwaabo_x=truth_x[c_index_init: c_index_final]
                # --- 
                jkimyei_x=truth_x[c_index_init: c_index_horizon]
                jkimyei_y=truth_y[c_index_init: c_index_horizon]
                # --- 
                # print("STEP:",c_index_final,seq_size+100)
                # print("c_index_init: ",c_index_init)
                # print("c_index_final: ",c_index_final)
                # print("c_index_horizon: ",c_index_horizon)
                # --- 
            else:
                # --- 
                uwaabo_x=truth_x[-c_horizon:]
                # --- 
                jkimyei_x=truth_x[-c_horizon-c_future:-c_future]
                jkimyei_y=truth_y[-c_horizon-c_future:-c_future]
                # --- 
                # print("uwaabo_x",uwaabo_x)
                # print("jkimyei_x",jkimyei_x)
                # print("jkimyei_y",jkimyei_y)
                # input("stop")
            
            gwk._set_knowledge_base_(jkimyei_x,jkimyei_y,learning_rate=learning_rate)
            gwk._jkimyei_(jkimyei_x,jkimyei_y,learning_rate=learning_rate,training_iter=training_iter)
            uwaabo_y=gwk._uwaabo_hash_(uwaabo_x)
            # print(uwaabo_y.sample())
            # --- 
            f, ax, figname = gw.__wlot_graph__(uwaabo_x,uwaabo_y,jkimyei_x,jkimyei_y,truth_x,truth_y)
            # --- --- ---
            _idx_+=1
        try:
            gf_path = gw.__wlot_gif__()
        except:
            logging.error("[Unable to build gif]")
        return gf_path
    else:
        level_delta=5.0
        # --- 
        # --- 
        uwaabo_x=truth_x[-c_horizon-c_future:]
        # --- 
        # --- --- --- (1)
        jkimyei_x=truth_x[-c_horizon-c_future:-c_future]
        jkimyei_y=truth_y[-c_horizon-c_future:-c_future]
        # --- 
        gwk._set_knowledge_base_(jkimyei_x,jkimyei_y,learning_rate=learning_rate)
        gwk._jkimyei_(jkimyei_x,jkimyei_y,learning_rate=learning_rate,training_iter=training_iter)
        uwaabo_y=gwk._uwaabo_hash_(uwaabo_x)
        # --- 
        f, ax, figname = gw.__wlot_graph__(uwaabo_x,uwaabo_y,jkimyei_x,jkimyei_y,truth_x,truth_y)
        # --- --- --- (2)
        jkimyei_x=truth_x[:-c_future]
        jkimyei_y=truth_y[:-c_future] + level_delta
        # --- 
        gwk._set_knowledge_base_(jkimyei_x,jkimyei_y,learning_rate=learning_rate)
        gwk._jkimyei_(jkimyei_x,jkimyei_y,learning_rate=learning_rate,training_iter=training_iter)
        uwaabo_y=gwk._uwaabo_hash_(uwaabo_x)
        # --- 
        gw.__wlot_graph__(uwaabo_x,uwaabo_y,jkimyei_x,jkimyei_y,truth_x,truth_y,ax)

        # figname=os.path.join(gw.out_wlot_folder_itm,"{}.{}-{}.png".format(active_coin,1,uuid.uuid4()))
        f.savefig(figname, dpi=gss_dpi, facecolor='black', edgecolor='black',
            orientation='portrait', format=None, transparent=False, 
            bbox_inches='tight', pad_inches=0.0,metadata=None)

        return figname, figname
