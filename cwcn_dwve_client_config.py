# --- --- --- --- 
# cwcn_dwve_client_config
# --- --- --- --- 
import os
import logging
import random
import torch
import time
from datetime import datetime, timedelta
import numpy as np
from ray import tune

# --- --- --- --- 
torch.distributions.Distribution.set_default_validate_args(True)
# --- --- --- --- 
class dwve_ethusdtperp_configuration:
    # --- --- --- --- 
    SYMBOL = 'ETHUSDTPERP' # 'BTCUSDTPERP'
    DATA_ROOT = os.path.normpath('{}/../dowaave_data_farm/FARM'.format(os.path.dirname(os.path.abspath(__file__))))
    # DATA_ROOT = os.path.normpath('{}/../data_farm/FARM'.format(os.path.dirname(os.path.abspath(__file__))))
    DATA_FILE='{}/{}.poloniex_ticker_data'.format(DATA_ROOT,SYMBOL)
    DATA_USED_COLUMNS = ['symbol','price','ts','sequence'] #['symbol','sequence','side','size','price','bestBidSize','bestBidPrice','bestAskPrice','tradeId','ts','bestAskSize'], 
    TRANSFORM_CANDLE = {
        'candle_flag':True,
        # 'candle_step':(20,(lambda x: abs(x['mean'])),'delta_price','price') # every 20 'std' of 'delta_price' aplied to 'price'
        'candle_item':'price',
        'candle_step':1.0,
        'prom_candle_step_multiplier':10.0,
    }
    train_load_n_seq = 2000000
    # --- --- --- --- 
    gss_training_iter=  100
    gss_learning_rate=  0.1
    gss_c_horizon= 50
    gss_c_horizon_delta= 0.5
    gss_c_iterations= 0xFFFF
    gss_c_backlash= -0.85
    gss_c_seq_size=100
    # gss_dpi=100
    # --- --- --- --- 
    tft_ALWAYS_SAVING_MODEL = 'lightning_logs/always_saving_tft.ckpt'
    tft_ACTUAL_MODEL_PATH = 'lightning_logs/always_saving_tft.ckpt'
    tft_DO_TUNNIN = False
    tft_FIND_OPTMAL_LR = False
    tft_LEARNING_RATE=0.01 # 0.12
    tft_max_prediction_length = 10
    tft_max_encoder_length = 100
    tft_validation_porcentaje = 0.12
    tft_n_epochs=5
    tft_batch_size = 64  # set this between 32 to 128
    tft_c_seq_size=tft_max_encoder_length
    # tft_dpi=100
    # --- --- --- --- 
    LEVERAGE = '100'
    # --- --- --- --- s
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
# --- --- --- --- --- THIS IS THE PLACE WHERE SYMBOL CONFIGURATION IS DONE
# --- --- --- --- --- 
dwve_instrument_configuration = dwve_ethusdtperp_configuration # < --- --- --- --- --- --- --- --- --
# --- --- --- --- 
# --- --- --- --- --- 
# --- --- --- --- --- 
# assert(os.environ['CWCN_CONFIG']==os.path.realpath(__file__)), '[ERROR:] wrong configuration import'
# ... #FIXME assert comulative munaajpi is in place, seems ok, gae takes the account
# --- --- --- --- 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- --- --- --- 

# --- --- 
# --- --- --- --- 
# --- --- --- --- --- --- --- ---  
PAPER_INSTRUMENT = False #FIXME # < --- --- --- --- FAKE / REAL ; (bool) flag
# --- --- --- --- 
# --- --- --- 
# --- --- 
# SYMBOL_INSTRUMENT = 'BTCUSDTPERP' #'BCHUSDTPERP' #'BTCUSDTPERP' #'SINE-100'#'BTCUSDTPERP' #'ADAUSDTPERP'/'BTCUSDTPERP'
# --- --- --- --- 
# --- --- --- 
# --- --- 
CLIENT_PROTOL = 'http'
CLIENT_DIR = 'cuwacunu'
CLIENT_PORT = 8080
CLIENT_URL = '{}://{}:{}/'.format(CLIENT_PROTOL,CLIENT_DIR,CLIENT_PORT)
# --- --- 
# --- --- --- --- --- --- --- ---  
# --- --- --- --- 
# --- --- --- AUTH
# --- --- 
VALID_NODES = {
    '20:1a:06:3a:b4:84':'cuwacunu',
    'f0:2f:74:1e:fb:45':'jrmedallo'
}
# --- --- 
ACTIVE_SYMBOLS = [
    'BTCUSDTPERP',
    'ETHUSDTPERP',
    # 'BSVUSDTPERP',
    # 'BCHUSDTPERP',
    # 'YFIUSDTPERP',
    # 'UNIUSDTPERP',
    # 'LINKUSDTPERP',
    # 'TRXUSDTPERP',
    # 'XRPUSDTPERP',
    # 'XMRUSDTPERP',
    # 'LTCUSDTPERP',
    # 'DOTUSDTPERP',
    # 'DOGEUSDTPERP',
    # 'FILUSDTPERP',
    # 'BNBUSDTPERP',
    # '1000SHIBUSDTPERP',
    # 'BTTUSDTPERP',
    # 'ADAUSDTPERP',
    # 'SOLUSDTPERP',
    # 'LUNAUSDTPERP',
    # 'ICPUSDTPERP',
]
# --- --- 
WEB_SOCKET_SUBS=['/contractAccount/wallet'] # account updates
WEB_SOCKET_SUBS+=['/contractMarket/ticker:{}'.format(__) for __ in ACTIVE_SYMBOLS] # price information in real time
# --- --- 
AHDO_FIELDS = [
        # --- --- WALLET
        'unrealisedPNL',
        # "frozenFunds",
        # "trialAccountOverview": {
        #     "totalBalance": 0.0,
        #     "holdBalance": 0.0,
        #     "availableBalance": 0.0
        # },
        "currency",
        "accountEquity",
        # "positionMargin",
        # "orderMargin",
        # "marginBalance",
        "availableBalance",
        # "realisedGrossPnl",
        "symbol",
        # "crossMode",
        # "liquidationPrice",
        # "posLoss",
        # "avgEntryPrice",
        "unrealisedPnl",
        # "markPrice",
        # "autoDeposit",
        # "posMargin",
        # "riskLimit",
        # "unrealisedCost",
        # "posComm",
        # "posMaint",
        # "posCost",
        # "id",
        # "maintMarginReq",
        # "bankruptPrice",
        # "realisedCost",
        # "markValue",
        # "posInit",
        "realisedPnl",
        # "maintMargin",
        # "realLeverage",
        # "currentCost",
        # "settleCurrency",
        # "openingTimestamp",
        "currentQty",
        # "delevPercentage",
        # "currentComm",
        # "realisedGrossCost",
        # "isOpen",
        # "posCross",
        # "currentTimestamp",
        # "unrealisedRoePcnt",
        # "unrealisedPnlPcnt",
        # --- --- COIN TICK
        # 'symbol':, # allready declared
        'sequence',
        # 'side',
        # 'size',
        'price',
        # 'bestBidSize',
        # 'bestBidPrice',
        # 'bestAskPrice',
        # 'tradeId',
        'ts',
        # 'bestAskSize',
        # --- --- EXPANSIONS 
        'date', 
]
# --- --- 
# --- --- 

class CWCN_DUURUVA_CONFIG:
    # --- --- --- 
    DUURUVA_MAX_COUNT = 100
    DUURUVA_READY_COUNT = 50
    MIN_STD = 0.000001
    # --- --- --- 


# --- --- --- --- 
class CWCN_FARM_CONFIG:
    FARM_FOLDER = '{}/../dowaave_data_farm/FARM'.format(os.path.dirname(os.path.abspath(__file__)))
    FARM_SYMBOLS = [
        # 'BTCUSDTPERP',
        # 'ETHUSDTPERP',
        # 'BSVUSDTPERP',
        # 'BCHUSDTPERP',
        # 'YFIUSDTPERP',
        # 'UNIUSDTPERP',
        # 'LINKUSDTPERP',
        # 'TRXUSDTPERP',
        # 'XRPUSDTPERP',
        # 'XMRUSDTPERP',
        # 'LTCUSDTPERP',
        # 'DOTUSDTPERP',
        # 'DOGEUSDTPERP',
        # 'FILUSDTPERP',
        # 'BNBUSDTPERP',
        # '1000SHIBUSDTPERP',
        # 'BTTUSDTPERP',
        # 'ADAUSDTPERP',
        # 'SOLUSDTPERP',
        # 'LUNAUSDTPERP',
        # 'ICPUSDTPERP',
        ]
    FARM_DATA_EXTENSION = '.poloniex_ticker_data'
    FRONT_WALLET_FILE = '{}/../WALLET/WALLET.poloniex_wallet_data'.format(os.path.dirname(os.path.abspath(__file__)))
# --- --- --- --- 
# print(CWCN_FARM_CONFIG.FARM_FOLDER)
# --- --- --- --- 
class CWCN_CURSOR: #FIXME not in use
    UP='\033[A'
    DOWN='\033[B'
    LEFT='\033[D'
    RIGHT='\033[C'
    CLEAR_LINE='\033[K'
    CARRIER_RETURN='\r'
    NEW_LINE='\n'
# --- --- --- --- 
    

# 'dist' : (lambda : torch.distributions.bernoulli.Bernoulli)
# 'args' : (lambda : ['probs'])

# munamunaake_torch_distributions={
# '__name__':{'':None},
# '__doc__':{'':None},
# '__package__':{'':None},
# '__loader__':{'':None},
# '__spec__':{'':None},
# '__path__':{'':None},
# '__file__':{'':None},
# '__cached__':{'':None},
# '__builtins__':{'':None},
# 'constraints':{'':None},
# 'utils':{'':None},
# 'distribution':{'':None},
# 'exp_family':{'':None},
# 'bernoulli':{'':None},
# 'Bernoulli':{'':None},
# 'dirichlet':{'':None},
# 'beta':{'':None},
# 'Beta':{'':None},
# 'binomial':{'':None},
# 'Binomial':{'':None},
# 'categorical':{'':None},
# 'Categorical':{'':None},
# 'cauchy':{'':None},
# 'Cauchy':{'':None},
# 'gamma':{'':None},
# 'chi2':{'':None},
# 'Chi2':{'':None},
# 'transforms':{'':None},
# 'constraint_registry':{'':None},
# 'biject_to':{'':None},
# 'transform_to':{'':None},
# 'continuous_bernoulli':{'':None},
# 'ContinuousBernoulli':{'':None},
# 'Dirichlet':{'':None},
# 'Distribution':{'':None},
# 'ExponentialFamily':{'':None},
# 'exponential':{'':None},
# 'Exponential':{'':None},
# 'fishersnedecor':{'':None},
# 'FisherSnedecor':{'':None},
# 'Gamma':{'':None},
# 'geometric':{'':None},
# 'Geometric':{'':None},
# 'uniform':{'':None},
# 'independent':{'':None},
# 'transformed_distribution':{'':None},
# 'gumbel':{'':None},
# 'Gumbel':{'':None},
# 'half_cauchy':{'':None},
# 'HalfCauchy':{'':None},
# 'normal':{'':None},
# 'half_normal':{'':None},
# 'HalfNormal':{'':None},
# 'Independent':{'':None},
# 'laplace':{'':None},
# 'multivariate_normal':{'':None},
# 'lowrank_multivariate_normal':{'':None},
# 'one_hot_categorical':{'':None},
# 'pareto':{'':None},
# 'poisson':{'':None},
# 'kl':{'':None},
# 'kl_divergence':{'':None},
# 'register_kl':{'':None},
# 'kumaraswamy':{'':None},
# 'Kumaraswamy':{'':None},
# 'Laplace':{'':None},
# 'lkj_cholesky':{'':None},
# 'LKJCholesky':{'':None},
# 'log_normal':{'':None},
# 'LogNormal':{'':None},
# 'logistic_normal':{'':None},
# 'LogisticNormal':{'':None},
# 'LowRankMultivariateNormal':{'':None},
# 'mixture_same_family':{'':None},
# 'MixtureSameFamily':{'':None},
# 'multinomial':{'':None},
# 'Multinomial':{'':None},
# 'MultivariateNormal':{'':None},
# 'negative_binomial':{'':None},
# 'NegativeBinomial':{'':None},
# 'Normal':{'':None},
# 'OneHotCategorical':{'':None},
# 'OneHotCategoricalStraightThrough':{'':None},
# 'Pareto':{'':None},
# 'Poisson':{'':None},
# 'relaxed_bernoulli':{'':None},
# 'RelaxedBernoulli':{'':None},
# 'relaxed_categorical':{'':None},
# 'RelaxedOneHotCategorical':{'':None},
# 'studentT':{'':None},
# 'StudentT':{'':None},
# 'TransformedDistribution':{'':None},
# 'AbsTransform':{'':None},
# 'AffineTransform':{'':None},
# 'CatTransform':{'':None},
# 'ComposeTransform':{'':None},
# 'CorrCholeskyTransform':{'':None},
# 'ExpTransform':{'':None},
# 'IndependentTransform':{'':None},
# 'LowerCholeskyTransform':{'':None},
# 'PowerTransform':{'':None},
# 'ReshapeTransform':{'':None},
# 'SigmoidTransform':{'':None},
# 'TanhTransform':{'':None},
# 'SoftmaxTransform':{'':None},
# 'StackTransform':{'':None},
# 'StickBreakingTransform':{'':None},
# 'Transform':{'':None},
# 'identity_transform':{'':None},
# 'Uniform':{'':None},
# 'von_mises':{'':None},
# 'VonMises':{'':None},
# 'weibull':{'':None},
# 'Weibull':{'':None},
# '__all__':{'':None},
# }


# --- --- --- --- 
# logging.info('Loading configuration file {}'.format(os.environ['CWCN_CONFIG']))
logging.info('Loading from kalamar source the configuration file {}. '.format(os.path.realpath(__file__)))
# --- --- --- --- 
