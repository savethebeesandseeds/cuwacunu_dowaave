# --- --- --- 
import os
import matplotlib.pyplot as plt
# --- --- --- 
import cwcn_dwve_client_config as dwvcc
from cwcn_dwve_client_config import dwve_instrument_configuration as dwvic
# --- --- --- 
ON_FILE_WLOT_FOLDER='./hrz_dumps'
hrz_dpi=120
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
class HORIZON_WLOTER:
    def __init__(self,wlot_itm,wlot_dimension):
        self.wlot_itm=wlot_itm
        self.wlot_dimension=wlot_dimension
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
    def __wlot_graph__(self,working_dataframe):
        # Initialize plot
        ax_flag=True
        f_, ax_ = plt.subplots(1, 1)
        # f_.canvas.manager.full_screen_toggle()
        f_.patch.set_facecolor((0,0,0))
        ax_.set_title("{} - {}".format(ax_.get_title(),self.wlot_itm),color=(0,0,0))
        ax_.set_facecolor((0,0,0))
        ax_.spines['bottom'].set_color('black')
        ax_.spines['top'].set_color('black')
        ax_.spines['left'].set_color('black')
        ax_.spines['right'].set_color('white')
        ax_.xaxis.label.set_color('white')
        ax_.yaxis.label.set_color('white')
        ax_.tick_params(colors='white',which='both')
        # --- --- --- ---             
        ax_.plot(working_dataframe.index.tolist(),working_dataframe[self.wlot_dimension].tolist(), 'g', linewidth=0.5, alpha=1.0)
        ax_.plot(working_dataframe.index.tolist()[::50],working_dataframe[self.wlot_dimension].tolist()[::50], 'r', linewidth=0.3, alpha=1.0)
        ax_.plot(working_dataframe.index.tolist()[-dwvic.__dict__[self.wlot_itm].gss_c_seq_size:],working_dataframe[self.wlot_dimension].tolist()[-dwvic.__dict__[self.wlot_itm].gss_c_seq_size:], 'w', linewidth=0.3, alpha=1.0)
        ax_.plot(working_dataframe.index.tolist()[-dwvic.__dict__[self.wlot_itm].tft_short_c_seq_size:],working_dataframe[self.wlot_dimension].tolist()[-dwvic.__dict__[self.wlot_itm].tft_short_c_seq_size:], 'w', linewidth=0.3, alpha=1.0)
        # --- --- --- --- 
        # # plt.show()
        ax_.set_title(self.wlot_itm,color="white")
        figname=os.path.join(self.out_wlot_folder_itm,"{}.{}.png".format(self.wlot_itm,self.itx_ctx))
        f_.savefig(figname, dpi=hrz_dpi, facecolor='black', edgecolor='black',
            orientation='portrait', format=None, transparent=False, 
            bbox_inches='tight', pad_inches=0.0,metadata=None)
        # --- --- --- --- 
        self.itx_ctx+=1
        return f_, ax_, figname
# --- --- ---
def relife_hrz(
    working_dataframe,
    symbol,
    active_dimension='real_price'):
    hw=HORIZON_WLOTER(wlot_itm=symbol,wlot_dimension=active_dimension)
    _,__,figname=hw.__wlot_graph__(working_dataframe=working_dataframe)
    return figname