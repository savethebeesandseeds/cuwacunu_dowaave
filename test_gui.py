# import matplotlib
# matplotlib.use('TkAgg')
# import numpy as np
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure
# import tkinter as tk
# # --- --- --- --- --- --- --- --- --- --- --- --- 
# PLOT_RENDER_BAO = [
#     {
#         'ID'    :(lambda klrm_state, render : 'waka'),
#         'type'  :(lambda klrm_state, render : 'plot'),
#         'figure':(lambda klrm_state, render : Figure(figsize=(4,2),facecolor="black")),
#         'axis'  :(lambda klrm_state, render : render['figure'].add_subplot(111)),
#         'facecolor':(lambda klrm_state, render : render['axis'].set_facecolor((0,0,0))),
#         'title' :(lambda klrm_state, render : render['axis'].set_title("Estimation Grid", fontsize=8,color=(1,1,1))),
#         'ylabel':(lambda klrm_state, render : render['axis'].set_ylabel("Y",fontsize=8)),
#         'xlabel':(lambda klrm_state, render : render['axis'].set_xlabel("X",fontsize=8)),
#         'plot'  :(lambda klrm_state, render : render['axis'].plot(klrm_state['x_vals'],klrm_state['y_vals'],color='red')),
#         'plot2'  :(lambda klrm_state, render : render['axis'].plot(klrm_state['x_vals'],-klrm_state['y_vals'],color='blue')),
#         'grid'  :(lambda klrm_state, render : render['axis'].grid(which='major',color='white',linestyle='-',linewidth=0.2)),
#         'tick'  :(lambda klrm_state, render : render['axis'].tick_params(colors='red',which='both')),
#         'canvas':(lambda klrm_state, render : FigureCanvasTkAgg(render['figure'],master=render['window'])),
#         'pack'  :(lambda klrm_state, render : render['canvas'].get_tk_widget().pack()),
#         'pos'   :(lambda klrm_state, render : render['canvas'].get_tk_widget().place(x=100, y=50)),
#         'draw'  :(lambda klrm_state, render : render['canvas'].draw()),
#     }
# ]
# # --- --- --- --- --- --- --- --- --- --- --- --- 
# class TK_RENDER:
#     def __init__(self):
#         self.window = tk.Tk()
#         self.window.configure(background='black')
#         self.window.attributes('-fullscreen',True)
#         self._render_graphs_()
#         self.window.mainloop()
#     def _render_graphs_(self):
#         klrm_state={}
#         klrm_state['x_vals']=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#         klrm_state['y_vals']=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#         for _bao in PLOT_RENDER_BAO:
#             c_render = {'window':self.window}
#             for __bao_key in list(_bao.keys()):
#                 c_render[__bao_key] = _bao[__bao_key](klrm_state,c_render)
# TK_RENDER()

# # from tkinter import *
# # import subprocess as sub
# # p = sub.Popen('./cwcn_kalamar_host.py',stdout=sub.PIPE,stderr=sub.PIPE)
# # output, errors = p.communicate()

# # root = Tk()
# # text = Text(root)
# # text.pack()
# # text.insert(END, output)
# # root.mainloop()






# import tkinter as tk

# class KeyTracker:
#     def __init__(self, on_key_press, on_key_release):
#         self.on_key_press = on_key_press
#         self.on_key_release = on_key_release
#         self._key_pressed = False

#     def report_key_press(self, event):
#         if not self._key_pressed:
#             self.on_key_press()
#         self._key_pressed = True

#     def report_key_release(self, event):
#         if self._key_pressed:
#             self.on_key_release()
#         self._key_pressed = False


# def start_recording(event=None):
#     print('Recording right now!')


# def stop_recording(event=None):
#     print('Stop recording right now!')
import io
from PIL import Image

def RCsi_CRYPT(key, data): # tehdujco, !
    S = list(range(256))
    j = 0
    for i in list(range(256)):
        j = (j + S[i] + ord(key[i % len(key)])) % 256
        S[i], S[j] = S[j], S[i]
    j = 0
    y = 0
    out = []
    for char in data:
        j = (j + 1) % 256
        y = (y + S[j]) % 256
        S[j], S[y] = S[y], S[j]
        out.append(chr(ord(char) ^ S[(S[j] + S[y]) % 256]))
    return ''.join(out)
    
def test_file_encode_image(c_path):
    with open(c_path,'rb') as _F:
        c_loaded = _F.read()
    # --- --- --- m1
    # print((repr(c_loaded)))
    # c_encoded=str(RCsi_CRYPT("waka","{}".format(c_loaded.decode('iso-8859-1'))))
    # # print((repr(c_encoded)))
    # c_decoded=str(RCsi_CRYPT("waka","{}".format(c_encoded))).encode('iso-8859-1')
    # # print((repr(c_decoded)))
    # c_stream = io.BytesIO(c_decoded)
    # img=Image.open(c_stream)
    # img.show()
    





    # --- --- --- m2
    c_key=r"waka"
    # print((repr(c_loaded)))
    c_encoded = ','.join([str(ord(__)) for __ in c_loaded.decode('iso-8859-1')])
    c_encoded=str(RCsi_CRYPT(c_key,"{}".format(c_encoded)))
    print((repr(c_encoded)))
    c_decoded=str(RCsi_CRYPT(c_key,"{}".format(c_encoded)))
    c_decoded=[chr(int(__)) for __ in c_decoded.split(',')]
    # print([chr(int(__)) for __ in c_decoded])
    # print(c_decoded)
    c_decoded=''.join(c_decoded).encode('iso-8859-1')
    # print((repr(c_decoded)))
    c_stream = io.BytesIO(c_decoded)
    
    # print((repr(c_stream)))
    img=Image.open(c_stream)
    img.show()
    # print((repr(c_stream)))

    
    
    # --- --- --- m3 fix
    # c_stream = io.BytesIO(c_loaded)
    # # print((repr(c_stream)))
    # import numpy as np
    # import ast
    # img=np.asarray(Image.open(c_stream))
    # c_encoded=str(RCsi_CRYPT("waka","{}".format(img.tolist())))
    # c_decoded=str(RCsi_CRYPT("waka","{}".format(c_encoded)))
    # c_decoded=ast.literal_eval(c_decoded)
    # c_decoded=np.array(c_decoded)
    # print(c_decoded)
    # img = Image.fromarray(c_decoded.astype('uint8'),'RGB')
    # img.show()

class lvl_1:
    a=1
    class lvl_2:
        aa=2
if __name__ == '__main__':
    # master = tk.Tk()

    # key_tracker = KeyTracker(start_recording, stop_recording)
    # master.bind("<KeyPress-Return>", key_tracker.report_key_press)
    # master.bind("<KeyRelease-Return>", start_recording)
    # master.mainloop()

    # test_file_encode_image('./tft_dumps/BTCUSDTPERP/BTCUSDTPERP-baed0311-1289-4c24-a76d-66de7e18a452.png')
    # test_file_encode_image('./tft_dumps/BTCUSDTPERP/BTCUSDTPERP.png')
    # test_file_encode_image('./gauss_dumps/BTCUSDTPERP/BTCUSDTPERP.0.png')
    # test_file_encode_image('./text.txt')

    c_loaded=''
    print((repr(c_loaded)))
    c_encoded=str(RCsi_CRYPT("niwaave","{}".format('niwaave')))
    print((repr(c_encoded)))
    c_decoded=str(RCsi_CRYPT("niwaave","{}".format(c_encoded)))
    print((repr(c_decoded)))

    # print(lvl_1.a)
    # print(lvl_1.__dict__['lvl_2'].aa)

    # import cwcn_duuruva_piaabo

    # measure_duuruva=cwcn_duuruva_piaabo.DUURUVA(_duuruva_vector_size=1,_wrapper_duuruva_normalize='norm', _d_name='measure_duuruva')

    # import math
    # import random
    # len_seq = 1000
    # x_measure = [_/len_seq for _ in range(len_seq)]
    # def y_fun(x):
    #     A = 1
    #     bias = 1
    #     freq = 25
    #     return A*math.sin(2*math.pi*freq*x) + bias*x
    # # def y_fun(x):
    # #     return 
    # y_measure = [y_fun(_) for _ in x_measure]

    # d_measure = [measure_duuruva._duuruva_value_wrapper_(_) for _ in y_measure]

    # import matplotlib.pyplot as plt

    # plt.plot(x_measure,y_measure)
    # plt.plot(x_measure,d_measure)
    # plt.show()

