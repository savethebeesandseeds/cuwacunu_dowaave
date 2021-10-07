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






import tkinter as tk

class KeyTracker:
    def __init__(self, on_key_press, on_key_release):
        self.on_key_press = on_key_press
        self.on_key_release = on_key_release
        self._key_pressed = False

    def report_key_press(self, event):
        if not self._key_pressed:
            self.on_key_press()
        self._key_pressed = True

    def report_key_release(self, event):
        if self._key_pressed:
            self.on_key_release()
        self._key_pressed = False


def start_recording(event=None):
    print('Recording right now!')


def stop_recording(event=None):
    print('Stop recording right now!')


if __name__ == '__main__':
    master = tk.Tk()

    key_tracker = KeyTracker(start_recording, stop_recording)
    master.bind("<KeyPress-Return>", key_tracker.report_key_press)
    master.bind("<KeyRelease-Return>", start_recording)
    master.mainloop()