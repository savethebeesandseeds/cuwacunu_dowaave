# --- --- ---
# cwcn_duuruva_piaabo.py
# --- --- ---
# a mayor TEHDUJCO to python fundation
# --- --- ---
# a mayor TEHDUJCO to the torch fundation
# --- --- ---
import math
# --- --- ---
import cwcn_dwve_client_config as dwvc
# --- --- ---
class DUURUVA:
    def __init__(self,_duuruva_vector_size : int,_wrapper_duuruva_normalize, _d_name : str=None):
        self._d_name=_d_name
        self._wrapper_duuruva_std_or_norm=_wrapper_duuruva_normalize
        self._duuruva_vector_size=_duuruva_vector_size
        self._reset_duuruva_()
    def _reset_duuruva_(self):
        self._d_count=0
        self._duuruva=[]
        for _ in range(self._duuruva_vector_size):
            aux_d={}
            aux_d['value'] = 0
            aux_d['diff_1'] = 0
            aux_d['diff_2'] = 0
            aux_d['max'] = 0
            aux_d['min'] = 0
            aux_d['variance'] = 0
            aux_d['mean'] = 0
            aux_d['M2'] = 0
            aux_d['M3'] = 0
            aux_d['M4'] = 0
            aux_d['kurtosis'] = 0
            aux_d['skewness'] = 0
            self._duuruva.append(aux_d)
    def _is_duuruva_ready_(self):
        return dwvc.CWCN_DUURUVA_CONFIG.DUURUVA_READY_COUNT<=self._d_count
    def _duuruva_inverse_value_wrapper_(self,c_vect):
        for _v_idx in range(self._duuruva_vector_size):
            if(self._duuruva_vector_size==1):
                c_value = c_vect
            else:
                c_value = c_vect[_v_idx]
            try:
                if(self._wrapper_duuruva_std_or_norm == 'norm'):
                    c_standar = (c_value)*(math.sqrt(self._duuruva[_v_idx]['variance']) + dwvc.CWCN_DUURUVA_CONFIG.MIN_STD) + self._duuruva[_v_idx]['mean']
                elif(self._wrapper_duuruva_std_or_norm == 'std'):
                    c_standar = (c_value - self._duuruva[_v_idx]['mean'])*(math.sqrt(self._duuruva[_v_idx]['variance']) + dwvc.CWCN_DUURUVA_CONFIG.MIN_STD) + self._duuruva[_v_idx]['mean']
                elif(self._wrapper_duuruva_std_or_norm == 'mean'):
                    c_standar = (c_value + self._duuruva[_v_idx]['mean'])
                elif(self._wrapper_duuruva_std_or_norm == 'not'):
                    c_standar = c_value
                else:
                    assert(False), "wrong wrapper_duuruva_std_or_norm configuration"
            except Exception as e:
                if(self._wrapper_duuruva_std_or_norm == 'not'):
                    c_standar=c_value
                else:
                    c_standar=0
                if(self._is_duuruva_ready_()):
                    raise Exception("Error processing duuruva : {}".format(e))
            # --- --- --- --- --- 
            if(self._is_duuruva_ready_() or self._wrapper_duuruva_std_or_norm == 'not'):
                if(self._duuruva_vector_size==1):
                    c_vect = c_standar
                else:
                    c_vect[_v_idx] = c_standar
            else:
                if(self._duuruva_vector_size==1):
                    c_vect = 0
                else:
                    c_vect[_v_idx] = 0
    def _duuruva_value_wrapper_(self,c_vect):
        self._d_count+=1
        _n = min(self._d_count,dwvc.CWCN_DUURUVA_CONFIG.DUURUVA_MAX_COUNT)
        for _v_idx in range(self._duuruva_vector_size):
            if(self._duuruva_vector_size==1):
                c_value = c_vect
            else:
                c_value = c_vect[_v_idx]
            # --- --- --- --- --- --- --- --- --- --- a mayor TEHDUJCO to the WIKI
            # --- --- --- --- https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            self._duuruva[_v_idx]['value']=c_value
            self._duuruva[_v_idx]['max']=max(self._duuruva[_v_idx]['max'], self._duuruva[_v_idx]['value'])
            self._duuruva[_v_idx]['min']=min(self._duuruva[_v_idx]['min'], self._duuruva[_v_idx]['value'])
            _delta = self._duuruva[_v_idx]['value'] - self._duuruva[_v_idx]['mean']
            _delta_n = _delta/_n
            _delta_n2 = _delta_n*_delta_n
            _term1 = _delta*_delta_n*(_n-1)
            self._duuruva[_v_idx]['mean'] += _delta_n
            self._duuruva[_v_idx]['M4'] += _term1*_delta_n2*(_n*_n-3*_n+3)+6*_delta_n2*self._duuruva[_v_idx]['M2']-4*_delta_n*self._duuruva[_v_idx]['M3']
            self._duuruva[_v_idx]['M3'] += _term1*_delta_n*(_n-2)-3*_delta_n*self._duuruva[_v_idx]['M2']
            self._duuruva[_v_idx]['M2'] += _term1
            try:
                self._duuruva[_v_idx]['variance'] = self._duuruva[_v_idx]['M2']/(_n-1)
                self._duuruva[_v_idx]['kurtosis'] = (_n*self._duuruva[_v_idx]['M4'])/(self._duuruva[_v_idx]['M2']*self._duuruva[_v_idx]['M2'])-3
                self._duuruva[_v_idx]['skewness'] = math.sqrt(_n)*self._duuruva[_v_idx]['M3']/(math.pow(self._duuruva[_v_idx]['M2'],3)*math.sqrt(self._duuruva[_v_idx]['M2'])) #FIXME check if is right
                if(self._wrapper_duuruva_std_or_norm == 'norm'):
                    c_standar = (c_value - self._duuruva[_v_idx]['mean'])/(math.sqrt(self._duuruva[_v_idx]['variance']) + dwvc.CWCN_DUURUVA_CONFIG.MIN_STD)
                elif(self._wrapper_duuruva_std_or_norm == 'std'):
                    c_standar = (c_value - self._duuruva[_v_idx]['mean'])/(math.sqrt(self._duuruva[_v_idx]['variance']) + dwvc.CWCN_DUURUVA_CONFIG.MIN_STD) + self._duuruva[_v_idx]['mean']
                elif(self._wrapper_duuruva_std_or_norm == 'mean'):
                    c_standar = (c_value - self._duuruva[_v_idx]['mean'])
                elif(self._wrapper_duuruva_std_or_norm == 'not'):
                    c_standar = c_value
                else:
                    assert(False), "wrong wrapper_duuruva_std_or_norm configuration"
            except Exception as e:
                if(self._wrapper_duuruva_std_or_norm == 'not'):
                    c_standar=c_value
                else:
                    c_standar=0
                if(self._is_duuruva_ready_()):
                    raise Exception("Error processing duuruva : {}".format(e))
            # --- --- --- --- --- 
            if(self._is_duuruva_ready_() or self._wrapper_duuruva_std_or_norm == 'not'):
                if(self._duuruva_vector_size==1):
                    c_vect = c_standar
                else:
                    c_vect[_v_idx] = c_standar
            else:
                if(self._duuruva_vector_size==1):
                    c_vect = 0
                else:
                    c_vect[_v_idx] = 0
        return c_vect

