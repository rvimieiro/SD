'''
UNIVERSIDADE FEDERAL DE PERNAMBUCO (UFPE)
Centro de Informatica (CIn)
http://www.cin.ufpe.br
Av. Jornalista Anibal Fernandes, s/n - Cidade Universitária (Campus Recife)
50.740-560 - Recife - PE - BRAZIL

Created on 2018-05-09

@author: Renato Vimieiro
@email: rv2 [at] cin [dot] ufpe [dot] br

Copyright (c) 2018 Renato Vimieiro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from utils.feature import Feature
import numpy as np

class __dict(dict):
    def __missing__(self,key):
        return key
        

def construct_features(df, target='last', value=None):
    '''
    This function constructs features according to the rules specified in Gamberger and Lavrac (2002).
    Each feature is an indicator function that establishes constraints on the values
    of individuals in the dataset. Features are constructed in the following way:
    
    * For discrete attributes, features are A_i=v_ip and A_i!=v_in. The values v_ip and v_in como
    from the domain of attribute A_i considering only positive and negative samples respectively.
    
    * For continuous attributes, features are:
        1 - A_i <= (v_ip+v_in)/2, for each pair of consecutive values v_ip and v_in (in this order), 
        conditioned that v_ip comes from the domain of A_i in positive samples and v_in 
        from the domain of A_i in negative samples.
        
        2 - A_i > (v_in+v_ip)/2, for each pair of consecutive values v_in and v_ip (in this order),
        and v_in and v_ip are values from domain of A_i as described above.
    
    * For integer attributes, features are generated considering the rules for discrete and continuous
    attributes.    
    '''
    
    _target = __dict({'last': df.columns.values.tolist()[-1], 
                           'first':df.columns.values.tolist()[0]})
    target = _target[target]
    if value is None: value = df[target].unique()[0]
    
    pos = df[target]==value    
        
    features = set()
        
    features.update([Feature(col,value,'eq') for col in 
                     df[pos].select_dtypes(include=['object','category']).columns.values 
                     for value in df.loc[pos,col].unique()])
    
    features.update([Feature(col,value,'ne') for col in 
                     df[~pos].select_dtypes(include=['object','category']).columns.values 
                     for value in df.loc[~pos,col].unique()])
    
    def chooseOP(e):
        if e: return 'le'
        return 'gt'
    chooseOP = np.vectorize(chooseOP)
    
    for col in df.select_dtypes(include='floating').columns.values:
        tmp = df.sort_values(by=col)
        pos = tmp[target]==value
        indices = np.where(pos != np.roll(pos,1))[0]
        if indices[0]==0: indices = indices[1:]                
        features.update([Feature(col,value,op) for value,op in zip((tmp[col].values[indices] + tmp[col].values[indices+1])/2,
                                                                   chooseOP(pos[indices]))])

        
       
    