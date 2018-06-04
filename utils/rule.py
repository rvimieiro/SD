#
#    UNIVERSIDADE FEDERAL DE PERNAMBUCO -- UFPE (http://www.ufpe.br)
#    CENTRO DE INFORMÁTICA -- CIn (http://www.cin.ufpe.br)
#    Av. Jornalista Anibal Fernandes, s/n - Cidade Universitária (Campus Recife)
#    50.740-560 - Recife - PE - BRAZIL
#
#    Copyright (C) 2018 Renato Vimieiro (rv2@cin.ufpe.br)
#
#    Created on: 2018-05-26
#    @author: Renato Vimieiro
#    @contact: rv2@cin.ufpe.br
#    @license: MIT

from functools import reduce
import numpy as np

class Rule:


    def __init__(self, ft=set(), target=None):
        '''
        Constructor
        '''
        self.__feat = ft
        self.__target = target
        self.__objs = None
    
    def __call__(self, data):
        self.__objs = reduce(lambda x,y: np.logical_and(x,y), map(lambda x: x(data),self.__feat))
        return self.__objs

    def _get_feat(self):
        return self.__feat

    def _get_target(self):
        return self.__target

    def _get_objs(self):
        return self.__objs

    def _set_feat(self, value):
        self.__feat = set(value)

    def _set_target(self, value):
        self.__target = value

    feat = property(_get_feat, _set_feat, None, "The antecedent of this rule.")
    target = property(_get_target, _set_target, None, "The consequent of this rule.")
    objs = property(_get_objs, None, None, "The set of examples that satisfy the rule")    
    
if __name__ == '__main__':
    import pandas as pd
    from textwrap import fill
    from reader import construct_features
    df = pd.read_csv("../test/data/abalone.data",header=None)
    names = ["sex", "length", "diameter", "height", "whole", "shucked", "viscera", "shell", "rings"]
    df.columns = names
    
    ft = list(construct_features(df, 'rings', 10))
    
    r = Rule()
    r.feat = [ft[0],ft[10]]
    r.target = 10
    print(r(df))