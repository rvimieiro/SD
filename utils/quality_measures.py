#
#    UNIVERSIDADE FEDERAL DE PERNAMBUCO -- UFPE (http://www.ufpe.br)
#    CENTRO DE INFORMÁTICA -- CIn (http://www.cin.ufpe.br)
#    Av. Jornalista Anibal Fernandes, s/n - Cidade Universitária (Campus Recife)
#    50.740-560 - Recife - PE - BRAZIL
#
#    Copyright (C) 2018 Renato Vimieiro (rv2@cin.ufpe.br)
#
#    Created on: 2018-06-13
#    @author: Renato Vimieiro
#    @contact: rv2@cin.ufpe.br
#    @license: MIT

import numpy as np

class __dict(dict):
    def __missing__(self,key):
        return key

def _map_target_column(columns, target):
    mapping = __dict({'last': columns[-1], 
                           'first':columns[0]})
    return mapping[target]

def wracc(df, rule, target='last'):
    '''
    This function computes the weighted relative accuracy of a rule
    in a data frame, considering the target column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The data where the rule shall be evaluated.
    
    target : string 
            The column of df that contains the target (class) attribute, or either
            'last' for the last column (default) or 'first' for the first.
    
    
    rule : Rule-object 
            The measure is computed taking rule.target as positive
            and the rest as negative examples.
            
    Returns
    -------
    score : float
            The non-normalized weighted relative accuracy of the rule.
            Values vary from -0.25 to 0.25. The larger the value, the more
            significant the rule is, zero means uninteresting. 
    '''
    
    target = _map_target_column(df.columns.values.tolist(), target)
    examples = rule(df)
    positive = df[target] == rule.target
    N = df.shape[0]
    probClass = np.sum(positive)/N
    probCond = np.sum(examples)/N
    accuracy = np.sum(positive & examples)/N
    return probCond * (accuracy - probClass)
    
    
      
    
    