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

from numpy import greater, less_equal, equal, not_equal

class Feature:   
    __operators = {'gt':greater,
                   'le':less_equal,
                   'eq': equal,
                   'ne': not_equal}
    
    __symbols = {'gt':'>',
                   'le':'\u2264',
                   'eq': '=',
                   'ne': "\u2260"}

    def __init__(self, column, value, op):

        self.__col = column
        self.__value = value
        self.__func = Feature.__operators[op]
        self.__sym = Feature.__symbols[op]
    
    def __call__(self, data):
        return self.__func(data[self.__col],self.__value)
    
    def __str__(self):
        return '{} {} {}'.format(self.__col,self.__sym,self.__value)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        if not isinstance(other, type(self)): return False
        return self.__col==other.__col and self.__value==other.__value and self.__sym == other.__sym
    
    def __hash__(self):
        return hash((self.__col,self.__value,self.__sym))