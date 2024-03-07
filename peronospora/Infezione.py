from Peronospora import ModelloPeronospora
from Tre_Dieci import Modello_Tre_dieci

import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math

class Infezione():
    def __init__(self, name_test):
        self.name_result_file = name_test
        # Modello Peronospora
        self.modello_prenospora_new = ModelloPeronospora(1)
        self.infezioni_modello_peronospora_new = [self.modello_prenospora_new]
        self.counter_peronospora = 2
        
        # Modello 3 dieci
        self.modello_tre_dieci = Modello_Tre_dieci(1)
        self.infezioni_modello_tre_dieci = [self.modello_tre_dieci]
        self.counter_tre_dieci = 2
        
        # Record varibles
        self.record_temperatura = []
        self.record_umidità = []
        self.record_pioggia = []
        self.record_bagnatura_fogliare = []
        self.record_tempo = []
        self.states = []

        self.cleaned_states = []
        
    def new_infezione_modello_peronospora_new(self):
        nuova_infezione = ModelloPeronospora(self.counter_peronospora)
        self.infezioni_modello_peronospora_new.append(nuova_infezione)
        self.counter_peronospora = self.counter_peronospora + 1
    
    def new_infezione_modello_tre_dieci(self):
        nuova_infezione = Modello_Tre_dieci(self.counter_tre_dieci)
        self.infezioni_modello_tre_dieci.append(nuova_infezione)
        self.counter_tre_dieci = self.counter_tre_dieci + 1
    
    def record_variables(self, temperatura, umidità, pioggia, bagnatura_fogliare, tempo):
        self.record_temperatura.append(temperatura)
        self.record_umidità.append(umidità)
        self.record_pioggia.append(pioggia)
        self.record_bagnatura_fogliare.append(bagnatura_fogliare)
        self.record_tempo.append(tempo)
    
    #update dello state
    def record_state(self,state):
        self.states.append(state)

    def analyze_results(self):
        final_result = []
        for infezioni in self.states:
            tmp = []
            for state in infezioni[0]: 
                # filtro per stati diversi
                if state != 0 and state != -1:
                    tmp.append(state)
            if len(tmp) > 0:
                final_result.append([tmp, infezioni[1]])
        self.cleaned_states = final_result        
