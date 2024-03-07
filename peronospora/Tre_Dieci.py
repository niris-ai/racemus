import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math

class Modello_Tre_dieci:
    def __init__(self, counter):
        self.threshold_temperatura = 10
        self.threshold_pioggia = 10
        
        self.temperatura_media_48h_istantanea_registrata = []
        self.temperatura_media_48h_istantanea = 0
        
        self.sommatoria_pioggia_48h_istantanea_registrata = []
        self.sommatoria_pioggia_48h_istantanea = 0
        
        self.temperatura_media_48h_prevista_registrata = []
        self.temperatura_media_48h_prevista = 0
        
        self.sommatoria_pioggia_48h_prevista_registrata = []
        self.sommatoria_pioggia_48h_prevista = 0
        
        self.giorni_passati = 0
        self.avvenuta_infezione = False
        self.avvenuta_infezione_futura = False
        self.counter = counter
    
    def check_infezione(self,tempo, pioggia, temperatura, tempo_futuro, pioggia_prevista, temperatura_prevista):
        # Ogni mezzanotte fai il check
        if(tempo.hour == 0 and tempo.minute == 0 and tempo.second == 0):
            self.giorni_passati = self.giorni_passati + 1
            # Ogni giorno puo avvenire solo una infezione
            self.avvenuta_infezione = False
            self.avvenuta_infezione_futura = False
            
            # Se sono passati due giorni, resetta tutto
            if(self.giorni_passati == 2):
                self.giorni_passati = 0
                self.temperatura_media_48h_istantanea_registrata = []
                self.temperatura_media_48h_istantanea = 0
            
                self.sommatoria_pioggia_48h_istantanea = 0
                self.temperatura_media_48h_prevista_registrata = []
                self.temperatura_media_48h_prevista = 0
                
                self.sommatoria_pioggia_48h_prevista = 0
        
        # Update variabili infezione
        #print(self.temperatura_media_48h_istantanea, self.sommatoria_pioggia_48h_istantanea)
        self.temperatura_media_48h_istantanea_registrata.append(temperatura)
        self.temperatura_media_48h_istantanea = sum(self.temperatura_media_48h_istantanea_registrata)/len(self.temperatura_media_48h_istantanea_registrata)
        
        self.sommatoria_pioggia_48h_istantanea = (self.sommatoria_pioggia_48h_istantanea + pioggia)
        
        self.temperatura_media_48h_prevista_registrata.append(temperatura_prevista)
        self.temperatura_media_48h_prevista = sum(self.temperatura_media_48h_prevista_registrata)/len(self.temperatura_media_48h_prevista_registrata)
        
        self.sommatoria_pioggia_48h_prevista = (self.sommatoria_pioggia_48h_prevista + pioggia_prevista)
        
        if((self.temperatura_media_48h_istantanea >= self.threshold_temperatura) and (self.sommatoria_pioggia_48h_istantanea >= self.threshold_pioggia) and (self.avvenuta_infezione==False)):
            print(str(self.counter) + "  AVVENUTA INFEZIONE IN DATA: " + str(tempo))
            self.temperatura_media_48h_istantanea = 0
            self.temperatura_media_48h_istantanea_registrata = []
            self.sommatoria_pioggia_48h_istantanea = 0
            self.avvenuta_infezione = True
            return True
        if(self.temperatura_media_48h_prevista >= self.threshold_temperatura and self.sommatoria_pioggia_48h_prevista >= self.threshold_pioggia and self.avvenuta_infezione_futura == False):
            print(str(self.counter) + "   "+ str(tempo) + "  POSSIBILE INFEZIONE IN DATA " + str(tempo_futuro))
            self.temperatura_media_48h_prevista = 0
            self.temperatura_media_48h_prevista_registrata = []
            self.sommatoria_pioggia_48h_prevista = 0
            self.avvenuta_infezione_futura = True
        return False
    
    def run_modello(self, ambient_temp, rainfall, time, future_ambient_temp, future_rainfall, future_time):
        return_value = False
        return_value = self.check_infezione(time, rainfall, ambient_temp, future_time, future_rainfall, future_ambient_temp)
        return return_value