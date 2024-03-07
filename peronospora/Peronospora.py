import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math

class ModelloPeronospora:
	# Una volta che il th supera il th non si può sapere la gravità dell'infezione 
	def __init__(self, counter):
		# Variabile che conta il numero del cluster
		self.counter = counter
		
		# VARIABILI CHECK MATURAZIONE OOSPORE
		self.threshold_maturazione = 140 
		self.sommatoria_maturazione = 0
		self.avvenuta_maturazione_oospore = False
		
		# VARIABILI CHECK GERMINAZIONE
		self.avvenuta_germinazione = False
		self.sommatoria_germinazione_pioggia_48h = 0
		self.giorno_passato = 0
		
		# VARIABILI CHECK DISPERSIONE
		self.ore_necessarie = 6
		self.avvenuta_dispersione = False
		self.count_down = 24
		self.valori_orari = []
		self.valori_orari_passati = []
		
		# VARIABILI CHECK INFEZIONE PRIMARIA
		self.counter_bagnatura_fogliare = 0
		self.record_temperature = []
		self.temperatura_media = 0
		self.avvenuta_infezione_primaria = False
		self.foglia_non_bagnata = []
		self.threshold_bagnatura = 20
		
		# VARIABILI CHECK INCUBAZIONE
		self.avvenuta_incubazione = False
		self.valori_orari_temperatura_mattina = []
		self.valori_orari_temperatura_sera = []
		self.sommatoria_percentuali = 0
		self.threshold_umidità = 60
		self.valori_percentuale_incubazione = {
			14 : (6.6,9.0),
			15 : (7.6,10.5),
			16 : (8.6,11.7),
			17: (10.0, 13.3),
			18 : (11.1, 15.3),
			19: (12.5, 16.6),
			20 : (14.2, 20.0),
			21 : (15.3, 22.2),
			22 : (16.6, 22.2),
			23 : (18.1, 25.0),
			24: (18.1, 25.0),
			25 : (16.6, 22.2),
			26 : (16.6, 22.2)
		}
		
		# VARIABILI CHECK SPORULAZIONE
		self.avvenuta_sporulazione = False
		self.record_t_media = []
		self.temperatura_media_sporulazione = 0
		self.check_orario_bagnatura_fogliare = {
			22 : [],
			23 : [],
			0 : [],
			1 : [],
			2 : [],
			3 : [],
			4 : []
		}
		self.check_orario_umidità = {
			22 : [],
			23 : [],
			0 : [],
			1 : [],
			2 : [],
			3 : [],
			4 : []
		}
		
		# VARIABILI CHECK INFEZIONE SECONDARIA
		self.avvenuta_infezione_secondaria = False
		self.threshold_bagnatura_temperatura = 50
		self.prodotto_bagnatura_temperatura = 1
		self.ore_bagnatura = 1
		
		# Statistiche
		self.lista_date_eventi = []
		
	def check_maturazione_oospore(self,temperatura, tempo, versione_verbose):
		# PAR -->  ambient_temp
		# Calcolo la sommatoria delle temperature registrate durante il giorno maggiori di 8 gradi
		# Se questa sommatoria supera il threshold_maturazione allora ritorna True
		# Ogni nuovo giorno fai il reset delle variabili (sommatoria)
		if(tempo.hour == 0 and tempo.minute == 0 and tempo.second == 0):
			#print("Tmp maturazione  " + str(self.sommatoria_maturazione) + "  " + str(tempo))
			self.sommatoria_maturazione = 0
		if(tempo.minute == 0):
			#print("wela" + str(self.sommatoria_maturazione))
			if(temperatura > 8):
				self.sommatoria_maturazione = self.sommatoria_maturazione + temperatura
				# Condizioni ottimali per la maturazione raggiunte
				if self.sommatoria_maturazione >= self.threshold_maturazione:
					self.avvenuta_maturazione_oospore = True
					if versione_verbose !=0:
						print(str(self.counter) + " MATURAZIONE OOSPORE RAGGIUNTO IN DATA:  " + str(tempo))
					self.lista_date_eventi.append(tempo)
					return 1
		return 0

	def check_germinazione_oospore(self, umidità, pioggia, temperatura, tempo, versione_verbose):
		# PAR -->  ambient_temp, humidity, rainfall
		
		# Algortimo 2: 5mm di pioggia in 48h e t > 8gradi 
		
		# Check ogni giorno per tenere traccia dei giorni passati 
		if(tempo.hour == 0 and tempo.minute == 0 and tempo.second == 0):
			self.giorno_passato = self.giorno_passato + 1
			if(self.giorno_passato == 2):
				# Fai il reset ogni 48h
				self.giorno_passato = 0
				self.sommatoria_germinazione_pioggia_48h = 0
		# Se la temperatura supera gli 8 gradi
		if (temperatura > 8):
			# Aggiorna la sommatoria dei millimetri di pioggia
			self.sommatoria_germinazione_pioggia_48h = self.sommatoria_germinazione_pioggia_48h + pioggia
			# Se superiamo il valore di 5mm a temperatura > 8 allora ritorna true
			if(self.sommatoria_germinazione_pioggia_48h >= 5):
				if versione_verbose !=0:
					print(str(self.counter) + " GERMINAZIONE OOSPORE RAGGIUNTO IN DATA:  " + str(tempo))
				self.lista_date_eventi.append(tempo)
				self.avvenuta_germinazione = True
				self.sommatoria_germinazione_pioggia_48h = 0
				self.giorno_passato = 0
				return 2
		return 0 		

	def check_dispersione(self, pioggia, tempo, versione_verbose):
		# Durante le 6 ore di germinazione se abbiamo pioggia >3 mm/ora allora ritorna true
		# Altrimenti torna alla germinazione
		
		
		# Fai il reset ogni ora della lista dei valori di piovosità visti nell'arco dell'ora
		# Check ogni ora dei valori
		
		# Appendi i valori di piovosità
		self.valori_orari.append(pioggia)
		# Se siamo ancora nel periodo delle 6 ore
		if(self.ore_necessarie > 0):
			# Se ho registrato tutti i valori di quell'ora
			if(len(self.valori_orari) == 4):
				# Fai la media dei valori e se superiamo i 3mm ritorna true
				if(max(self.valori_orari) >= 3):
					if versione_verbose !=0:
						print(str(self.counter) + " AVVENUTA DISPERSIONE IN DATA: " + str(tempo) + " per i seguenti valori di piovosità: " + str(self.valori_orari))
					self.lista_date_eventi.append((tempo))
					self.avvenuta_dispersione = True
					# Reset dei valori visti que'ora
					self.valori_orari = []
					return 3
				else:
					# Nel caso in cui dobbiamo ancora vedere n ore
					self.ore_necessarie = self.ore_necessarie - 1
				# Reset dei valori visti que'ora
				self.valori_orari_passati.append(self.valori_orari)
				self.valori_orari = []
		elif(self.ore_necessarie == 0):
			# se dopo le sei ore non abbiamo ancora avuto > 3mm di pioggia torna al punto precedente
			if versione_verbose !=0:
				print(str(self.counter) + " DISPERSIONE FALLITA IN DATA: " + str(tempo) + " per i seguenti valori di piovosità: " + str(self.valori_orari_passati))
			# Reset variabili
			self.avvenuta_germinazione = False
			self.avvenuta_dispersione = False
			self.ore_necessarie = 6
			self.valori_orari = []
			self.valori_orari_passati=[]
		return 0

	def check_infezione_primaria(self,temperatura, pioggia, leaf_wet, tempo, versione_verbose):
		if temperatura > 10:
			self.record_temperature.append(temperatura)
			#self.temperatura_media = (self.temperatura_media + temperatura)/2
			self.temperatura_media = sum(self.record_temperature)/len(self.record_temperature)
			
			#! possibile sostituzione formula per calcolare la media (--> più veloce )
			
			#print(leaf_wet, self.threshold_bagnatura)
			if pioggia >= 3:
				if versione_verbose !=0:
					print(str(self.counter) +  " INFEZIONE PRIMARIA IN DATA: "+ str(tempo))
				self.lista_date_eventi.append(tempo)
				self.avvenuta_infezione_primaria = True
				return 4
			if leaf_wet > 0:
				self.foglia_non_bagnata = []
				self.counter_bagnatura_fogliare = self.counter_bagnatura_fogliare + 1
				if self.temperatura_media * ((self.counter_bagnatura_fogliare*15)/60) >= 50:
					if versione_verbose !=0:
						print(str(self.counter) +  " INFEZIONE PRIMARIA IN DATA: "+ str(tempo))
					self.lista_date_eventi.append(tempo)
					self.avvenuta_infezione_primaria = True
					return 4
			else:
				self.foglia_non_bagnata.append(leaf_wet)
				if(len(self.foglia_non_bagnata) == 4):
					if versione_verbose !=0:
						print(str(self.counter) + " No infezione")
					self.avvenuta_dispersione = False
					self.foglia_non_bagnata = []
					self.counter_bagnatura_fogliare = 0
					self.temperatura_media = 0
					self.record_temperature = []
		return 0

	def check_incubazione(self, temperatura, humidity, tempo, versione_verbose):  
		
		# Ogni mezzanotte calcolo i valori registrati durante il giorno
		if((len(self.valori_orari_temperatura_mattina) != 0 and (len(self.valori_orari_temperatura_sera) != 0)) and (tempo.hour == 0 and tempo.minute == 0 and tempo.second == 0)):
			# Calcolo la media aritmetica giornaliera in base al max e min delle temperature registrate alle 9 e 21
			media_temperatura_registrata = math.floor((max(self.valori_orari_temperatura_mattina) + max(self.valori_orari_temperatura_sera) + (min(self.valori_orari_temperatura_mattina) + min(self.valori_orari_temperatura_sera)))/4)
			# Se la media calcolata risulta come un valore della tabella allora aumenta la sommatoria della percentuale
			if(media_temperatura_registrata in self.valori_percentuale_incubazione):
				# Se l'umidità è alta (supera il threshold) allora prendi il valore corrispondente della tabella
				if humidity >= self.threshold_umidità:
					self.sommatoria_percentuali = self.sommatoria_percentuali + self.valori_percentuale_incubazione[media_temperatura_registrata][1]
				else:
					self.sommatoria_percentuali = self.sommatoria_percentuali + self.valori_percentuale_incubazione[media_temperatura_registrata][0]
				# Se abbiamo raggiunto il valore max è l'ora di intervenire sul campo
				if(self.sommatoria_percentuali >= 100):
					if versione_verbose!=0:
						print(str(self.counter) + " INTERVENIRE INCUBAZIONE IN DATA: " + str(tempo))
					self.lista_date_eventi.append(tempo)
					self.avvenuta_incubazione = True
					self.sommatoria_percentuali = 0
					self.valori_orari_temperatura = []
					return 5
				# Avvisare di prepararsi se si raggiunge la soglia >80%
				if(self.sommatoria_percentuali >= 80):
					if versione_verbose!=0:
						print(str(self.counter) + " INIZIARE AD INTERVENIRE SOGLIA 80% RAGGIUNTA IN DATA: "  + str(tempo))
					self.lista_date_eventi.append(tempo)
					
			# Ogni mezzanotta reset dei valori per il giorno seguente
			self.valori_orari_temperatura_mattina = []
			self.valori_orari_temperatura_sera = []
		else:
			# Registrare i valori durante la giornata
			# controlla l'ora, bisogna ricordare il max e min alle 9 e 21 della giornata
			if(tempo.hour == 9):
				self.valori_orari_temperatura_mattina.append(temperatura)
			elif(tempo.hour == 21):
				self.valori_orari_temperatura_sera.append(temperatura)
		return 0
			
	def check_sporulazione(self, leaf_wet, humidity, temperatura, tempo, versione_verbose):
		# Avviene durante il buio dopo l'incubazione fra 22 e 4 mattino
		if(tempo.hour >= 22 or tempo.hour < 4):
			self.record_t_media.append(temperatura)
			#self.temperatura_media_sporulazione = (self.temperatura_media_sporulazione + temperatura)/2
			self.temperatura_media_sporulazione = sum(self.record_t_media)/len(self.record_t_media)
			if(self.temperatura_media_sporulazione >= 11):            
				# se la foglia è bagnata
				if(leaf_wet > self.threshold_bagnatura):
					self.check_orario_bagnatura_fogliare[tempo.hour].append(pioggia)
				# Se c'è tanta umidità
				if(humidity >= self.threshold_umidità):
					self.check_orario_umidità[tempo.hour].append(humidity)
		if(tempo.hour == 4 and tempo.minute == 15):
			# check se ci sono 4 ore consecutive di umidità alta
			ore_consecutive_umidità = 0
			for key in self.check_orario_umidità:
				if(len(self.check_orario_umidità[key]) == 4):
					ore_consecutive_umidità = ore_consecutive_umidità + 1
				else:
					if(ore_consecutive_umidità >= 4):
						break
					else:
						ore_consecutive_umidità = 0
						
			# check se ci sono 4 ore consecutive di bagnatura fogliare
			ore_consecutive_bagnatura_fogliare = 0
			for key in self.check_orario_bagnatura_fogliare:
				if(len(self.check_orario_bagnatura_fogliare[key]) == 4):
					ore_consecutive_bagnatura_fogliare = ore_consecutive_bagnatura_fogliare + 1
				else:
					if(ore_consecutive_bagnatura_fogliare >= 4):
						break
					else:
						ore_consecutive_bagnatura_fogliare = 0
			if(ore_consecutive_umidità == 4 or ore_consecutive_bagnatura_fogliare == 4):
				if versione_verbose !=0:
					print(str(self.counter) + " AVVENUTA SPORULAZIONE IN DATA: " + str(tempo))
				self.lista_date_eventi.append(tempo)
				self.avvenuta_sporulazione = True
				return 6
			else:    
				#reset variabili appena dopo
				self.check_orario_bagnatura_fogliare = {
					22 : [],
					23 : [],
					0 : [],
					1 : [],
					2 : [],
					3 : [],
					4 : []
				}
				self.check_orario_umidità = {
					22 : [],
					23 : [],
					0 : [],
					1 : [],
					2 : [],
					3 : [],
					4 : []
				}
				self.temperatura_media_sporulazione = 0
				self.record_t_media = []
		return 0
    
    
	def check_infezione_secondaria(self, humidity, leaf_wet, temperatura, tempo, pioggia, versione_verbose):
		# Prima condizione necessaria
		if(temperatura > 11):
			# Da vedere se il th è giusto o meno
			if(leaf_wet > 0):
				self.ore_bagnatura = self.ore_bagnatura + 1
			#print(temperatura * (self.ore_bagnatura *15 / 60))
			if(temperatura * (self.ore_bagnatura * 15 / 60) >= self.threshold_bagnatura_temperatura):
				self.avvenuta_infezione_secondaria = True
				if versione_verbose !=0:
					print(str(self.counter) + " AVVENUTA INFEZIONE SECONDARIA IN DATA: " + str(tempo))
				self.lista_date_eventi.append(tempo)
				self.ore_bagnatura = 0
				return 7
		return 0
	
	def run_modello(self, ambient_temp, humidity, rainfall, leaf_wet_high, time, versione_verbose):
		return_value = -1
		if(self.avvenuta_maturazione_oospore == False):
			return_value = self.check_maturazione_oospore(ambient_temp, time, versione_verbose)
		elif(self.avvenuta_maturazione_oospore == True and self.avvenuta_germinazione==False):
			return_value = self.check_germinazione_oospore(humidity, rainfall, ambient_temp, time, versione_verbose)
		elif(self.avvenuta_germinazione == True and self.avvenuta_dispersione == False):
			return_value = self.check_dispersione(rainfall, time, versione_verbose)
		elif(self.avvenuta_dispersione and  self.avvenuta_infezione_primaria == False):
			return_value = self.check_infezione_primaria(ambient_temp, rainfall, leaf_wet_high, time, versione_verbose)
		elif(self.avvenuta_infezione_primaria and self.avvenuta_incubazione == False):
			return_value = self.check_incubazione(ambient_temp, humidity, time, versione_verbose)
		elif(self.avvenuta_incubazione and self.avvenuta_sporulazione == False):
			return_value = self.check_sporulazione(leaf_wet_high, humidity, ambient_temp, time, versione_verbose)
		elif(self.avvenuta_sporulazione and self.avvenuta_infezione_secondaria == False):
			return_value = self.check_infezione_secondaria(humidity, leaf_wet_high, ambient_temp, time, rainfall, versione_verbose)
		return 0 if return_value is None else return_value