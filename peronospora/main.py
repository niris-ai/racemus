import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import math
from Infezione import Infezione
from save_results import check_folder_exists, save_in_folder


# Converto tutto da stringa a datetime per poterle comparare
def all_to_datetime_and_from_1GEN(file):
	for i in range(len(file["time"])):
		file["time"][i] =  datetime.strptime(file["time"][i][:-6], '%Y-%m-%d %H:%M:%S')
	#! IMPORTANTE
	# Si considera tutto dal primo gennaio in poi
	file = file[file["time"] >= datetime.strptime("2022-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')]
	file = file.reset_index(drop=True)
	return file


#import dei dati storici

humidity = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_humidity_19_07_2022.csv"))
pressure = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_pressure_19_07_2022.csv"))
rainfall = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_rainfall_19_07_2022.csv"))
soil_temp = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_soil_temp_19_07_2022.csv"))
water_content = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_water_content_19_07_2022.csv"))
leaf_wet_low = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_leaf_wet_low_19_07_2022.csv"))
leaf_wet_high = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_leaf_wet_high_19_07_2022.csv"))
leaf_temp_low = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_leaf_temp_low_19_07_2022.csv"))
leaf_temp_high = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_leaf_temp_high_19_07_2022.csv"))
ambient_temp = all_to_datetime_and_from_1GEN(pd.read_csv("../../dati_pulizia/done/15MIN_RESULT_ambient_temp_19_07_2022.csv"))
print("----------->   Caricato i dati storici")


# Se vogliamo vedere i print degli eventi mentre succedono
versione_verbose = int(sys.argv[1])

solo_una_infezione = int(sys.argv[2])


check_folder_exists("esperimenti")


Check_infezioni = Infezione("test_1")
for i in range(len(humidity["time"])):
	stati_di_ritorno = []
	first_check = True
	for infezioni in Check_infezioni.infezioni_modello_peronospora_new:
		try:
			leaf_wet = float(leaf_wet_high[leaf_wet_high["time"] == ambient_temp["time"][i]]["value"])
			stati_di_ritorno.append(infezioni.run_modello(ambient_temp["value"][i], humidity["value"][i], rainfall["value"][i], leaf_wet, ambient_temp["time"][i], versione_verbose))
			if(first_check):
				Check_infezioni.record_variables(ambient_temp["value"][i], humidity["value"][i], rainfall["value"][i], leaf_wet, ambient_temp["time"][i])
				first_check = False
		except:
			stati_di_ritorno.append(infezioni.run_modello(ambient_temp["value"][i], humidity["value"][i], rainfall["value"][i], 0, ambient_temp["time"][i], versione_verbose))
			if(first_check):
				Check_infezioni.record_variables(ambient_temp["value"][i], humidity["value"][i], rainfall["value"][i], 0, ambient_temp["time"][i])
				first_check = False
	nuova_germinazione = False
	if solo_una_infezione > 0:
		for stato in stati_di_ritorno:
			if stato == 1:
				nuova_germinazione = True
				break
	if nuova_germinazione:
		Check_infezioni.new_infezione_modello_peronospora_new()
	
	Check_infezioni.record_state((stati_di_ritorno,humidity["time"][i] ))

#print(Check_infezioni.states)

print("Analizzo i risultati...")
Check_infezioni.analyze_results()
print("fine analisi")
print("Inizio creazione plot e salvataggio dati")
#print(Check_infezioni.cleaned_states)
# salvo i risultati nella cartella
save_in_folder(Check_infezioni)
print("Fine")
