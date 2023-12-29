import os
import pandas as pd
import plotly.express as px
import plotly


def check_folder_exists(name):
    print(os.listdir("."))
    for file in os.listdir("."):
        if file == name:
            print(f"Cartella esiste già, salvo risultati nella cartella --> {name}")
            os.chdir(f"{name}")
            return
        
    print("creo cartelle necessarie... ")
    os.mkdir(f"{name}")
    os.chdir(f"{name}")
    os.mkdir("plots")



def save_in_folder(Infezione):
    lunghezza_dati_esperimenti = len(os.listdir("."))
    with open(f'{lunghezza_dati_esperimenti - 1}.txt', 'w') as f:
        f.write(str(Infezione.cleaned_states))

    # Creare i plot e sarvarli nella cartella plots
    os.chdir("plots")

    tmp_lista_eventi = []
    lunghezza_eventi = len(Infezione.cleaned_states)
    
    tmp_lista_evento1 = []
    tmp_lista_evento2 = []
    tmp_lista_evento3 = []
    tmp_lista_evento4 = []
    tmp_lista_evento5 = []
    tmp_lista_evento6 = []
    tmp_lista_evento7 = []
    
    counter = 0
    
    for data in Infezione.record_tempo:
        if counter >= lunghezza_eventi:
            tmp_lista_eventi.append(0)
            tmp_lista_evento1.append(0)
            tmp_lista_evento2.append(0)
            tmp_lista_evento3.append(0)
            tmp_lista_evento4.append(0)
            tmp_lista_evento5.append(0)
            tmp_lista_evento6.append(0)
            tmp_lista_evento7.append(0)
        else:
            if(data == Infezione.cleaned_states[counter][1]):
                tmp_lista_eventi.append(len(Infezione.cleaned_states[counter][0]))
                tmp_lista_evento1.append(Infezione.cleaned_states[counter][0].count(1))
                tmp_lista_evento2.append(Infezione.cleaned_states[counter][0].count(2))
                tmp_lista_evento3.append(Infezione.cleaned_states[counter][0].count(3))
                tmp_lista_evento4.append(Infezione.cleaned_states[counter][0].count(4))
                tmp_lista_evento5.append(Infezione.cleaned_states[counter][0].count(5))
                tmp_lista_evento6.append(Infezione.cleaned_states[counter][0].count(6))
                tmp_lista_evento7.append(Infezione.cleaned_states[counter][0].count(7))
                counter = counter + 1
            else:
                tmp_lista_eventi.append(0)
                tmp_lista_evento1.append(0)
                tmp_lista_evento2.append(0)
                tmp_lista_evento3.append(0)
                tmp_lista_evento4.append(0)
                tmp_lista_evento5.append(0)
                tmp_lista_evento6.append(0)
                tmp_lista_evento7.append(0)

    dataframe_dati = pd.DataFrame(list(zip(Infezione.record_tempo, Infezione.record_temperatura, Infezione.record_pioggia, Infezione.record_bagnatura_fogliare, Infezione.record_umidità, tmp_lista_eventi)), columns=["tempo", "temperatura", "pioggia", "bagnatura-fogliare", "umidità", "numero-eventi"])
    dataframe_eventi = pd.DataFrame(list(zip(Infezione.record_tempo,tmp_lista_eventi, tmp_lista_evento1, tmp_lista_evento2, tmp_lista_evento3, tmp_lista_evento4, tmp_lista_evento5, tmp_lista_evento6, tmp_lista_evento7, Infezione.record_temperatura, Infezione.record_pioggia)), columns=["tempo", "numero-eventi", "maturazione", "germinazione", "dispersione", "infezione-primaria", "incubazione", "sporulazione", "infezione-secondaria", "temperatura", "pioggia"])


    fig = px.scatter(dataframe_eventi, x="tempo", y=["maturazione", "germinazione", "dispersione", "infezione-primaria", "incubazione", "sporulazione", "infezione-secondaria", "temperatura", "pioggia"],
                 title="Eventi",
                )
    plotly.offline.plot(fig, filename=f'{lunghezza_dati_esperimenti - 1}_eventi.html')
#
    #(dataframe_dati["numero-eventi"])
    fig2 = px.scatter(dataframe_dati, x="tempo", y=["temperatura", "pioggia", "umidità", "bagnatura-fogliare", "numero-eventi"],
                 title="Dati",
                )
    plotly.offline.plot(fig2, filename=f'{lunghezza_dati_esperimenti - 1}_dati.html')

    # ritorno alla root quando finito
    os.chdir("..")
    os.chdir("..")
    print("finito di salvare i risultati")