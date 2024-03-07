Struttura:
--> main.py : Da questo file parte la simulazione dell'algoritmo
--> Infezione.py : classe che contiene i diversi (Peronospora, 3dieci) modelli di infezione ed è responsabile per la raccolta dei dati durante la simulazione
--> Peronospora.py : classe progettata secondo le informazioni estratte da papers
--> Tre_dieci.py : modello che simula il modello "base" ( con questo modello non funzionano i plots ancora )
--> save_results.py : file usato per creare le cartelle per salvare i risulati delle simulazioni

Per far partire un esperimento: python main.py flag1 flag2
 --> Se il flag1 è settato a 0: Salta tuti i "print" della simulazione (versione più veloce)
 --> Se il flag1 è settato a 1: Printa lo stato della simulazione

 --> Se flag2 è settato a 0: L'algoritmo simulerà solo UNA infezione (utile per testare cambiamenti perchè dà risultati rapidi)
 --> Se flag2 è settato a 1: L'algoritmo simula pienamente la situazione sul campo

Il file Peronospora.py:
Contiene la classe che rappresenta l'algoritmo con cui si indetifica lo stato di avanzamento di una o più infezioni di Peronospora.
Ogni fase (sette totali ) è caratterizzata dalla sua funzione con i suoi rispettivi parametri. 
Il ciclo di una singola infezione è rappresentato da una coda di stati, i quali vengono verificati uno alla volta.
Si passa da uno stato all'altro se e solo se si oltrepassano certi thresholds.
I vari stati possono essere: 
	--> maturazione
	--> germinaizione
	--> dispersione
	--> infezione primaria
	--> incubazione
	--> sporulazione
	--> infezione secondaria

il file main.py:
Contiene l'algoritmo principale. All'inizio carichiamo i dati dei sensori e prendiamo solo i datapoints dal primo gennaio in poi.
Dopodichè si cicla sul dataset finale e sul numero delle infezioni (inizialmente 1 ), passando i vari parametri. Se un'infezione a tempo t ha passato lo stato di "maturazione", 
si crea una nuova possibile infezione, che partirà da t+1 in poi. Alla fine del ciclo puliamo, analizziamo i dati e salviamo tutti i risultati nella cartella dell'esperimento