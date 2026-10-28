# Federated learning per il supporto alla Protezione Civile

Questo repository contiene il codice e i risultati della simulazione utilizzata per studiare una possibile integrazione del federated learning in un sistema informativo per la Protezione Civile. Il lavoro nasce da un problema concreto di organizzazione dei dati: informazioni utili per il monitoraggio di un territorio possono essere disponibili presso enti diversi, con competenze, strumenti, qualità e vincoli di accesso differenti. Il federated learning permette di addestrare un modello condiviso senza raccogliere tutti i record di training in un unico archivio.

La simulazione serve a misurare cosa accade, in condizioni controllate, quando cinque fonti con distribuzioni statistiche differenti partecipano a un apprendimento federato. Non è un sistema di allertamento utilizzabile sul campo: non impiega misure storiche di una calamità, non descrive una previsione georeferenziata e non fornisce indicazioni operative per dichiarare uno stato di allerta.

**Quale codice usare:** per replicare l'esperimento descritto nell'articolo si deve utilizzare la cartella **paper_reconstruction/**. Il nome è rimasto invariato perché quel percorso è già stato citato nel manoscritto. Il contenuto non è soltanto una proposta: le **240 valutazioni** della regressione logistica, le **24 righe di riepilogo** e le **12 valutazioni** dell'analisi dei pesi sono state rigenerate e confrontate con i risultati conservati. Il confronto completo è superato anche dal workflow GitHub. L'esatta copia dell'intero sorgente storico non era più disponibile; la corrispondenza numerica è stata verificata, mentre l'identità del commit originario non è dimostrabile.

Questo è l'unico README del progetto. Contiene sia le istruzioni per eseguire il codice sia la descrizione della metodologia, delle scelte progettuali e dei risultati.

## Per iniziare

L'esperimento principale richiede Python 3.11 e NumPy. Per creare un ambiente isolato, partendo dalla radice del repository:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd paper_reconstruction
~~~

Su Windows il comando per attivare l'ambiente virtuale cambia; gli altri comandi Python restano gli stessi. La regressione logistica non richiede TensorFlow. La rete neurale, descritta più avanti, è un esperimento aggiuntivo conservato per completezza e richiede un'installazione separata di TensorFlow.

Il primo controllo non avvia il training. Verifica la generazione dei nodi, le dimensioni delle partizioni, alcuni valori statistici salvati e il fatto che la scelta delle soglie non dipenda dalle etichette del test.

~~~bash
python experiment.py --self-check
python audit_archived.py
~~~

Il secondo comando controlla i risultati archiviati senza addestrare modelli. Se vuoi verificare autonomamente **tutti** i valori della regressione logistica, puoi eseguire:

~~~bash
OPENBLAS_NUM_THREADS=1 python verify_full.py --full
OPENBLAS_NUM_THREADS=1 python verify_full.py --sensitivity
~~~

Il primo comando ripete i dieci seed, addestra tutti gli otto modelli previsti per ciascun seed e controlla le tre policy di soglia per ciascun modello: 10 × 8 × 3 = 240 righe. Il secondo ripete quattro configurazioni dei pesi delle fonti e confronta 4 × 3 = 12 valutazioni. Il parametro ambientale OPENBLAS_NUM_THREADS=1 evita un uso eccessivo di thread nelle operazioni matriciali e riduce le possibili differenze numeriche fra sistemi. Non modifica il protocollo statistico.

Per produrre nuovi file JSON e CSV, separati da quelli usati nell'articolo:

~~~bash
OPENBLAS_NUM_THREADS=1 python experiment.py --all-seeds --output reconstructed_outputs
~~~

Questo comando esegue anche l'analisi di sensibilità iniziale. I nuovi file vengono scritti in reconstructed_outputs/ (dentro la directory di lavoro corrente); i dati conservati in archived_results/ non vengono sovrascritti.

## Come è organizzato il repository

La cartella paper_reconstruction/ contiene il programma che riproduce l'esperimento finale. I file da conoscere sono questi:

| File | Cosa contiene |
| --- | --- |
| **config.py** | Feature, intervalli numerici, nodi, coefficienti di generazione delle etichette, iperparametri, seed e scenari di pesatura. È il punto in cui leggere le impostazioni dell'esperimento. |
| **experiment.py** | Generatore dei dati, partizioni train/validation/test, aggregazione federata, addestramento dei diversi metodi, scelta delle soglie, metriche e salvataggio dei risultati. |
| **model.py** | Implementazione della regressione logistica e delle funzioni di training; contiene anche l'implementazione Keras/TensorFlow usata come riferimento neurale. |
| **verify_full.py** | Riesegue il protocollo della regressione logistica e confronta ciascuna metrica con gli output archiviati, entro una tolleranza numerica dichiarata. |
| **audit_archived.py** | Controlla che file multi-seed e riepiloghi già salvati siano coerenti fra loro, senza training. |
| **schema_adapter.py** | Interfaccia opzionale per leggere colonne CSV e tradurle nella rappresentazione comune a dieci feature. Non è stata validata su dati istituzionali reali. |
| **archived_results/** | JSON relativi agli esperimenti e alle analisi conservati dall'esecuzione precedente, distinti dagli eventuali risultati rigenerati. |

La radice del repository contiene soltanto le dipendenze essenziali, il workflow automatico e questo README. Il vecchio codice della prima simulazione non è più presentato come una seconda implementazione attiva: chi ne avesse bisogno può recuperarlo dalla cronologia Git, precedente al consolidamento finale.

Per orientarsi nel programma, una lettura utile è config.py → generate_synthetic() → stratified_split() → make_nodes() → train_federated() e run_one_seed() → evaluate() → verify_full.py. Sono le funzioni che collegano le ipotesi iniziali ai valori riportati nel paper.

## Cosa rappresentano i cinque nodi

I client rappresentano **categorie ipotetiche di fonti**, non vere organizzazioni che hanno condiviso dati. Le denominazioni istituzionali servono a rendere leggibile lo scenario.

| Nodo | Fonte simulata | Record | Peso di fonte | Probabilità di corruzione etichetta |
| --- | --- | ---: | ---: | ---: |
| A | Fonte nazionale / INGV-DPC | 60.000 | 1,80 | 0,008 |
| B | Regione / Centro Funzionale | 45.000 | 1,35 | 0,012 |
| C | Comune / COC | 30.000 | 1,00 | 0,018 |
| D | Sensori su infrastrutture critiche | 50.000 | 1,20 | 0,014 |
| E | Segnalazioni dei cittadini | 25.000 | 0,55 | 0,030 |
| **Totale** | | **210.000** | | |

Il peso di fonte è un **coefficiente assunto a priori**: non è un indice calcolato da un algoritmo di reputazione, non proviene da una valutazione reale dell'ente e non viene prodotto automaticamente da un Data Lifecycle Model. Nell'architettura descritta nell'articolo, un futuro processo di governance potrebbe assegnare pesi motivati e verificabili; nel codice presente sono parametri impostati dall'autore per studiarne l'effetto.

I cinque nodi hanno quantità diverse di dati, distribuzioni diverse delle variabili e diverse frequenze della classe positiva. È questo il significato preciso di **non-IID** (non indipendenti e identicamente distribuiti) in questo esperimento. Non viene simulata, invece, una differenza nel numero o nel significato delle colonne: tutti i client possiedono le stesse dieci feature.

## Quali informazioni vengono simulate

Ogni record contiene dieci variabili. I valori del generatore sono già nell'intervallo [0,1]. L'adattatore CSV dispone anche di intervalli convenzionali per trasformare eventuali valori grezzi; tali intervalli **non** indicano misure osservate o una calibrazione idrologica.

| Feature, nell'ordine usato dal modello | Interpretazione | Intervallo grezzo previsto dall'adattatore |
| --- | --- | --- |
| rainfall_1h | Pioggia cumulata nell'ultima ora | 0–80 |
| rainfall_3h | Pioggia cumulata nelle ultime tre ore | 0–160 |
| rainfall_24h | Pioggia cumulata nelle ultime 24 ore | 0–300 |
| river_level | Livello del fiume | 0–8 |
| river_delta_1h | Variazione oraria del livello | −2–2 |
| soil_saturation | Indicatore di saturazione del suolo | 0–100 |
| event_reports | Numero/indicatore di segnalazioni | 0–500 |
| road_closures | Numero/indicatore di chiusure | 0–100 |
| sensor_anomaly | Indicatore di anomalia dei sensori | 0–1 |
| territorial_risk_class | Classe convenzionale del rischio territoriale | 0–1 |

Le prime nove variabili sono numeriche; territorial_risk_class prende soltanto uno dei valori sintetici 0,2, 0,5 e 0,8. La variabile obiettivo, chiamata alert, è binaria: 1 denota una condizione positiva secondo la regola sintetica, 0 una negativa. Non rappresenta un evento verificato da un ufficio di Protezione Civile.

L'ordine delle dieci feature è essenziale. La regressione logistica aggrega coefficienti associati alla stessa posizione del vettore. Se un nodo scambiasse, per esempio, rainfall_24h con river_level, l'aggregazione avrebbe una struttura matematica formalmente valida ma un significato sbagliato. Il cosiddetto *feature contract* è precisamente l'accordo su nomi, ordine e scala.

## Come vengono create le distribuzioni

La funzione generate_synthetic() parte da un generatore NumPy indipendente per ciascun nodo. Gli offset iniziali sono 101, 202, 303, 404 e 505; a essi si aggiunge l'offset di replica, se presente.

Per le variabili continue principali viene usata la distribuzione **Beta(a,b)**. Una Beta è definita sull'intervallo [0,1] e permette di generare distribuzioni asimmetriche, concentrate vicino a zero o distribuite più uniformemente, senza introdurre misure fisiche fuori scala. Non è stata stimata da piogge, livelli fluviali o segnalazioni reali. È una scelta per costruire un benchmark controllabile.

La media teorica di una Beta è a/(a+b). Per esempio, rainfall_1h del nodo A usa Beta(2,1; 5,2), con media circa 0,288, mentre il nodo E usa Beta(1,6; 5,8), con media circa 0,216. Questo introduce una differenza statistica sistematica fra client, senza affermare che i cittadini osservino realmente precipitazioni più basse.

Qui sotto sono riportati **tutti** i parametri delle otto colonne estratte direttamente da Beta. In ogni cella compare la coppia **(a; b)**.

| Variabile | A nazionale | B regionale | C comunale | D sensori | E cittadini |
| --- | --- | --- | --- | --- | --- |
| rainfall_1h | (2,1; 5,2) | (2,0; 5,0) | (1,8; 5,5) | (1,7; 5,5) | (1,6; 5,8) |
| rainfall_24h | (2,8; 3,2) | (2,5; 3,5) | (2,0; 4,0) | (2,0; 4,2) | (1,8; 4,6) |
| river_level | (2,3; 2,9) | (2,0; 3,0) | (1,8; 3,8) | (2,4; 3,0) | (1,6; 4,0) |
| river_delta_1h | (2,1; 4,8) | (1,8; 5,0) | (1,5; 4,5) | (2,0; 4,2) | (1,4; 4,8) |
| soil_saturation | (2,7; 2,4) | (2,5; 2,5) | (2,0; 3,0) | (2,3; 3,0) | (1,8; 3,4) |
| event_reports | (1,3; 7,5) | (1,2; 8,0) | (2,5; 4,0) | (1,1; 9,0) | (3,2; 3,6) |
| road_closures | (1,2; 9,0) | (1,1; 10,0) | (2,0; 5,0) | (1,1; 10,0) | (2,6; 4,2) |
| sensor_anomaly | (2,2; 4,5) | (1,3; 7,0) | (1,2; 8,0) | (2,8; 3,8) | (1,1; 8,5) |

**rainfall_3h non è estratta da una Beta indipendente.** Viene costruita a partire da rainfall_1h aggiungendo un termine normale, quindi troncando il risultato nell'intervallo [0,1]:

~~~text
rainfall_3h = clip(rainfall_1h + Normal(mu, sigma), 0, 1)
~~~

| Nodo | mu | sigma |
| --- | ---: | ---: |
| A | 0,10 | 0,08 |
| B | 0,10 | 0,10 |
| C | 0,08 | 0,15 |
| D | 0,07 | 0,12 |
| E | 0,06 | 0,18 |

Questa dipendenza è introdotta per evitare che tutte le feature risultino indipendenti fra loro. È comunque una rappresentazione semplificata: non garantisce la coerenza fisica fra cumulata a una, tre e ventiquattro ore.

La classe territoriale è campionata con le probabilità seguenti, nell'ordine dei valori (0,2; 0,5; 0,8):

| Nodo | P(0,2) | P(0,5) | P(0,8) |
| --- | ---: | ---: | ---: |
| A | 0,40 | 0,42 | 0,18 |
| B | 0,45 | 0,40 | 0,15 |
| C | 0,55 | 0,30 | 0,15 |
| D | 0,50 | 0,35 | 0,15 |
| E | 0,60 | 0,28 | 0,12 |

Le forme differenti delle distribuzioni, insieme ai bias e al rumore delle etichette descritti nel prossimo paragrafo, sono ciò che controlla la difficoltà dell'esperimento. Non vanno interpretate come stime della qualità delle fonti istituzionali italiane.

## Come viene attribuita l'etichetta

Una volta generate le dieci feature, per ogni record viene calcolato un punteggio **logit** mediante una formula fissata prima dell'esperimento. Il generatore è diverso dal classificatore che successivamente cercherà di apprendere i dati.

La formula utilizzata è:

~~~text
z = -20
    + 2.4*rainfall_1h
    + 3.3*rainfall_3h
    + 5.4*rainfall_24h
    + 5.1*river_level
    + 4.2*river_delta_1h
    + 4.8*soil_saturation
    + 3.6*event_reports
    + 3.0*road_closures
    + 4.5*sensor_anomaly
    + 3.6*territorial_risk_class
    + 5.4*(rainfall_24h*soil_saturation)
    + 3.6*(river_level*river_delta_1h)
    + node_bias
~~~

Gli ultimi due termini moltiplicativi servono a produrre interazioni non lineari: a parità di pioggia, per esempio, la saturazione del suolo può modificare il punteggio sintetico. Non sono coefficienti derivati da un modello fisico validato.

Il valore z viene convertito in una probabilità tramite la sigmoide: p = 1/(1+exp(−z)). Poi si estrae un'etichetta da una Bernoulli(p). Questo significa che, anche con le stesse feature, l'etichetta può variare: il target non viene fissato deterministicamente applicando una semplice soglia al logit.

Il bias cambia da nodo a nodo:

| Fonte | Bias sul logit | Probabilità di inversione dell'etichetta |
| --- | ---: | ---: |
| A nazionale | −0,40 | 0,8% |
| B regionale | −0,10 | 1,2% |
| C comunale | +0,35 | 1,8% |
| D sensori | +0,20 | 1,4% |
| E cittadini | +0,90 | 3,0% |

Dopo il campionamento Bernoulli, ogni etichetta ha una probabilità indipendente, definita per fonte, di essere invertita da 0 a 1 oppure da 1 a 0. È il **label noise**. La percentuale di etichette cambiate non coincide necessariamente con la prevalenza di classi positive osservata.

Nel dataset di riferimento le prevalenze positive sono circa 13,4%, 5,9%, 5,3%, 7,0% e 5,5% nei nodi A–E. Queste prevalenze sono il risultato del generatore, **non un parametro fissato direttamente**. Nel file archived_results/node_non_iid_diagnostics.json si trovano prevalenze e statistiche descrittive per ciascuna partizione.

## Come si ottengono train, validation e test

Il generatore crea 210.000 esempi complessivi. Ogni nodo viene partizionato separatamente, stratificando per etichetta 0 e 1, così da mantenere quanto possibile le rispettive proporzioni.

La procedura esatta è importante per riprodurre i numeri. Per ogni nodo viene inizializzato un nuovo generatore casuale con **lo stesso seed usato per creare quel nodo**. Gli indici della classe 0 sono mescolati, poi vengono mescolati quelli della classe 1. **Per ciascuna classe**, prima si prendono int(0,25 × n_classe) indici per il test, poi int(0,15 × n_classe) per la validation; il resto va al training. La funzione int() tronca verso zero, non arrotonda al valore più vicino.

Da qui nasce la differenza fra il semplice 60% di 210.000 e il numero effettivo di righe di training:

| Nodo | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| A | 36.001 | 8.999 | 15.000 |
| B | 27.002 | 6.749 | 11.249 |
| C | 18.001 | 4.499 | 7.500 |
| D | 30.002 | 7.499 | 12.499 |
| E | 15.000 | 3.750 | 6.250 |
| **Totale** | **126.006** | **31.496** | **52.498** |

L'assenza di sovrapposizioni fra i tre insiemi è verificata da assert nel codice. La validation serve **soltanto** per selezionare le soglie decisionali. Il test viene usato dopo quella scelta, per calcolare le prestazioni presentate nel paper. Il confronto di controllo modifica le etichette del test e verifica che le soglie scelte sulla validation non cambino: è una verifica specifica contro il *test leakage*.

In questa simulazione non esiste uno split temporale, perché non esistono cronologie autentiche degli eventi. Per un vero sistema di allertamento sarebbe necessario separare nel tempo le osservazioni e controllare che nessuna variabile incorpori informazioni disponibili solo dopo l'evento.

## I quattro modi in cui viene fatto il training

Il confronto si svolge sempre sullo stesso insieme globale di test, ottenuto unendo le partizioni di test dei cinque nodi. Per ciascun seed vengono addestrati **otto classificatori**: cinque locali, due federati e uno centralizzato.

**Local-only.** Si addestra una regressione logistica distinta sui dati di training di ciascun nodo. Ognuna delle cinque regressioni viene poi valutata sul test complessivo dei cinque nodi. In questo modo si misura quanto un modello addestrato su una sola distribuzione sappia generalizzare sull'intero ecosistema. Non è un confronto delle prestazioni dei cinque nodi soltanto sui rispettivi test locali.

**FedAvg classico.** A ogni round, tutti i nodi ricevono lo stesso vettore globale iniziale, lo aggiornano con i propri dati e restituiscono i nuovi parametri. L'orchestratore calcola la media usando come pesi le dimensioni dei training set. Nessun record locale entra nell'insieme di training dell'orchestratore.

**FedAvg pesato per fonte.** Si usa lo stesso procedimento, ma la quota di ciascun nodo è proporzionale sia al numero di record di training sia al suo peso di fonte w. Viene descritta con la formula alpha_i = n_i·w_i / somma_j(n_j·w_j). Il peso è costante durante i round e definito dalla configurazione; non è apprendimento automatico della reputazione.

**Centralizzato.** Si concatenano le cinque partizioni di training e si addestra un unico modello, utilizzando una sola macchina logica che dispone di tutti i record. È un riferimento sperimentale che rinuncia al vincolo di località dei dati. Non è un limite superiore matematico: tempi, traiettorie di ottimizzazione e numero totale di aggiornamenti non sono resi identici a quelli di FedAvg.

Questa è una **federazione a un solo livello**. I client sono separati logicamente nello stesso processo software, non distribuiti su server di cinque istituzioni. La gerarchia regionale–nazionale del paper è una proposta architetturale; non è stata implementata o misurata nell'esperimento.

## La regressione logistica, senza scorciatoie

Il modello principale usa dieci coefficienti, uno per feature, più un'intercetta: **11 parametri** in totale. La probabilità predetta è una sigmoide applicata a una combinazione lineare delle dieci variabili. A differenza del generatore, il modello **non riceve esplicitamente le due interazioni** pioggia–suolo e livello–variazione. Non si tratta quindi di ricostruire banalmente la formula generativa: il classificatore cerca una buona approssimazione con una rappresentazione più semplice.

I valori della configurazione principale sono:

| Opzione | Valore |
| --- | ---: |
| Ottimizzazione | Gradient descent full-batch |
| Learning rate | 0,12 |
| Penalità L2 | 0,001 |
| Bilanciamento delle classi | "balanced" |
| Iterazioni local-only | 900 |
| Iterazioni centralizzato | 900 |
| Round federati | 60 |
| Iterazioni locali per round | 18 |
| Parametri scambiati per modello | 11 |

Per il bilanciamento, il peso di ciascuna classe è n/(2·n_classe), calcolato sul training set utilizzato da quel modello. Nella funzione di perdita e nel gradiente questi pesi contrastano lo sbilanciamento delle etichette. L'intercetta non riceve penalità L2, mentre gli altri coefficienti sì. La configurazione non attua *gradient clipping*, differential privacy o ottimizzazione asincrona.

I cinque modelli federati locali partono dallo stesso vettore di parametri ricevuto dal server a ogni round; tutti partecipano a tutti i round. Non sono simulate interruzioni di rete, dropout di client, ritardi o decisioni di ammissione dinamiche.

L'ottimizzazione federata comprende 60 × 18 = **1.080 iterazioni locali per client**; il centralizzato e ciascun local-only eseguono **900 iterazioni**. Questo è uno dei motivi per cui una piccola differenza di F1 fra FedAvg e centralizzato non può essere automaticamente interpretata come superiorità dell'algoritmo di aggregazione.

## Cosa cambia quando usiamo i pesi delle fonti

La media FedAvg classica attribuisce al nodo A circa il 28,6% dell'aggregazione e al nodo E circa l'11,9%, in funzione delle rispettive dimensioni di training. Con i pesi istituzionali ipotizzati nella configurazione, le quote diventano circa **39,6%** per A e **5,0%** per E.

I pesi effettivi dipendono quindi da due fattori distinti: numerosità e coefficiente w. Un nodo piccolo ma molto pesato può aumentare la propria influenza, ma non è corretto dire che il valore w da solo coincida con la percentuale finale di contributo.

Per controllare la dipendenza dai pesi abbiamo salvato quattro scenari:

| Scenario | A nazionale | B regione | C comune | D sensori | E cittadini |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| baseline | 1,80 | 1,35 | 1,00 | 1,20 | 0,55 |
| extreme | 3,00 | 1,80 | 1,00 | 1,50 | 0,25 |
| inverted | 0,55 | 1,00 | 1,35 | 1,20 | 1,80 |

L'analisi usa il **dataset di riferimento con offset seed 0**, non i dieci offset della tabella principale. Per questo il baseline presenta F1 = 0,5442 nella sensitivity analysis, mentre la media multi-seed del source-weighted FedAvg è circa 0,5554: **non sono due misure dello stesso insieme di repliche**.

I quattro risultati F1, sotto la policy best-F1, sono rispettivamente **0,5451**, **0,5442**, **0,5419** e **0,5441**. Le variazioni sono modeste in questa realizzazione. Non provano che la pesatura funzioni sempre, né che difenda da nodi malevoli. I pesi non effettuano ispezione degli aggiornamenti e non implementano una difesa *Byzantine-robust*.

## Come vengono decise le soglie

I modelli producono uno score continuo. La decisione binaria non viene fissata alla soglia 0,5: il codice seleziona una soglia su un insieme di candidati da **0,05 a 0,95 inclusi**, con passo **0,01**. Per ciascun modello si calcolano gli score su tutte le partizioni di validation, che vengono riunite in un unico insieme di valutazione.

Le tre policy sono:

- **best_f1:** la soglia che massimizza l'F1 sulla validation;
- **best_balanced:** la soglia che massimizza la balanced accuracy sulla validation;
- **best_operational:** la soglia che massimizza la precision fra quelle la cui recall sulla validation è almeno 0,90.

Se più soglie producono lo stesso valore della metrica ottimizzata, il codice seleziona la prima nell'ordine crescente della griglia. Se nessuna soglia soddisfa il vincolo operativo di recall, quella policy non viene prodotta.

Dopo aver scelto le soglie, si calcolano gli score sul test, **senza riottimizzare le soglie**. I risultati best_operational possono quindi avere recall sul test inferiore a 0,90 anche quando il vincolo era rispettato sulla validation.

Lo score di una regressione allenata con pesi di classe non è automaticamente una probabilità ben calibrata di allerta reale. L'uso operativo richiederebbe anche calibrazione, stima del costo dei falsi allarmi e validazione rispetto a eventi osservati.

## Come leggere le metriche

La classe positiva è alert = 1. Una volta selezionata la soglia si ottengono TP (veri positivi), FP (falsi positivi), TN (veri negativi) e FN (falsi negativi). Le metriche principali sono:

| Metrica | Calcolo / significato |
| --- | --- |
| Precision | TP/(TP+FP): fra gli allarmi emessi, quanti erano positivi |
| Recall | TP/(TP+FN): fra i positivi, quanti vengono intercettati |
| Specificity | TN/(TN+FP): fra i negativi, quanti non producono allarme |
| F1 | 2·precision·recall/(precision+recall) |
| Balanced accuracy | (recall+specificity)/2 |
| FPR | FP/(FP+TN): frequenza di falsi allarmi fra i negativi |
| FNR | FN/(FN+TP): frazione dei positivi mancati |
| ROC-AUC | Area sotto la curva che confronta TPR e FPR a diverse soglie |
| PR-AUC nel paper | **Average precision**, non integrazione trapezoidale generica della curva PR |

La ROC-AUC e la average precision sono calcolate sugli score continui del **test**; non dipendono dalla soglia selezionata. Per questo i loro valori sono ripetuti nelle righe delle diverse policy relative allo stesso modello e seed.

La average precision è ottenuta ordinando i record per score decrescente e facendo la media della precision ai ranghi dei campioni positivi. È questa definizione, utilizzata dal codice, che nel testo del paper viene indicata come PR-AUC.

## I risultati da citare nel paper

I valori sotto si riferiscono alla regressione logistica, alla policy **best_f1**, a **dieci seed** e al test globale dei cinque nodi. Le deviazioni standard sono quelle delle dieci repliche; non sono intervalli di confidenza.

| Impostazione | F1 (media ± dev. std) | ROC-AUC media | Average precision media |
| --- | ---: | ---: | ---: |
| Local-only, media dei cinque nodi | 0,539 | 0,855 | 0,556 |
| Local-only, miglior nodo A | 0,547 ± 0,006 | 0,855 | 0,567 |
| Local-only, peggior nodo E | 0,525 ± 0,008 | 0,852 | 0,539 |
| **FedAvg classico** | **0,557 ± 0,008** | **0,863** | **0,579** |
| FedAvg pesato per fonte | 0,555 ± 0,007 | 0,862 | 0,578 |
| Centralizzato | 0,548 ± 0,004 | 0,854 | 0,567 |

La media local-only in tabella è una media aritmetica delle cinque medie per nodo, non un nono modello addestrato. Il risultato più difendibile è che **il federato raggiunge prestazioni comparabili al centralizzato in questa simulazione mantenendo separati i dati di training**. Non emerge un vantaggio sistematico della pesatura delle fonti rispetto a FedAvg classico. Il piccolo vantaggio medio di FedAvg sul centralizzato non basta, da solo, a provare una superiorità generale: anche gli schemi di ottimizzazione e il numero di passi differiscono.

Nella policy best_operational la recall media sul test è circa **0,903** per FedAvg classico, **0,902** per quello pesato e **0,902** per il centralizzato. Il prezzo è una frequenza di falsi positivi elevata: circa **0,525** e **0,526** per i due federati, contro **0,551** per il centralizzato. Questi numeri illustrano il compromesso fra copertura dei positivi e falsi allarmi, non una soluzione già pronta per un ufficio di emergenza.

La sensitivity analysis, i risultati a soglia fissa, le altre policy e le diagnostiche complete si trovano nei JSON di archived_results/. I file principali sono multiseed_results.json, multiseed_summary.json, source_weight_sensitivity.json, threshold_search_summary.json e node_non_iid_diagnostics.json.

## Quanti dati vengono spostati

Nella simulazione, gli esempi non vengono inviati al server federato. La misura quantitativa presente nel paper è però una **stima del numero di valori scalari** trasferiti, non una misura di byte su rete.

Per il riferimento centralizzato si conta un singolo trasferimento degli N_train record, ognuno con dieci feature più l'etichetta:

~~~text
126006 × (10 + 1) = 1.386.066 scalari
~~~

Per FedAvg con regressione logistica si contano, a ogni round, l'invio del modello globale a cinque nodi e il ritorno degli aggiornamenti: due trasferimenti per nodo, 11 coefficienti per modello, 60 round.

~~~text
2 × 5 × 11 × 60 = 6.600 scalari
~~~

Il rapporto 1.386.066/6.600 è circa **210**. Questa riduzione è specifica della dimensione del modello logistico, del numero di client e del numero di round scelto. Non è una proprietà generale del federated learning.

Per la rete neurale di riferimento i parametri sono **11.777**: con 25 round e cinque nodi il proxy diventa **2.944.250 scalari**, cioè supera il conteggio del trasferimento centralizzato. Nel calcolo non rientrano identificativi, intestazioni dei messaggi, cifratura, ritrasmissioni, compressione, metadati, tempi di rete o energia.

Il vantaggio generale che qui si osserva è la **località dei record**, non la certezza di usare meno banda in qualsiasi condizione.

## I risultati della rete neurale

In model.py è presente anche un modello Keras/TensorFlow con tre strati nascosti da **128, 64 e 32 unità**, attivazioni Swish, dropout **0,20 / 0,15 / 0,10** e uscita sigmoide. La configurazione attiva non inserisce Batch Normalization. L'ottimizzatore è AdamW con learning rate **0,0008**, weight decay **0,0001**, batch size **512** e class weighting bilanciato.

Il training standalone prevede al massimo **40 epoche**, con early stopping (patience 6) e una validation interna al fitting pari al 12% del relativo insieme di training. La versione federata prevede **25 round**, ciascuno con **3 epoche locali**, senza early stopping locale. Il seed configurato per la rete è **777**.

Gli output della rete neurale sono conservati nell'archivio del progetto, ma **non sono stati riprodotti indipendentemente nella verifica finale**. La riproducibilità numerica attestata dagli script e da GitHub Actions riguarda i risultati della **regressione logistica**, non la DNN. Il comando opzionale seguente esegue una nuova sperimentazione, non una verifica certificata del risultato neurale storico:

~~~bash
python -m pip install tensorflow==2.19.1
OPENBLAS_NUM_THREADS=1 python experiment.py --neural --output neural_new_run
~~~

Una nuova esecuzione neurale può essere influenzata dalla versione TensorFlow, dal dispositivo e dalla non-deterministicità di alcune operazioni. Non va mescolata alla tabella multi-seed della regressione logistica, che contiene dieci repliche.

## L'interfaccia per dati CSV

schema_adapter.py conserva un meccanismo per leggere un CSV, ricostruire le dieci feature nell'ordine comune e riportarle negli intervalli [0,1]. Una regola di mapping può indicare una colonna esistente, una costante, una media di più colonne oppure una trasformazione lineare con *scale* e *offset*.

Dopo il mapping, per ciascuna feature il valore grezzo viene trasformato con (x − minimo)/(massimo − minimo) e troncato nell'intervallo [0,1]. I valori non finiti vengono sostituiti con il punto medio dell'intervallo previsto. Il target è letto dalla colonna alert e convertito a intero.

Questo adapter è un **punto di integrazione**, non una validazione del sistema su dati veri. Per collegare un ente reale servirebbero un contratto semantico, controlli di qualità, gestione del dato mancante motivata scientificamente, procedure autorizzative, definizione della variabile obiettivo e test di generalizzazione. Riempire una feature mancante con una costante può rendere il codice eseguibile, ma non garantisce che il risultato abbia un significato predittivo corretto.

## Come riconosciamo una riproduzione corretta

La verifica non consiste nel guardare soltanto l'F1 medio arrotondato a tre cifre. verify_full.py prende ogni riga ricalcolata e la associa alla riga archivata mediante seed, modello, paradigma, impostazione e policy. Confronta **F1, recall, specificity, balanced accuracy, FPR, FNR, ROC-AUC, average precision e soglia**. La tolleranza impostata è 10^−9 sulla differenza assoluta; se un record manca o il valore di una metrica differisce oltre tale soglia, il controllo fallisce.

L'audit completo ha trovato corrispondenza su **240/240** righe multi-seed e **24/24** riepiloghi statistici, con differenze numeriche massime nell'ordine di 10^−16. La sensibilità ha dato corrispondenza su **12/12** righe. La verifica GitHub ripete questi controlli in ambiente isolato.

Questa evidenza consente di affermare che **il protocollo logistico finale è numericamente riproducibile**. Non consente di affermare che la cartella ricostruita sia una copia byte-per-byte degli script usati mesi prima. Il generatore, lo split e l'ottimizzazione sono stati recuperati confrontando codice e output fino a ottenere le stesse metriche; l'intero stato originale dell'ambiente di esecuzione non è stato conservato.

La cartella archived_results/ è intenzionalmente separata dal nuovo output. È l'archivio di confronto, non una destinazione di salvataggio. Non modificare i suoi file per fare tornare un test: una differenza deve essere analizzata nel codice o nella configurazione.

## Cosa modificare, e con quali conseguenze

Le scelte principali si trovano in config.py. Cambiare una distribuzione Beta, un coefficiente del logit, un bias, il label noise, un seed o una dimensione campionaria produce un **altro dataset** e quindi un **altro esperimento**. Lo stesso vale per learning rate, L2, round, epoche o pesi di fonte. È lecito farlo per ricerche successive, ma i nuovi risultati non vanno sostituiti ai valori del paper senza dichiarare il cambiamento di protocollo.

Sono presenti anche alcune impostazioni denominate QUICK_ e la variabile EXPERIMENT_MODE. Riguardano un'impostazione di debug conservata nella configurazione fornita dagli autori. **Il programma principale experiment.py usa esplicitamente le costanti PAPER_**: cambiare soltanto EXPERIMENT_MODE non trasforma l'esecuzione in una prova rapida. Per una prova senza training usa --self-check; per l'esperimento del paper usa --all-seeds. Questa distinzione evita che una modifica apparentemente innocua produca aspettative sbagliate.

In una futura estensione andrebbero versionati esplicitamente: la definizione delle feature, l'origine delle misure, i criteri di qualità, la definizione temporale del target, la strategia di partizionamento, la funzione di perdita, le soglie, la versione del modello e la configurazione di comunicazione.

## Quali conclusioni non possiamo trarre

Questo repository permette di studiare una procedura di apprendimento federato in un ambiente sintetico. Non dimostra che l'architettura più ampia proposta nell'articolo sia già stata installata presso la Protezione Civile, che gli aggiornamenti del modello siano protetti da secure aggregation o differential privacy, o che i nodi siano resistenti a tentativi di poisoning. Nel prototipo i dati sono separati **logicamente in memoria**; il trasferimento dei soli parametri non basta a garantire la riservatezza formale.

La simulazione non comprende veri eventi, calendario, sequenze temporali, bacini georeferenziati, previsioni meteo, reti di sensori fisici o attivazione di procedure istituzionali. Pur rappresentando diverse distribuzioni statistiche, usa lo stesso insieme di feature in ogni nodo e non verifica l'interoperabilità di sistemi informativi reali. Non sono misurate latenza di allerta, energia, throughput, continuità operativa o banda effettiva su una rete geografica.

Per queste ragioni il risultato deve essere descritto come **proof of concept sintetico single-level**. La federazione gerarchica regionale–nazionale, il modello di governance basato su Data Lifecycle Models e gli eventuali controlli di sicurezza sono elementi della proposta architetturale, non risultati sperimentali già dimostrati.

## Provenienza, versione e riferimento nel manoscritto

Il primo codice pubblico, oggi conservato nella **cronologia Git**, utilizzava un'intercetta sintetica pari a −8, tre seed e una selezione delle soglie sul test. Non corrisponde alla metodologia né ai valori della versione finale dell'articolo.

La versione consolidata conserva il codice di configurazione e dei modelli fornito per la fase del paper, ricostruisce il resto del protocollo e lo verifica contro gli output originali archiviati. Il collegamento stabile alla cartella sperimentale è:

https://github.com/christianriccio/federated-disaster/tree/main/paper_reconstruction

Per citare un insieme immutabile di sorgenti, risultati e configurazione, è preferibile utilizzare anche **l'hash del commit finale** mostrato nella cronologia Git, invece di fare affidamento esclusivamente sul ramo main, che potrebbe essere aggiornato in futuro.

Nel manoscritto si può dichiarare che il codice e i risultati sintetici sono disponibili pubblicamente, che l'esperimento di regressione logistica è stato riprodotto numericamente e che la rete neurale costituisce un riferimento separato non riprodotto nell'audit. Non si devono attribuire all'esperimento dati osservati, garanzie di privacy o prestazioni di una rete gerarchica che non sono stati verificati.
