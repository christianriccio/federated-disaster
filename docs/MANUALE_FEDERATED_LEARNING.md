# Manuale teorico-pratico del progetto Federated Learning

Questo documento spiega il progetto dal punto di vista teorico, pratico e
implementativo. L'obiettivo e' permettere di validare il codice, spiegare le
scelte metodologiche e riusare il materiale nella stesura del paper.

## 1. Obiettivo del progetto

Il progetto simula un sistema di monitoraggio e allarme per la prevenzione dei
disastri in una rete federata ispirata alla Protezione Civile.

L'idea e' che dati e modelli siano distribuiti tra nodi diversi:

- fonte nazionale, per esempio INGV/DPC;
- fonte regionale, per esempio Centro Funzionale;
- fonte locale, per esempio Comune/COC;
- sensori e infrastrutture critiche;
- segnalazioni dei cittadini.

Nel paradigma centralizzato tutti i dati vengono spostati verso un unico server.
Nel paradigma federato, invece, ogni nodo mantiene localmente i propri dati,
addestra un modello locale e invia all'orchestratore solo i parametri del
modello. L'orchestratore aggrega i parametri e produce un modello globale.

La domanda sperimentale principale e':

> Un modello federato puo' ottenere prestazioni comparabili a un modello
> centralizzato, riducendo al tempo stesso la necessita' di spostare dati?

Questa domanda va letta con attenzione. Il progetto non vuole dimostrare che il
federated learning sia sempre migliore del centralizzato. Vuole verificare se,
in un dominio dove i dati sono distribuiti per natura, il paradigma federato
riesce a ottenere prestazioni utili mantenendo i dati presso le fonti. La
valutazione quindi deve guardare insieme:

- prestazioni predittive;
- robustezza su nodi eterogenei;
- movimento di dati o parametri;
- interpretabilita' del benchmark logistico;
- utilita' operativa delle soglie in uno scenario di early warning.

## 2. Benchmark sperimentali

Il progetto confronta due famiglie di modelli:

1. Regressione logistica
2. Rete neurale Keras/TensorFlow

Per ciascun modello vengono confrontati:

1. baseline local-only, riportando un modello per ogni nodo;
1. training federato con FedAvg classico;
2. training federato con FedAvg pesato per fonte;
3. training centralizzato, cioe' tutti i dati in un unico server.

La baseline local-only significa: "addestro un modello usando solo i dati di un
singolo nodo, poi lo valuto sul test globale composto dai test set di tutti i
nodi". Nel codice non viene piu' scelto soltanto il miglior local-only sul test,
perche' quella scelta sarebbe troppo ottimistica per un paper. Viene invece
salvata una riga per ogni nodo locale. In questo modo puoi leggere:

- quale nodo generalizza meglio;
- quale nodo generalizza peggio;
- quanto varia la performance tra fonti;
- se il modello federato migliora rispetto allo scenario in cui gli enti
  lavorano isolatamente.

Nel paper puoi poi calcolare, se serve, il migliore, il peggiore e la media dei
local-only, ma il codice conserva l'informazione completa.

La regressione logistica e' il benchmark principale perche':

- e' interpretabile;
- produce pesi leggibili feature per feature;
- e' un modello standard per classificazione binaria;
- permette di verificare se la rete neurale porta reale beneficio.

La rete neurale Keras/TensorFlow e' il modello piu' flessibile:

- puo' modellare relazioni non lineari;
- usa framework standard in ambito machine learning;
- permette di rappresentare uno scenario piu' vicino a un'applicazione ML reale.

## 3. Struttura del progetto

La directory principale contiene:

```text
federated/
  artifacts.py
  config.py
  data.py
  federated.py
  main.py
  metrics.py
  model.py
  reporting.py
requirements.txt
outputs/
  results_summary.csv
  results_summary.json
  federated_round_history.csv
  federated_round_history.json
  threshold_search_summary.csv
  threshold_search_summary.json
  node_non_iid_diagnostics.csv
  node_non_iid_diagnostics.json
  multiseed_results.csv
  multiseed_results.json
  multiseed_summary.csv
  multiseed_summary.json
  paper_ready_results.csv
  paper_ready_results.json
  plots/
    convergence_train_loss.svg
    convergence_train_f1.svg
    convergence_train_accuracy.svg
    convergence_train_precision.svg
    convergence_train_recall.svg
    final_f1_comparison.svg
    final_accuracy_comparison.svg
    training_time_comparison.svg
    data_movement_comparison.svg
```

Ruolo dei file:

- `config.py`: contiene configurazioni, feature, nodi simulati, iperparametri.
- `data.py`: genera dati sintetici e carica dataset reali eterogenei.
- `model.py`: implementa regressione logistica e rete neurale Keras.
- `federated.py`: implementa FedAvg classico e FedAvg pesato.
- `metrics.py`: calcola metriche binarie, AUC, matrice di confusione e soglie.
- `reporting.py`: stampa riepiloghi e interpretazione dei risultati.
- `artifacts.py`: salva CSV, JSON e grafici SVG per il paper.
- `main.py`: orchestra l'intero esperimento.

## 4. Flusso generale dell'esperimento

Il flusso eseguito da `main.py` e':

```text
1. Crea i nodi federati sintetici
2. Stampa la sintesi dei nodi
3. Esegue benchmark regressione logistica
   3.1 local-only baseline, un modello per ogni nodo
   3.2 FedAvg classico
   3.3 FedAvg pesato per fonte
   3.4 modello centralizzato
4. Esegue benchmark rete neurale Keras/TensorFlow se TensorFlow e' installato
   4.1 FedAvg classico
   4.2 FedAvg pesato per fonte
   4.3 modello centralizzato
5. Stampa pesi della regressione logistica federata pesata
6. Stampa tabella finale di confronto
7. Salva risultati e grafici in outputs/
```

## 5. Configurazione del progetto

File: `federated/config.py`

### 5.1 Feature canoniche

La lista `FEATURES` definisce l'ordine standard delle colonne:

```python
FEATURES = [
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_24h",
    "river_level",
    "river_delta_1h",
    "soil_saturation",
    "event_reports",
    "road_closures",
    "sensor_anomaly",
    "territorial_risk_class",
]
```

L'ordine e' importante: ogni nodo deve produrre una matrice `X` con le stesse
colonne nello stesso ordine. Solo cosi' i pesi dei modelli locali sono
comparabili e aggregabili dall'orchestratore.

### 5.2 Range fisici

`FEATURE_RANGES` definisce i range usati per normalizzare dati reali:

```python
"rainfall_24h": (0.0, 300.0)
```

Se un dataset reale riporta pioggia in millimetri, il valore viene portato nel
range 0-1 con una normalizzazione min-max:

```text
x_norm = (x - min_value) / (max_value - min_value)
```

I valori fuori range vengono tagliati con `clip` tra 0 e 1.

### 5.3 Nodi sintetici

`NODE_SPECS` definisce i nodi simulati:

```python
{
    "name": "Nodo A - Fonte nazionale / INGV-DPC",
    "node_type": "national",
    "n_samples": 60000,
    "seed": 101,
    "source_weight": 1.80,
}
```

Ogni nodo ha:

- nome;
- tipo di fonte;
- numero di campioni;
- seed per riproducibilita';
- peso della fonte.

### 5.4 Peso della fonte

`source_weight` rappresenta un fattore di affidabilita' o rilevanza della
fonte. L'idea e' che una fonte istituzionale nazionale possa pesare piu' di una
segnalazione cittadina.

Nel codice attuale:

- fonte nazionale: 1.80;
- fonte regionale: 1.35;
- sensori: 1.20;
- Comune/COC: 1.00;
- cittadini: 0.55.

Questi valori sono configurabili e possono diventare oggetto di esperimenti di
sensibilita'.

La scala pratica consigliata e':

- `1.0`: peso neutro. Il nodo pesa solo in base al numero di campioni.
- valori tra `0.1` e `1.0`: fonte utile ma da ridimensionare per rumore,
  copertura parziale o minore verificabilita'.
- valori tra `1.0` e `2.0`: fonte ritenuta piu' affidabile, autorevole o
  operativamente rilevante.
- valori maggiori di `2.0`: possibili, ma da usare con cautela perche' possono
  dominare l'aggregazione e schiacciare il contributo degli altri nodi.

Non esiste un minimo/massimo universale dal punto di vista teorico, ma nel
paper conviene dichiarare un intervallo ragionevole e fare analisi di
sensibilita'. In questa simulazione si usa un intervallo moderato, da `0.55` a
`1.80`, proprio per evitare che un singolo nodo annulli gli altri.

Il peso fonte non deve essere presentato come una verita' oggettiva. E' un
prior operativo: incorpora nel modello una valutazione esterna sulla qualita',
autorevolezza, copertura o affidabilita' della fonte. Puo' essere attribuito in
diversi modi:

- da esperti di dominio, per esempio Protezione Civile o responsabili di sala
  operativa;
- da indicatori storici di qualita', come tasso di errori, ritardi o falsi
  allarmi della fonte;
- da audit istituzionali, per distinguere fonte certificata e fonte non
  verificata;
- da esperimenti di sensibilita', provando diverse scale e osservando
  stabilita' delle metriche.

Formula usata dall'orchestratore:

```text
effective_i = n_i * source_weight_i
alpha_i = effective_i / somma_j(effective_j)
```

Quindi il peso finale di aggregazione non dipende solo dal `source_weight`, ma
dal prodotto tra numerosita' del nodo e peso fonte.

### 5.5 Controllo del tasso di positivi sintetico

`SYNTHETIC_BASE_LOGIT` controlla quanto e' rara la classe positiva nei dati
sintetici:

```python
SYNTHETIC_BASE_LOGIT = -8.00
```

Nel generatore, il logit dell'evento parte da questa intercetta e poi viene
aumentato o ridotto dalle feature. Se il valore diventa piu' negativo, gli
eventi positivi diventano piu' rari; se diventa meno negativo, diventano piu'
frequenti.

Questo parametro e' importante perche' un vero evento emergenziale non dovrebbe
essere troppo frequente. Il valore attuale rende la classe positiva
relativamente rara nella simulazione, pur mantenendo abbastanza esempi positivi
per addestrare e valutare i modelli. Per un paper puoi eseguire scenari
diversi:

```text
scenario ordinario: eventi relativamente rari
scenario stress: eventi piu' frequenti
scenario sbilanciato: pochi positivi e molti negativi
```

Quando il positive rate e' basso, i modelli usano anche `class_weight =
"balanced"` nelle configurazioni. Questo rende l'esperimento piu' coerente con
un sistema di allerta: sbagliare la classe positiva rara deve pesare di piu'
durante l'apprendimento.

## 6. Dati sintetici

File: `federated/data.py`

La funzione principale per la simulazione e':

```python
create_federated_nodes()
```

Questa funzione legge `NODE_SPECS`, genera dati sintetici per ogni nodo e
divide ogni dataset locale in training set e test set.

### 6.1 FederatedNode

`FederatedNode` e' il contenitore di un nodo federato:

```python
FederatedNode(
    name,
    X_train,
    y_train,
    X_test,
    y_test,
    node_type,
    source_weight,
)
```

Contiene:

- `X_train`: feature locali per il training;
- `y_train`: etichette locali per il training;
- `X_test`: feature locali per la valutazione;
- `y_test`: etichette locali per la valutazione;
- `node_type`: tipo di fonte;
- `source_weight`: peso della fonte nell'aggregazione.

### 6.2 Generazione sintetica

La funzione:

```python
generate_node_data(node_type, n_samples, seed)
```

genera dati diversi per tipo di nodo.

Esempio concettuale:

- il nodo nazionale ha dati piu' affidabili e meno rumore;
- il nodo cittadino ha piu' rumore nelle etichette;
- il nodo sensori ha maggiore peso su anomalie strumentali;
- il nodo locale ha piu' segnalazioni operative.

Il target binario `y` rappresenta la presenza o assenza di una condizione di
allerta.

Il target viene generato tramite un logit:

```text
logit = intercetta + somma(feature * coefficiente) + interazioni + bias_nodo
probabilita = sigmoid(logit)
y ~ Bernoulli(probabilita)
```

Le interazioni modellano fenomeni realistici, per esempio:

- pioggia nelle 24 ore moltiplicata per saturazione del suolo;
- livello del fiume moltiplicato per variazione del livello.

## 7. Pipeline per dataset reali eterogenei

File: `federated/data.py`

Questa e' una parte importante per passare dai dati giocattolo ai dati reali.

Nella realta', fonti diverse possono avere colonne diverse:

- una Regione puo' chiamare la pioggia `prec_24h_mm`;
- un Comune puo' chiamarla `rain_day`;
- un sensore puo' chiamarla `rainfall24`;
- il target puo' chiamarsi `alert`, `alarm`, `evento`, ecc.

Il modello federato, pero', deve ricevere sempre le stesse feature canoniche.

Il concetto non riguarda solo le piogge. La pipeline e' generale: qualunque
fonte esterna puo' essere trasformata nello schema comune, purche' tu definisca
quali colonne locali corrispondono alle feature canoniche. Le feature canoniche
attuali rappresentano un caso idrogeologico/idraulico, ma il meccanismo puo'
essere esteso ad altri domini:

- dati sismici, per esempio magnitudo, profondita', distanza dall'epicentro;
- dati meteo, per esempio vento, temperatura, pressione, umidita';
- dati infrastrutturali, per esempio interruzioni elettriche, traffico,
  chiusure stradali;
- dati sociali/operativi, per esempio chiamate, ticket, segnalazioni app;
- dati territoriali, per esempio vulnerabilita', esposizione, classe di rischio.

Per aggiungere nuove variabili non devi riscrivere FedAvg. Devi aggiornare:

1. `FEATURES`, con i nomi canonici e l'ordine delle nuove colonne;
2. `FEATURE_RANGES`, con il range fisico o operativo per normalizzarle;
3. `feature_mapping`, per dire come ogni CSV reale mappa le proprie colonne
   verso quelle canoniche;
4. eventuale generatore sintetico, solo se vuoi simulare anche quelle nuove
   variabili.

Il punto metodologico e' questo: i dataset possono essere eterogenei in origine,
ma devono diventare omogenei prima del modello. L'eterogeneita' resta nei dati,
nelle distribuzioni e nella qualita' delle fonti; non deve restare nella forma
della matrice `X`, altrimenti i pesi dei modelli locali non sarebbero
aggregabili.

### 7.1 SourceDataSchema

`SourceDataSchema` descrive come leggere un CSV reale:

```python
SourceDataSchema(
    path="dati_regione.csv",
    name="Regione Campania",
    node_type="regional",
    source_weight=1.35,
    target_column="alarm",
    feature_mapping={
        "rainfall_24h": "prec_24h_mm",
        "event_reports": ["calls_112", "app_reports"],
        "sensor_anomaly": {"column": "anomaly_score", "scale": 0.01},
        "territorial_risk_class": 0.5,
    },
)
```

### 7.2 Regole di mapping

Il mapping supporta piu' forme.

Colonna singola:

```python
"rainfall_24h": "prec_24h_mm"
```

Media di piu' colonne:

```python
"event_reports": ["calls_112", "app_reports"]
```

Scala e offset:

```python
"sensor_anomaly": {"column": "anomaly_score", "scale": 0.01}
```

Valore costante:

```python
"territorial_risk_class": 0.5
```

Questa pipeline rende possibile usare fonti reali diverse senza cambiare la
struttura del modello.

### 7.3 Cosa succede se una fonte non ha una feature

Se una fonte non contiene una variabile, ci sono tre strategie pratiche:

- usare una costante, se il valore e' noto a livello territoriale;
- costruire una proxy da altre colonne;
- escludere temporaneamente quella fonte o quella feature, se la mancanza rende
  il dato troppo ambiguo.

Esempio con costante:

```python
"territorial_risk_class": 0.5
```

Esempio con proxy:

```python
"event_reports": ["calls_112", "app_reports", "municipal_tickets"]
```

In un paper questa parte va dichiarata perche' il preprocessing influenza il
significato scientifico del confronto.

## 8. Regressione logistica

File: `federated/model.py`

La regressione logistica e' implementata con NumPy. Il modello calcola:

```text
p(y = 1 | x) = sigmoid(theta_0 + theta_1*x_1 + ... + theta_n*x_n)
```

Dove:

- `theta_0` e' l'intercetta;
- `theta_i` sono i pesi associati alle feature;
- `sigmoid` trasforma il logit in probabilita'.

La loss e' la binary cross-entropy:

```text
L = - media[y*log(p) + (1-y)*log(1-p)]
```

Il training usa discesa del gradiente con regolarizzazione L2.

La regressione logistica e' utile nel paper perche' produce pesi interpretabili:

```text
rainfall_24h    peso positivo -> aumenta il rischio
road_closures   peso negativo -> riduce il rischio nel modello appreso
```

Attenzione: il segno dei pesi va interpretato rispetto alla distribuzione
sintetica e alle correlazioni tra feature.

### 8.1 Class weighting per eventi rari

Poiche' la simulazione ora rappresenta eventi positivi relativamente rari, la
configurazione usa:

```python
"class_weight": "balanced"
```

Senza questa correzione un classificatore puo' ottenere accuracy alta
predicendo sempre la classe negativa, cioe' "nessuna allerta". Per un sistema
di prevenzione questo sarebbe inaccettabile: l'accuracy sembrerebbe buona, ma
recall e F1 sarebbero nulli.

Il class weighting assegna piu' peso agli errori sulla classe rara. Nel caso
binario bilanciato:

```text
weight_0 = n_samples / (2 * n_negative)
weight_1 = n_samples / (2 * n_positive)
```

Se i positivi sono pochi, `weight_1` diventa maggiore di `weight_0`. In questo
modo il modello riceve un segnale piu' forte quando sbaglia un evento critico.
La valutazione sul test set resta invariata: i pesi di classe servono solo nel
training.

## 9. Rete neurale Keras/TensorFlow

File: `federated/model.py`

La rete neurale e' costruita con Keras/TensorFlow tramite:

```python
build_keras_neural_network(n_features, model_config)
```

La rete e' ora configurabile come un puzzle di blocchi. In `config.py` la chiave
`architecture` contiene una lista ordinata di pezzi:

```python
"architecture": [
    {"type": "dense", "units": 128, "l2": 0.0001},
    {"type": "batch_norm"},
    {"type": "activation", "name": "swish"},
    {"type": "dropout", "rate": 0.20},
    {"type": "dense", "units": 64, "l2": 0.0001},
    {"type": "batch_norm"},
    {"type": "activation", "name": "swish"},
    {"type": "dropout", "rate": 0.15},
    {"type": "dense", "units": 32, "l2": 0.0001},
    {"type": "batch_norm"},
    {"type": "activation", "name": "swish"},
    {"type": "dropout", "rate": 0.10},
    {"type": "output", "units": 1, "activation": "sigmoid"},
]
```

Questa scelta e' piu' adatta del genetic algorithm in questa fase. Un genetic
algorithm per cercare architetture sarebbe interessante, ma aumenterebbe molto
il costo computazionale e renderebbe piu' difficile isolare il contributo del
federated learning. La configurazione a blocchi, invece, permette di modificare
la rete in modo chiaro, riproducibile e documentabile.

Blocchi supportati:

- `dense`: layer completamente connesso, con numero di neuroni configurabile;
- `batch_norm`: Batch Normalization;
- `activation`: funzione di attivazione, per esempio `relu`, `swish`, `tanh`;
- `dropout`: dropout con tasso configurabile;
- `output`: layer finale, di solito `1` neurone con `sigmoid`.

Il modello produce una probabilita' binaria di allerta.

Questa architettura e' piu' vicina alle prassi moderne per dati tabellari e
problemi di rischio:

- layer densi progressivamente piu' piccoli, per comprimere rappresentazioni;
- Batch Normalization, per stabilizzare il training;
- attivazione Swish, spesso piu' morbida di ReLU in reti dense;
- Dropout, per ridurre overfitting;
- regolarizzazione L2 sui pesi;
- AdamW, quando disponibile, per combinare Adam e weight decay;
- class weighting bilanciato, per gestire eventi positivi rari;
- Early stopping nel training centralizzato, per fermarsi quando la validation
  loss non migliora.

Per esempio, per rendere la rete piu' piccola puoi usare:

```python
"architecture": [
    {"type": "dense", "units": 64, "l2": 0.0001},
    {"type": "batch_norm"},
    {"type": "activation", "name": "relu"},
    {"type": "dropout", "rate": 0.10},
    {"type": "output", "units": 1, "activation": "sigmoid"},
]
```

Per renderla piu' profonda puoi aggiungere altri blocchi `dense ->
batch_norm -> activation -> dropout`. L'importante e' che tutti i nodi usino
la stessa architettura, altrimenti i pesi non avrebbero la stessa forma e
l'orchestratore non potrebbe aggregarli.

Nel setting federato l'early stopping locale non viene usato per mantenere
round confrontabili tra nodi. Ogni nodo esegue lo stesso numero di epoche locali
e poi invia i pesi all'orchestratore.

### 9.1 Perche Keras/TensorFlow

Keras/TensorFlow e' un riferimento standard per reti neurali. Usarlo permette
di rendere il progetto piu' aderente alla pratica comune del machine learning.

### 9.2 Come funziona nel federato

Keras rappresenta il modello tramite una lista di array:

```python
weights = model.get_weights()
```

Con Batch Normalization la lista contiene non solo pesi e bias dei layer dense,
ma anche parametri e statistiche dei layer di normalizzazione:

```text
[W_dense, b_dense, gamma_bn, beta_bn, moving_mean_bn, moving_var_bn, ...]
```

Per questo motivo il codice salva i pesi Keras insieme a una piccola metadata
di architettura. In questo modo la predizione puo' ricostruire lo stesso modello
prima di chiamare `set_weights`.

Nel training federato:

1. l'orchestratore invia i pesi globali ai nodi;
2. ogni nodo crea il modello Keras e carica i pesi globali con `set_weights`;
3. ogni nodo addestra il modello sui propri dati locali;
4. ogni nodo restituisce i nuovi pesi locali;
5. l'orchestratore aggrega i pesi con FedAvg.

Questa e' la forma naturale di FedAvg per reti neurali.

### 9.3 Dipendenza TensorFlow

TensorFlow non e' installato nell'ambiente corrente. Il codice gestisce questo
caso senza interrompere l'esperimento:

```text
TensorFlow/Keras is not installed, so the neural-network experiment was not executed.
Install it with: python3 -m pip install -r requirements.txt
```

Per abilitare la rete neurale:

```bash
python3 -m pip install -r requirements.txt
python3 federated/main.py
```

## 10. Federated learning e FedAvg

File: `federated/federated.py`

Il training federato e' implementato da:

```python
federated_training(...)
```

### 10.1 FedAvg classico

Nel FedAvg classico, il peso del nodo e' proporzionale al numero di campioni:

```text
alpha_i = n_i / somma_j(n_j)
```

Il modello globale e':

```text
theta_global = somma_i(alpha_i * theta_i)
```

Dove:

- `theta_i` sono i parametri locali del nodo i;
- `n_i` e' il numero di campioni locali;
- `alpha_i` e' il peso del nodo nell'aggregazione.

### 10.2 FedAvg pesato per fonte

La variante implementata introduce un peso di fonte:

```text
effective_i = n_i * source_weight_i
alpha_i = effective_i / somma_j(effective_j)
```

Quindi:

```text
theta_global = somma_i(alpha_i * theta_i)
```

Interpretazione metodologica:

- `n_i` rappresenta la quantita' di dati;
- `source_weight_i` rappresenta affidabilita' o rilevanza della fonte;
- il prodotto unisce numerosita' e qualita'/autorevolezza.

Questo ha senso nel dominio Protezione Civile: una fonte istituzionale puo'
avere maggiore affidabilita' di una segnalazione cittadina, anche se entrambe
contribuiscono al sistema.

Esempio numerico semplice:

```text
Nodo A: 1000 campioni, source_weight = 1.80 -> effective = 1800
Nodo B: 1000 campioni, source_weight = 1.00 -> effective = 1000
Nodo C: 1000 campioni, source_weight = 0.50 -> effective = 500
totale effective = 3300
```

I pesi di aggregazione diventano:

```text
alpha_A = 1800 / 3300 = 0.545
alpha_B = 1000 / 3300 = 0.303
alpha_C =  500 / 3300 = 0.152
```

Anche se i tre nodi hanno lo stesso numero di campioni, l'orchestratore
attribuisce piu' influenza al nodo A perche' il suo peso fonte e' maggiore.

Nel paper questa scelta va presentata come variante metodologica. Non bisogna
dire che `1.80` sia "il vero peso" di INGV/DPC o che `0.55` sia "il vero peso"
dei cittadini. Bisogna dire che il codice permette di simulare e testare una
politica di aggregazione che incorpora affidabilita' o autorevolezza della
fonte. Per renderla piu' forte scientificamente, conviene riportare anche una
analisi di sensibilita':

```text
source_weight cittadini = 0.40, 0.55, 0.80, 1.00
source_weight nazionale = 1.20, 1.50, 1.80, 2.00
```

## 11. Centralized oracle

File: `federated/main.py`

Il modello centralizzato viene addestrato da:

```python
train_centralized_oracle(nodes, model_config)
```

Questa funzione unisce virtualmente i dati:

```python
X_train_centralized = np.vstack([node.X_train for node in nodes])
y_train_centralized = np.concatenate([node.y_train for node in nodes])
```

Questo benchmark e' chiamato "oracle" perche' vede tutti i dati di training.
Non rispetta il vincolo federato, ma serve come confronto ideale.

Nel paper e' utile per rispondere a:

> Quanto perde o guadagna il paradigma federato rispetto a un modello che vede
> tutti i dati?

## 12. Metriche

File: `federated/metrics.py`

Le metriche calcolate sono:

- accuracy;
- precision;
- recall;
- specificity;
- balanced accuracy;
- F1-score;
- false positive rate;
- false negative rate;
- ROC-AUC;
- PR-AUC;
- true positive;
- true negative;
- false positive;
- false negative.

Nel dominio allerta disastri, recall e F1 sono particolarmente importanti:

- `accuracy`: quota totale di classificazioni corrette. E' intuitiva, ma puo'
  essere ingannevole se gli eventi positivi sono rari.
- `precision`: tra gli allarmi emessi dal modello, quanti erano davvero eventi
  positivi. Alta precision significa meno falsi allarmi.
- `recall`, o sensibilita': tra gli eventi positivi reali, quanti sono stati
  intercettati. Alto recall significa meno eventi critici persi.
- `specificity`: tra i casi negativi reali, quanti sono stati riconosciuti come
  negativi. Alta specificity significa buona capacita' di non allertare quando
  non serve.
- `balanced accuracy`: media tra recall e specificity. E' utile quando le classi
  sono sbilanciate perche' pesa allo stesso modo classe positiva e negativa.
- `F1-score`: media armonica tra precision e recall. Cresce solo se entrambe
  sono ragionevolmente buone.
- `false positive rate`: quota di negativi scambiati per positivi. In pratica,
  probabilita' di falso allarme tra i casi non critici.
- `false negative rate`: quota di positivi scambiati per negativi. In pratica,
  probabilita' di mancata allerta tra i casi critici.
- `ROC-AUC`: misura quanto il modello separa positivi e negativi considerando
  tutte le possibili soglie. Vale 0.5 per un modello casuale e 1.0 per un
  ordinamento perfetto.
- `PR-AUC`: area sotto la curva precision-recall. E' particolarmente utile
  quando la classe positiva e' rara, perche' mette al centro il comportamento
  sugli eventi di interesse.

Conteggi della matrice di confusione:

- `TP`, true positives: eventi critici correttamente allertati;
- `TN`, true negatives: condizioni non critiche correttamente non allertate;
- `FP`, false positives: falsi allarmi;
- `FN`, false negatives: eventi critici mancati.

Nel dominio Protezione Civile, `FN` e `FP` non hanno lo stesso significato. Un
falso negativo puo' implicare mancata prevenzione; un falso positivo puo'
implicare attivazione inutile di risorse. Per questo il codice non salva solo
accuracy, ma anche metriche capaci di distinguere questi errori.

La soglia di classificazione e' configurata in `THRESHOLD`, attualmente:

```python
THRESHOLD = 0.5
```

Il codice implementa anche una ricerca soglia, salvata in:

```text
outputs/threshold_search_summary.csv
outputs/threshold_search_summary.json
```

Le strategie esportate sono:

- `best_f1`: soglia che massimizza F1;
- `best_balanced`: soglia che massimizza balanced accuracy;
- `best_operational`: soglia che massimizza precision mantenendo recall almeno
  pari a 0.90.

La soglia operativa e' importante per un sistema di early warning perche'
permette di imporre un requisito minimo di sensibilita' agli eventi critici.

Le AUC non dipendono da una soglia specifica. Per questo sono utili come
misure generali di ranking del modello. Precision, recall, specificity, FPR e
FNR invece dipendono dalla soglia scelta. Nel paper conviene riportare entrambe
le cose:

- metriche a soglia fissa `0.50`, per avere un confronto standard;
- metriche dopo threshold search, per valutare politiche operative.

## 13. Movimento dati e parametri

File: `federated/main.py`

Il progetto stima anche il numero di valori scalari spostati.

Nel centralizzato:

```text
moved_scalars = numero_campioni_training * (numero_feature + 1)
```

Il `+1` rappresenta la label.

Nel federato:

```text
moved_scalars = 2 * round * numero_nodi * numero_parametri
```

Il fattore 2 indica:

- invio modello globale dal server al nodo;
- ritorno modello locale dal nodo al server.

Questa e' una proxy semplificata, ma molto utile per il paper per mostrare il
trade-off:

- centralizzato: sposta dati;
- federato: sposta parametri.

## 14. Output sperimentali

File: `federated/artifacts.py`

A fine esecuzione vengono salvati:

```text
outputs/results_summary.csv
outputs/results_summary.json
outputs/federated_round_history.csv
outputs/federated_round_history.json
outputs/threshold_search_summary.csv
outputs/threshold_search_summary.json
outputs/node_non_iid_diagnostics.csv
outputs/node_non_iid_diagnostics.json
outputs/multiseed_results.csv
outputs/multiseed_results.json
outputs/multiseed_summary.csv
outputs/multiseed_summary.json
outputs/paper_ready_results.csv
outputs/paper_ready_results.json
outputs/plots/convergence_train_loss.svg
outputs/plots/convergence_train_f1.svg
outputs/plots/convergence_train_accuracy.svg
outputs/plots/convergence_train_precision.svg
outputs/plots/convergence_train_recall.svg
outputs/plots/final_f1_comparison.svg
outputs/plots/final_accuracy_comparison.svg
outputs/plots/training_time_comparison.svg
outputs/plots/data_movement_comparison.svg
```

### 14.1 results_summary

Contiene una riga per ogni esperimento:

- modello;
- paradigma;
- impostazione;
- metriche;
- tempo di training;
- tempo di valutazione;
- valori scalari movimentati.

Esempio di riga:

```text
Logistic Regression - Federated source-weighted FedAvg
```

Per la baseline local-only contiene una riga per ogni nodo, per esempio:

```text
Logistic Regression - Local-only trained on Nodo A - Fonte nazionale / INGV-DPC
Logistic Regression - Local-only trained on Nodo B - Regione / Centro Funzionale
...
```

Queste righe non vanno interpretate come modelli federati. Servono a mostrare
cosa succede quando ogni fonte lavora da sola.

### 14.2 federated_round_history

Contiene l'andamento round per round del training federato:

- round;
- train loss;
- train accuracy;
- train precision;
- train recall;
- train F1.

Serve per produrre grafici di convergenza.

### 14.3 threshold_search_summary

Contiene le soglie ottimali per ogni modello/paradigma secondo tre criteri:

- migliore F1;
- migliore balanced accuracy;
- migliore precision con vincolo `recall >= 0.90`.

Questa tabella e' utile per discutere la soglia non come costante arbitraria ma
come scelta sperimentale e operativa.

### 14.4 node_non_iid_diagnostics

Contiene diagnostica per ogni nodo:

- numero campioni;
- tasso di positivi train/test;
- source weight;
- peso FedAvg classico;
- peso FedAvg source-weighted;
- media e deviazione standard di ogni feature.

Serve a dimostrare che i nodi sono eterogenei e quindi che il caso federato non
e' una semplice partizione IID.

### 14.5 multiseed_results e multiseed_summary

`multiseed_results` contiene una riga per ogni seed, configurazione e policy di
soglia. Le policy sono:

```text
best_f1
best_balanced
best_operational
```

`multiseed_summary` aggrega i risultati con media e deviazione standard:

```text
mean F1 +/- std
mean recall +/- std
mean specificity +/- std
mean balanced accuracy +/- std
mean false positive rate +/- std
mean false negative rate +/- std
mean ROC-AUC +/- std
mean PR-AUC +/- std
```

Questa parte rende il benchmark piu' robusto per il paper, evitando di basare
la conclusione su un singolo run.

### 14.6 paper_ready_results

`paper_ready_results.csv` e `paper_ready_results.json` sono pensati come tabella
finale gia' vicina alla forma da usare nel paper. Ogni riga contiene:

- modello;
- paradigma;
- setting;
- policy di soglia;
- threshold;
- F1;
- recall;
- specificity;
- balanced accuracy;
- false positive rate;
- false negative rate;
- ROC-AUC;
- PR-AUC;
- moved scalars;
- train seconds.

Le policy includono:

- `fixed_threshold_0.50`, cioe' valutazione standard con soglia fissa;
- `best_f1`;
- `best_balanced`;
- `best_operational`.

Questa tabella e' quella piu' comoda per costruire la sezione risultati,
perche' tiene insieme performance predittiva, interpretazione operativa della
soglia e costo di movimento dati/parametri.

### 14.7 Grafici SVG

I grafici sono generati senza dipendenze esterne:

- `convergence_train_loss.svg`;
- `convergence_train_f1.svg`.
- `convergence_train_accuracy.svg`;
- `convergence_train_precision.svg`;
- `convergence_train_recall.svg`;
- `final_f1_comparison.svg`;
- `final_accuracy_comparison.svg`;
- `training_time_comparison.svg`;
- `data_movement_comparison.svg`.

Sono importabili in documenti Word, editor vettoriali o pipeline LaTeX.

Interpretazione consigliata:

- loss: mostra se il training federato converge;
- F1: mostra il bilanciamento tra precision e recall durante i round;
- accuracy: utile come indicatore generale, ma non sufficiente da sola;
- precision: utile per discutere falsi allarmi;
- recall: utile per discutere eventi reali non intercettati;
- final F1: confronto sintetico tra modelli/paradigmi;
- final accuracy: confronto generale tra modelli/paradigmi;
- training time: costo computazionale osservato nella simulazione;
- data movement: costo architetturale stimato in termini di valori spostati.

## 15. Esecuzione pratica

### 15.1 Esecuzione senza TensorFlow

```bash
python3 federated/main.py
```

In questo caso viene eseguita la regressione logistica. La rete neurale viene
saltata con un messaggio esplicativo se TensorFlow non e' installato.

### 15.2 Installazione TensorFlow

```bash
python3 -m pip install -r requirements.txt
```

Poi:

```bash
python3 federated/main.py
```

### 15.3 Verifica sintattica

```bash
python3 -m py_compile federated/*.py
```

## 16. Guida file per file

### 16.1 config.py

Contiene:

- `SEED`: seed globale;
- `FEATURES`: feature canoniche;
- `FEATURE_RANGES`: range per normalizzazione dati reali;
- `TARGET_COLUMN`: nome target di default;
- `NODE_SPECS`: configurazione dei nodi sintetici;
- `LOGISTIC_CONFIG`: iperparametri regressione logistica;
- `NEURAL_NETWORK_CONFIG`: iperparametri rete Keras;
- `class_weight`: gestione dello sbilanciamento delle classi;
- `THRESHOLD`: soglia classificazione;
- `OUTPUT_DIR`: cartella output.

### 16.2 data.py

Contiene:

- `FederatedNode`: struttura di un nodo federato;
- `SourceDataSchema`: schema per CSV reali eterogenei;
- `SharedFeatureProcessor`: mapping, normalizzazione e caricamento dati reali;
- `train_test_split_local`: split locale train/test;
- `generate_node_data`: dati sintetici;
- `create_federated_nodes`: nodi sintetici;
- `create_node_from_csv`: singolo nodo reale da CSV;
- `create_nodes_from_csv_schemas`: piu' nodi reali da schemi.

### 16.3 model.py

Contiene:

- funzioni matematiche della regressione logistica;
- training regressione logistica;
- costruzione rete Keras;
- training rete Keras;
- predizione per entrambi i modelli;
- copia, inizializzazione e media pesata dei parametri;
- conteggio dei parametri.

### 16.4 federated.py

Contiene:

- `compute_aggregation_weights`: calcola pesi di aggregazione;
- `fedavg_aggregate`: aggrega parametri locali;
- `federated_training`: esegue i round federati.

### 16.5 metrics.py

Contiene:

- `classification_metrics`;
- `roc_auc_score_binary`;
- `pr_auc_score_binary`;
- `threshold_search`;
- `collect_model_predictions`;
- `evaluate_model`.

### 16.6 reporting.py

Contiene funzioni di stampa:

- sintesi nodi;
- pesi modello logistico;
- pesi di aggregazione;
- tabella finale;
- interpretazione benchmark.

### 16.7 artifacts.py

Contiene:

- conversione tipi NumPy in tipi Python;
- export CSV;
- export JSON;
- tabella `paper_ready_results`;
- generazione grafici SVG.

### 16.8 main.py

E' il punto di ingresso:

```bash
python3 federated/main.py
```

Coordina l'intero esperimento.

## 17. Come leggere i risultati nel paper

Una possibile lettura scientifica e':

1. La regressione logistica fornisce un benchmark interpretabile.
2. Il modello centralizzato e' il riferimento ideale, ma richiede centralizzare
   i dati.
3. Il modello federato mantiene i dati nei nodi e sposta solo parametri.
4. La variante source-weighted introduce conoscenza di dominio nella fase di
   aggregazione.
5. La rete neurale Keras/TensorFlow permette di valutare un modello non lineare
   piu' moderno, con BatchNorm, Dropout, AdamW e regolarizzazione.
6. Il confronto va letto su due dimensioni:
   - performance predittiva;
   - costo di centralizzazione o movimentazione dati.

La logica di modelling completa e':

```text
dati grezzi di fonte
-> mapping verso feature canoniche
-> normalizzazione condivisa
-> split locale train/test
-> training locale o federato
-> aggregazione dei parametri
-> predizione probabilistica
-> scelta soglia
-> metriche finali
-> tabelle e grafici per il paper
```

Ogni modello restituisce probabilita', non direttamente decisioni finali. La
decisione `allerta/non allerta` nasce solo dopo aver applicato una soglia. Per
questo la threshold search e' parte del modelling, non un dettaglio secondario.

Nel centralizzato la fase di training vede tutti i dati di training insieme.
Nel federato ogni nodo vede solo i propri dati, aggiorna il modello e invia
parametri. Nel local-only ogni nodo produce un modello isolato. Questi tre
scenari rispondono a tre domande diverse:

- local-only: quanto puo' fare un ente da solo?
- federated: quanto si guadagna collaborando senza centralizzare i dati?
- centralized oracle: quale sarebbe il riferimento ideale se tutti i dati
  fossero disponibili in un unico posto?

## 18. Punti di forza

- Architettura modulare.
- Benchmark federato e centralizzato.
- Regressione logistica interpretabile.
- Rete neurale Keras/TensorFlow piu' vicina alle prassi moderne.
- Supporto per dati reali eterogenei.
- Output automatici per tabelle e grafici.
- Aggregazione pesata coerente con il dominio.

## 19. Limiti attuali

- I dati sintetici sono plausibili ma non reali.
- I pesi `source_weight` sono configurati manualmente.
- La stima `moved_scalars` e' una proxy semplificata.
- La simulazione federata gira in un unico processo, quindi non rappresenta
  latenza di rete o parallelismo reale.
- TensorFlow deve essere installato per eseguire la rete neurale.

## 20. Possibili estensioni per il paper

- Ampliare gli esperimenti multi-seed a piu' di tre seed.
- Riportare media e deviazione standard anche per la rete neurale, se il tempo
  computazionale lo consente.
- Fare analisi di sensibilita' sui `source_weight`.
- Usare dataset reali con `SourceDataSchema`.
- Confrontare FedAvg, source-weighted FedAvg e altre strategie robuste.
- Valutare scenari con positive rate piu' raro modificando
  `SYNTHETIC_BASE_LOGIT`.
- Studiare varianti federate specifiche per BatchNorm, per esempio FedBN.

## 21. Sintesi finale

Il codice implementa un esperimento completo per confrontare apprendimento
centralizzato e apprendimento federato in un contesto di monitoraggio disastri.

La struttura e' coerente con la logica del paper:

- scenario applicativo: Protezione Civile e fonti distribuite;
- problema: classificazione binaria di allerta;
- dati: sintetici ora, reali predisposti;
- modelli: regressione logistica e rete neurale Keras;
- paradigma: centralizzato vs federato;
- contributo specifico: aggregazione pesata per autorevolezza/rilevanza della
  fonte;
- output: tabelle, metriche, grafici e stime di movimento dati.

## 22. Notazione matematica completa

Questa sezione formalizza il problema in modo piu' adatto alla scrittura del
paper.

### 22.1 Dataset distribuito

Si consideri una rete federata composta da `K` nodi:

```text
N = {1, 2, ..., K}
```

Ogni nodo `i` possiede un dataset locale:

```text
D_i = {(x_i1, y_i1), ..., (x_in_i, y_in_i)}
```

dove:

- `x_ij` e' il vettore delle feature del campione `j` nel nodo `i`;
- `y_ij` e' la label binaria, 0 oppure 1;
- `n_i` e' il numero di campioni del nodo `i`.

Il dataset centralizzato equivalente sarebbe:

```text
D = unione_i(D_i)
```

Nel paradigma centralizzato, `D` viene materialmente o logicamente raccolto in
un unico server. Nel paradigma federato, `D_i` resta presso il nodo `i`.

### 22.2 Obiettivo predittivo

Il task e' una classificazione binaria:

```text
y in {0, 1}
```

Nel contesto applicativo:

- `y = 1`: condizione di allerta o rischio elevato;
- `y = 0`: assenza della condizione di allerta.

Il modello apprende una funzione:

```text
f_theta(x) -> p(y = 1 | x)
```

dove `theta` rappresenta i parametri del modello.

### 22.3 Loss globale centralizzata

Nel caso centralizzato si minimizza:

```text
L(theta) = (1 / N) * somma_i somma_j l(f_theta(x_ij), y_ij)
```

dove:

```text
N = somma_i n_i
```

e `l` e' la binary cross-entropy.

### 22.4 Loss federata implicita

Nel caso federato ogni nodo minimizza localmente:

```text
L_i(theta) = (1 / n_i) * somma_j l(f_theta(x_ij), y_ij)
```

L'orchestratore non vede i dati, ma combina i parametri locali.

Nel FedAvg classico l'ottimizzazione approssima:

```text
L_fed(theta) = somma_i (n_i / N) * L_i(theta)
```

Nella variante source-weighted:

```text
L_weighted(theta) = somma_i alpha_i * L_i(theta)
```

con:

```text
alpha_i = (n_i * s_i) / somma_j(n_j * s_j)
```

dove `s_i` e' il peso di fonte del nodo `i`.

### 22.5 Interpretazione del peso di fonte

Il peso `s_i` non rappresenta il numero di campioni. Rappresenta un fattore di
affidabilita', autorevolezza o rilevanza della fonte.

Nel dominio Protezione Civile:

- una fonte nazionale puo' essere piu' autorevole;
- un sensore puo' avere alta frequenza ma anche rumore strumentale;
- una segnalazione cittadina puo' essere preziosa, ma mediamente meno
  verificata;
- un Comune puo' avere conoscenza locale dettagliata ma meno copertura
  territoriale.

Questa modellazione e' utile per incorporare conoscenza di dominio
nell'orchestrazione federata.

## 23. Spiegazione approfondita dei dati sintetici

I dati sintetici non sono casuali in modo uniforme. Sono costruiti per
simulare differenze realistiche tra fonti.

### 23.1 Distribuzioni Beta

Molte feature sono generate con distribuzioni Beta:

```python
local_rng.beta(a, b, n_samples)
```

La distribuzione Beta e' adatta per valori normalizzati tra 0 e 1. Questo e'
coerente con il fatto che il modello riceve feature normalizzate.

Esempio:

```python
rainfall_24h = local_rng.beta(2.8, 3.2, n_samples)
```

Con parametri diversi si ottengono comportamenti diversi:

- valori piu' spesso bassi;
- valori piu' spesso intermedi;
- valori piu' spesso alti;
- distribuzioni piu' o meno disperse.

### 23.2 Bias di nodo

Ogni nodo ha un `node_bias`. Questo sposta la probabilita' di allerta del nodo.

Serve a simulare eterogeneita' non-IID:

```text
P_i(y | x) diversa da P_j(y | x)
```

In federated learning questo e' importante perche' i nodi non hanno dati
identicamente distribuiti.

### 23.3 Rumore di label

Il parametro `label_noise` ribalta una quota di label:

```python
y[noisy_idx] = 1 - y[noisy_idx]
```

Interpretazione:

- errori di misura;
- segnalazioni non verificate;
- dati storici incompleti;
- ritardi o discrepanze nella validazione dell'evento.

Il nodo cittadini ha rumore maggiore per simulare incertezza informativa.

### 23.4 Interazioni tra feature

Il logit include termini non lineari:

```text
rainfall_24h * soil_saturation
river_level * river_delta_1h
```

Queste interazioni sono rilevanti nel dominio:

- pioggia persistente e suolo saturo aumentano rischio idrogeologico;
- livello del fiume alto e crescita rapida aumentano rischio idraulico.

Questo spiega perche' una rete neurale puo' essere teoricamente utile: puo'
apprendere interazioni non lineari senza specificarle esplicitamente.

## 24. Walkthrough dettagliato del training federato

Questa sezione segue il codice di `federated/federated.py`.

### 24.1 Inizializzazione

Il training parte da:

```python
n_features = nodes[0].X_train.shape[1]
global_params = init_model_params(model_config, n_features)
```

Significato:

- si legge il numero di feature dal primo nodo;
- si crea un modello globale iniziale;
- per la logistica i parametri sono un vettore NumPy;
- per Keras i parametri sono i pesi del modello Keras.

### 24.2 Calcolo dei pesi di aggregazione

La funzione:

```python
compute_aggregation_weights(nodes, use_source_weights=True)
```

calcola `alpha_i`.

Caso classico:

```python
node_weight = 1.0
effective_size = len(node.y_train)
```

Caso source-weighted:

```python
node_weight = node.source_weight
effective_size = len(node.y_train) * node.source_weight
```

Poi:

```python
aggregation_weights = effective_sizes / total_size
```

Questi pesi sommano sempre a 1.

### 24.3 Round federato

Ogni round esegue:

```python
for node in nodes:
    params_local, _ = train_model(
        node.X_train,
        node.y_train,
        model_config,
        params_init=global_params,
        epochs=local_epochs,
    )
```

Significato:

1. il nodo riceve i parametri globali;
2. inizializza il modello locale con quei parametri;
3. fa training solo sui propri dati;
4. restituisce i nuovi parametri locali.

### 24.4 Aggregazione

Dopo il training locale:

```python
global_params = fedavg_aggregate(local_params, nodes, use_source_weights=True)
```

L'aggregazione e':

```text
global_params = somma_i(alpha_i * local_params_i)
```

Nel caso Keras, `local_params_i` e' una struttura con:

```python
{
    "weights": [...],
    "model_config": {...}
}
```

La funzione `weighted_average_params` sa aggregare:

- vettori NumPy;
- dizionari NumPy;
- liste di pesi Keras;
- strutture Keras con `weights`.

### 24.5 Valutazione interna al round

A ogni round il codice calcola metriche sul training globale virtuale:

```python
prob = predict_proba_model(node.X_train, global_params, model_config["kind"])
```

Questo non viola la simulazione federata? Nel mondo reale il server non dovrebbe
avere accesso ai dati dei nodi. Qui e' una simulazione single-process, quindi il
codice puo' calcolare metriche aggregate per scopi sperimentali.

Nel paper si puo' specificare:

> During simulation, global training metrics are computed by evaluating the
> global model over the union of local training partitions. This is used only
> for experimental monitoring and does not alter the federated training logic.

## 25. Walkthrough dettagliato della regressione logistica

La regressione logistica e' in `model.py`.

### 25.1 Intercetta

La funzione:

```python
add_intercept(X)
```

aggiunge una colonna di 1:

```text
X_aug = [1, X]
```

Questo permette di apprendere l'intercetta `theta_0`.

### 25.2 Predizione

La predizione e':

```python
sigmoid(X_aug.dot(theta))
```

Equivalente a:

```text
p = 1 / (1 + exp(-(theta_0 + theta_1*x_1 + ... + theta_n*x_n)))
```

### 25.3 Gradiente

Il gradiente medio e':

```python
grad = X_aug.T.dot(y_prob - y) / len(y)
```

Questa e' la forma vettoriale standard per la regressione logistica.

### 25.4 Regolarizzazione L2

Il codice aggiunge:

```python
reg = l2 * theta
reg[0] = 0.0
```

L'intercetta non viene regolarizzata. Questo e' comune perche' l'intercetta
non rappresenta una feature, ma uno shift globale del logit.

### 25.5 Aggiornamento

L'aggiornamento e':

```python
theta -= lr * (grad + reg)
```

Dove:

- `lr` e' learning rate;
- `grad` e' gradiente della loss;
- `reg` e' penalita' L2.

## 26. Walkthrough dettagliato della rete Keras moderna

La rete Keras e' costruita da:

```python
build_keras_neural_network(n_features, model_config)
```

### 26.1 Struttura a blocchi

La rete e' definita da una lista `architecture`. Ogni elemento della lista e'
un blocco. L'ordine dei blocchi e' l'ordine reale della rete.

Il pattern attuale e':

```text
Dense -> BatchNormalization -> Activation -> Dropout
```

Questo e' un pattern moderno per reti dense su dati tabellari.

Il vantaggio pratico e' che puoi modificare la rete senza toccare `model.py`.
Per esempio:

- aggiungere un layer significa aggiungere blocchi nella lista;
- cambiare attivazione significa modificare `{"type": "activation", "name": ...}`;
- cambiare dropout significa modificare `{"type": "dropout", "rate": ...}`;
- cambiare regolarizzazione significa modificare `l2` nel blocco `dense`.

Questa configurazione e' completamente deterministica e riproducibile: a parita'
di seed e architettura, l'esperimento resta confrontabile.

### 26.2 Dense

Il layer Dense apprende:

```text
h = W*x + b
```

Il numero di neuroni e' configurato in:

```python
hidden_layers = [128, 64, 32]
```

La struttura decrescente forza il modello a comprimere progressivamente
l'informazione.

### 26.3 Batch Normalization

BatchNorm normalizza le attivazioni interne:

```text
z_norm = (z - mean_batch) / sqrt(var_batch + eps)
```

Poi apprende parametri `gamma` e `beta`.

Vantaggi:

- training piu' stabile;
- minore sensibilita' all'inizializzazione;
- utile quando feature e nodi hanno distribuzioni diverse.

Nel federated learning con dati non-IID, BatchNorm va interpretata con cautela:
le statistiche locali possono differire tra nodi. In questa simulazione vengono
aggregate insieme ai pesi per mantenere la struttura semplice. In lavori futuri
si potrebbe valutare FedBN, dove le statistiche BatchNorm restano locali.

### 26.4 Swish

L'attivazione Swish e':

```text
swish(x) = x * sigmoid(x)
```

E' liscia e spesso funziona bene in reti dense moderne.

### 26.5 Dropout

Dropout spegne casualmente una quota di neuroni durante il training:

```python
dropout_rates = [0.20, 0.15, 0.10]
```

Nel modello:

- primo blocco: dropout 20%;
- secondo blocco: dropout 15%;
- terzo blocco: dropout 10%.

Questo riduce overfitting, specialmente nel centralizzato.

### 26.6 AdamW

AdamW separa il weight decay dal gradiente adattivo. E' spesso preferibile ad
Adam con L2 implicito.

Il codice usa:

```python
tf.keras.optimizers.AdamW(...)
```

se disponibile. Altrimenti usa Adam come fallback.

### 26.7 Early stopping

Nel centralizzato il codice puo' usare:

```python
EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
```

Questo evita di continuare il training quando il modello non migliora sulla
validation split.

Nel federato, invece, il codice disattiva early stopping locale quando
`params_init` non e' `None`, perche' ogni round deve essere comparabile tra
nodi e tutti i nodi devono completare lo stesso numero di epoche locali.

## 27. Come validare metodologicamente l'esperimento

Questa checklist serve per controllare che l'esperimento sia difendibile.

### 27.1 Validazione del setup

- I nodi hanno distribuzioni diverse?
- Le feature sono nello stesso ordine in tutti i nodi?
- Il test set e' separato dal training set?
- Il modello centralizzato usa solo training set per apprendere?
- La valutazione finale usa test set?

### 27.2 Validazione federata

- I nodi addestrano localmente?
- L'orchestratore riceve parametri e non dati?
- I pesi di aggregazione sommano a 1?
- FedAvg classico e source-weighted sono entrambi riportati?
- Il numero di round e local epochs e' documentato?

### 27.3 Validazione dei modelli

- La regressione logistica e' usata come benchmark?
- La rete neurale e' confrontata sia federata sia centralizzata?
- Gli iperparametri sono riportati?
- La rete neurale usa una architettura giustificabile?
- I risultati sono salvati in CSV/JSON?

### 27.4 Validazione delle metriche

- Accuracy da sola non basta.
- Precision e recall devono essere riportate.
- F1 e' utile come metrica sintetica.
- Nel dominio allerta e' importante discutere il trade-off falsi negativi/falsi
  positivi.
- Specificity serve a leggere la capacita' di non generare allarmi quando la
  condizione e' negativa.
- Balanced accuracy serve quando la classe positiva e quella negativa non hanno
  la stessa frequenza.
- ROC-AUC e PR-AUC servono a valutare il ranking probabilistico del modello
  indipendentemente da una singola soglia.

### 27.5 Validazione della narrativa scientifica

La tesi non deve essere:

> Il federato e' sempre migliore.

La tesi piu' corretta e':

> Il federato puo' ottenere prestazioni comparabili al centralizzato riducendo
> la movimentazione dei dati e rispettando vincoli organizzativi, privacy e
> governance distribuita.

### 27.6 Robustezza multi-seed

Il codice esegue una sintesi multi-seed per la regressione logistica usando i
seed configurati in `MULTI_SEED_VALUES`.

L'obiettivo non e' sostituire una validazione estesa, ma evitare di presentare
un solo run come se fosse definitivo.

Il multi-seed viene salvato per piu' policy:

```text
best_f1
best_balanced
best_operational
```

Nel paper si possono riportare:

```text
F1 medio +/- deviazione standard
recall media +/- deviazione standard
specificity media +/- deviazione standard
balanced accuracy media +/- deviazione standard
FPR medio +/- deviazione standard
FNR medio +/- deviazione standard
ROC-AUC medio +/- deviazione standard
PR-AUC medio +/- deviazione standard
```

Questi valori sono salvati in:

```text
outputs/multiseed_summary.csv
```

### 27.7 Diagnostica non-IID

Il file:

```text
outputs/node_non_iid_diagnostics.csv
```

serve a documentare che i nodi hanno distribuzioni diverse. Questa informazione
e' importante perche' il federated learning e' particolarmente rilevante quando
i dati sono distribuiti e non identicamente distribuiti.

## 28. Esempio completo con dataset reali eterogenei

Supponiamo di avere tre CSV reali:

```text
data/ingv.csv
data/regione.csv
data/comune.csv
```

Con colonne diverse.

### 28.1 Schema INGV/DPC

```python
{
    "path": "data/ingv.csv",
    "name": "INGV-DPC",
    "node_type": "national",
    "source_weight": 1.80,
    "target_column": "alert",
    "feature_mapping": {
        "rainfall_1h": "rain_1h_mm",
        "rainfall_3h": "rain_3h_mm",
        "rainfall_24h": "rain_24h_mm",
        "river_level": "river_m",
        "river_delta_1h": "river_delta_m",
        "soil_saturation": "soil_sat_pct",
        "event_reports": "official_reports",
        "road_closures": "closed_roads",
        "sensor_anomaly": {"column": "anomaly_pct", "scale": 0.01},
        "territorial_risk_class": "risk_class_norm",
    },
}
```

### 28.2 Schema regionale

```python
{
    "path": "data/regione.csv",
    "name": "Centro Funzionale Regionale",
    "node_type": "regional",
    "source_weight": 1.35,
    "target_column": "alarm",
    "feature_mapping": {
        "rainfall_1h": "prec_1h",
        "rainfall_3h": "prec_3h",
        "rainfall_24h": "prec_24h",
        "river_level": "idrometro_livello",
        "river_delta_1h": "idrometro_delta",
        "soil_saturation": "saturazione_suolo",
        "event_reports": ["sala_operativa", "segnalazioni_enti"],
        "road_closures": "strade_chiuse",
        "sensor_anomaly": "anomalia_sensori",
        "territorial_risk_class": "classe_rischio",
    },
}
```

### 28.3 Schema comunale/cittadini

```python
{
    "path": "data/comune.csv",
    "name": "Comune / COC",
    "node_type": "local",
    "source_weight": 1.00,
    "target_column": "evento",
    "feature_mapping": {
        "rainfall_1h": "pioggia_ora",
        "rainfall_3h": "pioggia_3ore",
        "rainfall_24h": "pioggia_giorno",
        "river_level": "livello_fiume",
        "river_delta_1h": "variazione_fiume",
        "soil_saturation": "suolo_saturo",
        "event_reports": ["telefonate", "app_cittadini"],
        "road_closures": "chiusure",
        "sensor_anomaly": {"constant": 0.0},
        "territorial_risk_class": 0.5,
    },
}
```

### 28.4 Creazione dei nodi reali

Il codice da usare sarebbe:

```python
from data import create_nodes_from_csv_schemas

schemas = [
    ingv_schema,
    regione_schema,
    comune_schema,
]

nodes = create_nodes_from_csv_schemas(schemas)
```

Dopo questa chiamata, `nodes` ha la stessa struttura dei nodi sintetici e puo'
essere passato direttamente a `run_experiment`.

## 29. Traccia per la sezione metodologia del paper

Questa traccia puo' essere trasformata direttamente in testo accademico.

### 29.1 Scenario

Il sistema considerato e' una rete multi-sorgente per monitoraggio e allerta.
Le fonti informative sono distribuite tra livelli istituzionali e operativi.
Questa struttura rende naturale l'adozione del federated learning, perche' i
dati possono essere sensibili, eterogenei e difficili da centralizzare.

### 29.2 Modelli

Sono stati confrontati due modelli:

1. regressione logistica, usata come benchmark interpretabile;
2. rete neurale Keras/TensorFlow, usata come modello non lineare.

Entrambi sono stati valutati in configurazione centralizzata e federata.

### 29.3 Federated aggregation

Oltre al FedAvg classico, e' stata introdotta una variante source-weighted:

```text
alpha_i = (n_i * s_i) / somma_j(n_j * s_j)
```

dove `s_i` rappresenta il peso della fonte.

Questa formulazione consente di integrare conoscenza di dominio nel processo di
aggregazione.

### 29.4 Valutazione

Le prestazioni sono valutate tramite:

- accuracy;
- precision;
- recall;
- specificity;
- balanced accuracy;
- F1-score;
- false positive rate;
- false negative rate;
- ROC-AUC;
- PR-AUC;
- matrice di confusione.

Inoltre si misura una proxy di data movement per confrontare il costo del
paradigma centralizzato rispetto a quello federato.

## 30. Traccia per la sezione risultati del paper

Nel paper, la tabella principale puo' essere costruita da:

```text
outputs/paper_ready_results.csv
```

Colonne consigliate:

- model_name;
- paradigm;
- setting;
- policy;
- threshold;
- f1;
- recall;
- specificity;
- false_positive_rate;
- false_negative_rate;
- roc_auc;
- pr_auc;
- moved_scalars.

`results_summary.csv` resta utile come tabella di controllo a soglia fissa
`0.50`, mentre `paper_ready_results.csv` include anche le policy di soglia.

I grafici di convergenza possono essere presi da:

```text
outputs/plots/convergence_train_loss.svg
outputs/plots/convergence_train_f1.svg
outputs/plots/convergence_train_accuracy.svg
outputs/plots/convergence_train_precision.svg
outputs/plots/convergence_train_recall.svg
```

I grafici comparativi finali possono essere presi da:

```text
outputs/plots/final_f1_comparison.svg
outputs/plots/final_accuracy_comparison.svg
outputs/plots/training_time_comparison.svg
outputs/plots/data_movement_comparison.svg
```

L'interpretazione dovrebbe evidenziare:

1. differenza tra centralizzato e federato;
2. effetto del source-weighted FedAvg;
3. confronto regressione logistica/rete neurale;
4. trade-off tra prestazioni e movimento dati;
5. limiti della simulazione.

## 31. Troubleshooting

### 31.1 TensorFlow non installato

Messaggio:

```text
TensorFlow/Keras is not installed
```

Soluzione:

```bash
python3 -m pip install -r requirements.txt
```

### 31.2 Errore tensorflow-metal su macOS

Se compare un errore su `libmetal_plugin.dylib` o
`_pywrap_tensorflow_internal.so`, di solito e' un mismatch tra TensorFlow e
tensorflow-metal.

Nel progetto i requirements sono pinnati:

```text
tensorflow==2.19.1
tensorflow-metal==1.2.0
```

Reinstallazione pulita:

```bash
conda activate fl-tf
python -m pip uninstall -y tensorflow tensorflow-metal keras tensorboard tensorflow-io-gcs-filesystem
python -m pip install --no-cache-dir -r requirements.txt
```

### 31.3 Python troppo nuovo

TensorFlow puo' non supportare versioni Python molto recenti. Per Apple Silicon
si consiglia un ambiente Conda con Python 3.11:

```bash
conda create -n fl-tf python=3.11 -y
conda activate fl-tf
python -m pip install -r requirements.txt
```

### 31.4 Output non aggiornati

Se i file in `outputs/` sembrano vecchi, rilanciare:

```bash
python3 federated/main.py
```

Il codice sovrascrive CSV, JSON e SVG.

## 32. Glossario essenziale

- Federated Learning: apprendimento distribuito in cui i dati restano locali.
- Nodo: entita' che possiede dati locali e addestra un modello locale.
- Orchestratore: server che coordina i round e aggrega i parametri.
- FedAvg: algoritmo che media i parametri locali pesandoli per numerosita'.
- Source-weighted FedAvg: FedAvg modificato con peso di fonte.
- Centralized oracle: modello che vede tutti i dati di training.
- Binary cross-entropy: loss per classificazione binaria probabilistica.
- Batch Normalization: normalizzazione delle attivazioni interne.
- Dropout: tecnica di regolarizzazione che spegne neuroni casualmente.
- AdamW: ottimizzatore Adam con weight decay separato.
- Non-IID: dati locali con distribuzioni diverse tra nodi.

## 33. Checklist finale prima di usare i risultati nel paper

Prima di inserire risultati nel paper:

- eseguire il codice in ambiente pulito;
- verificare che TensorFlow sia installato se si riportano risultati della rete;
- salvare versione Python e versione TensorFlow;
- riportare numero di nodi e campioni per nodo;
- riportare feature usate;
- riportare iperparametri principali;
- indicare se FedAvg e' classico o source-weighted;
- riportare metriche globali sul test set;
- includere grafico di convergenza;
- discutere limiti della simulazione single-process;
- chiarire che i dati sintetici sono una proxy sperimentale;
- se si usano dati reali, descrivere mapping e normalizzazione.
