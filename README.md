> **October 2026 paper reproducibility notice.** The historical `federated/` scripts and root `outputs/` in this repository belong to an earlier simulation configuration and **do not reproduce the final FGCS manuscript metrics**. For the reconstructed and numerically validated logistic-regression experiments, frozen paper results, verification commands and scientific limitations, see **[paper_reconstruction/README.md](paper_reconstruction/README.md)**. The historical scripts are intentionally preserved for provenance.\n\n# Federated Learning per la Protezione Civile — Early Warning distribuito

Simulazione sperimentale di un sistema di **monitoraggio e allerta per la
prevenzione dei disastri**, costruita per confrontare un paradigma di
apprendimento **federato** con un paradigma **centralizzato**, in uno scenario
ispirato alla Protezione Civile italiana (fonti nazionali, regionali,
comunali, sensori, cittadini).

Il progetto è alla base delle sezioni *"The operational system"* e *"A
proof-of-concept implementation"* di un paper accademico in corso di
scrittura.

---

## 1. L'idea in una frase

> Le fonti informative su rischio idrogeologico sono per natura distribuite e
> spesso non centralizzabili (privacy, competenze, silos istituzionali). Il
> federated learning permette a ciascun ente di addestrare un modello sui
> propri dati **senza spostarli**, condividendo solo i parametri del modello
> con un orchestratore che li aggrega in un modello globale.

Il progetto verifica se, in questo scenario, l'approccio federato riesce ad
avvicinarsi alle prestazioni di un modello centralizzato che invece vede tutti
i dati, quantificando allo stesso tempo quanto dato/parametro viene
effettivamente "mosso" nei due paradigmi.

## 2. Lo scenario simulato

Cinque nodi eterogenei, ciascuno con distribuzione dati e affidabilità
diverse:

| Nodo | Tipo | Ruolo | Peso di fonte |
|---|---|---|---|
| A | `national` | INGV / Dipartimento Protezione Civile | 1.80 |
| B | `regional` | Centro Funzionale regionale | 1.35 |
| C | `sensors` | Rete di sensori/infrastrutture critiche | 1.20 |
| D | `local` | Comune / Centro Operativo Comunale (COC) | 1.00 |
| E | `citizens` | Segnalazioni cittadine | 0.55 |

Ogni nodo osserva le stesse 10 feature fisiche (pioggia a 1h/3h/24h, livello e
delta del fiume, saturazione del suolo, segnalazioni di eventi, chiusure
stradali, anomalie sensore, classe di rischio territoriale) e predice un
target binario `alert` (evento sì/no). Il **peso di fonte** (`source_weight`)
è un prior operativo — non una verità oggettiva — che incorpora quanto
l'orchestratore "si fida" istituzionalmente di una fonte nell'aggregazione.

## 3. Cosa viene confrontato

Due famiglie di modelli:

- **Regressione logistica** — benchmark interpretabile, pesi leggibili
  feature per feature.
- **Rete neurale (Keras/TensorFlow)** — modello non lineare più espressivo,
  usato per capire se la maggiore capacità del modello porta reale beneficio.

Per ciascun modello, quattro configurazioni:

1. **Local-only**: un modello per nodo, addestrato solo sui suoi dati e
   valutato sul test set globale (non si sceglie il "migliore" a posteriori:
   si conservano tutti i local model per un confronto onesto).
2. **FedAvg classico**: aggregazione pesata solo per numerosità campionaria.
3. **FedAvg source-weighted**: variante originale del progetto, che pesa
   l'aggregazione anche per l'affidabilità istituzionale della fonte:
   `α_i = (n_i · s_i) / Σ_j(n_j · s_j)`.
4. **Centralized oracle**: tutti i dati in un unico server — limite superiore
   di riferimento, non un "modello da battere" in senso assoluto.

## 4. Come si legge il risultato

La valutazione non si riduce a una singola metrica. Vengono guardati insieme:

- **prestazioni predittive** (F1, recall, specificity, balanced accuracy,
  ROC-AUC, PR-AUC, soglie ottimali);
- **robustezza su nodi non-IID** (le distribuzioni locali sono
  volutamente diverse tra loro, come nella realtà);
- **movimento di dati/parametri**, come proxy di costo/privacy tra
  paradigma centralizzato e federato;
- **utilità operativa** delle soglie di allerta in uno scenario di early
  warning (dove falsi negativi e falsi positivi hanno costi molto diversi).

### Un punto metodologico centrale: il tetto di Bayes

Il generatore di dati sintetici ha un **limite teorico di separabilità**
(Bayes-optimal ROC-AUC ≈ 0.67, con `SYNTHETIC_BASE_LOGIT = -8.0`). Significa
che *nessun* classificatore, per quanto perfetto, può superare stabilmente
quella soglia su questi dati: il rumore è strutturale, non un difetto del
modello. Un AUC osservato intorno a 0.65 va quindi letto come **vicino
all'ottimo**, non come prestazione debole. Questo tetto va sempre riportato
nel paper come linea di riferimento accanto ai risultati assoluti, altrimenti
il modello appare ingiustamente sottoperformante.

## 5. Struttura del codice

```text
federated/
  config.py       # feature canoniche, range fisici, spec dei nodi, iperparametri
  data.py         # generatore dati sintetici + caricamento dataset reali eterogenei
  model.py        # regressione logistica e rete neurale Keras
  federated.py    # FedAvg classico e source-weighted
  metrics.py      # metriche binarie, AUC, matrice di confusione, ricerca soglia
  reporting.py    # stampa riepiloghi leggibili a schermo
  artifacts.py    # esportazione CSV/JSON/SVG per il paper
  main.py         # orchestrazione dell'intero esperimento
outputs/          # risultati generati dall'ultima esecuzione (CSV + JSON)
docs/
  MANUALE_FEDERATED_LEARNING.md   # manuale teorico-pratico completo, sezione per sezione
  Federated_Learning_Data_Lifecycle_Protezione_Civile.docx
requirements.txt
```

Il flusso eseguito da `main.py`:

```text
1. Genera/carica i nodi federati
2. Stampa la sintesi dei nodi (numerosità, positive rate, pesi)
3. Benchmark regressione logistica: local-only → FedAvg → source-weighted → centralizzato
4. Benchmark rete neurale (se TensorFlow è installato)
5. Stampa pesi del modello federato pesato (interpretabilità)
6. Stampa tabella di confronto finale
7. Salva tutti gli artefatti (CSV/JSON/grafici) in outputs/
```

Per la spiegazione riga per riga di ogni modulo, le formule complete
(aggregazione, normalizzazione, generazione sintetica), i walkthrough passo
passo del training e la checklist pre-paper, vedere
**[`docs/MANUALE_FEDERATED_LEARNING.md`](docs/MANUALE_FEDERATED_LEARNING.md)**
— è il riferimento esaustivo, questo README è volutamente la sintesi.

## 6. Esecuzione

```bash
python3 -m pip install -r requirements.txt
python3 federated/main.py
```

Su Apple Silicon con TensorFlow, si consiglia un ambiente Conda dedicato
(Python 3.11) per evitare conflitti con `tensorflow-metal`; dettagli nel
manuale, sezione Troubleshooting.

## 7. Stato attuale e note metodologiche aperte

Questo repository riflette un lavoro in corso, non un risultato definitivo.
Al momento sono aperti alcuni punti che **vanno risolti prima di considerare
definitivi i numeri riportati nel paper**:

- gli artefatti in `outputs/` in questo commit possono essere stati generati
  con una configurazione precedente e vanno rigenerati dopo ogni modifica a
  `config.py` (in particolare a `SYNTHETIC_BASE_LOGIT`) prima di citarli;
- la selezione della soglia operativa va fatta su un validation set separato,
  non sul test set, per evitare ottimismo nella stima;
- manca un'analisi di sensibilità sistematica sui pesi di fonte
  (`source_weight`);
- la media dei parametri di Batch Normalization tra nodi non-IID in FedAvg è
  metodologicamente delicata e va discussa esplicitamente nei limiti;
- compatibilità NumPy 2.x da verificare per le funzioni di integrazione usate
  nel calcolo delle AUC.

Il flusso di lavoro adottato è deliberatamente sequenziale: **prima si
corregge e si convalida il codice, poi si rigenerano gli artefatti, poi si
scrivono i numeri nel paper** — per evitare di costruire il testo su risultati
stale o non riproducibili.

## 8. Limiti dichiarati della simulazione

- i dati sono sintetici (tranne quando si usa la pipeline CSV per dataset
  reali, già predisposta in `data.py`);
- l'esecuzione federata è single-process (simula i round, non una rete
  distribuita reale con comunicazione di rete);
- il tetto di Bayes ≈ 0.67 è una proprietà del generatore sintetico attuale,
  non un limite universale del dominio applicativo.

## 9. Licenza e citazione

Progetto di ricerca accademica. Aggiungere qui la licenza scelta (es. MIT per
il codice) e la citazione del paper una volta pubblicato.
