# Benchmark aggiornati - 25 novembre 2026

## Sintesi per slide

Tempi mediani di tre esecuzioni; fra parentesi, minimo e massimo.

| Scenario | Dataset / configurazione | Tempo | Qualità su test |
|---|---|---:|---:|
| Regressione lineare | 100.000 campioni, 10 feature | 1.252 ms (1.187–1.419) | R² 0,9988 |
| Regressione logistica | 100.000 campioni, 10 feature | 8.588 ms (7.161–8.678) | Accuracy 94,72% |
| Rete Dense | 10.000 campioni, 10 feature, 100 epoche | 1.305 ms (1.261–1.313) | Accuracy 87,5%; baseline 51,3% |
| SimpleRNN | 256 sequenze di training, 8 passi, 20 epoche | 26,5 ms (24,9–39,8) | MSE 0,0353 |
| LSTM | Stesso dataset e configurazione | 166,1 ms (139,6–169,7) | MSE 0,0125 |
| GRU | Stesso dataset e configurazione | 63,6 ms (54,1–113,4) | MSE 0,0100 |

Per le sequenze il test contiene 64 sequenze distinte e il predittore costante zero ha MSE 0,1521. Per i benchmark lineare, logistico e Dense, la dimensione indicata è quella totale: 80% training e 20% test. Le metriche di qualità sono calcolate su dati sintetici e non dimostrano prestazioni su dati reali.

Il checkpoint GRU con Adam e regolarizzazione L2 è stato serializzato **in memoria**, senza I/O su disco: mediana 0,035 ms in salvataggio e 16,61 ms in caricamento. Sono state verificate predizioni identiche e prosecuzione del training identica dopo il ripristino.

## Contesto e limiti

- Intel Core 5 120U, 1 CPU logica disponibile; GCC 14.4.1, build Release, OpenMP disabilitato. La revisione base è `493b43a`, con modifiche locali ai benchmark: vedere `metadata.json` per stato Git e hash degli eseguibili.
- Le tre esecuzioni usano gli stessi dataset e seed; misurano la variabilità temporale, non la variabilità statistica tra dataset o inizializzazioni. Il carico dell'ambiente condiviso può influire notevolmente sui risultati.
- I tempi si riferiscono al `fit` e includono l'inizializzazione della rete; escludono generazione dei dati e valutazione. Non riguardano i binding Python.
- Il dataset neurale precedente era sbilanciato; **i nuovi valori non sono direttamente confrontabili** con quelli della cartella precedente.
- Nel test Dense che varia il numero di feature, solo le prime quattro contribuiscono direttamente alla generazione dell'etichetta; le altre sono feature casuali aggiuntive.
- Le misure coprono i casi presenti nella suite, non tutte le funzionalità della libreria. Nessuna soglia di regressione temporale è stata applicata a questo run: costituisce una possibile baseline per confronti futuri in condizioni equivalenti.

## File per la presentazione

- `results.csv`: misure individuali e mediane, minimi e massimi.
- `results.json`: stessi dati strutturati, con metriche e baseline.
- `benchmark_*_results.png`: grafici per regressione lineare, logistica, rete Dense e sequenze.
- `*_run_*.txt`: output completo delle singole esecuzioni.
