# Benchmark performance - 25 novembre 2026

> **Attenzione:** questa è la suite precedente all'aggiornamento dei benchmark. I risultati sono conservati come riferimento storico, non sono una validazione completa della versione attuale e non sono direttamente confrontabili con la suite aggiornata. In particolare, il dataset neurale era fortemente sbilanciato: l'accuracy al 100% non dimostra capacità di classificazione.

## Sintesi per le slide

- **Regressione lineare:** su 500.000 campioni e 10 feature, training in **8.299 ms**, R² **0,9988**. A 10.000 campioni: **66 ms**. Con 500 feature: **7.256 ms**. Sul caso da 10.000 campioni/10 feature, Normal Equation e SVD richiedono **1 ms**, contro **20 ms** per Gradient Descent.
- **Regressione logistica:** su 100.000 campioni e 10 feature, training in **7.680 ms**, accuracy **94,72%**. Con 500 feature: **12.544 ms**, accuracy **93,15%**.
- **Rete neurale:** su 10.000 campioni e 10 feature, training in **1.128 ms**. La rete `10->256->1` (3.073 parametri) richiede **841 ms**; nel confronto ottimizzatori, Adam richiede **218 ms**. L'accuracy rilevata è 100%, ma non è rappresentativa per via dello sbilanciamento dei dati.

## Come leggere i risultati

I tempi sono quelli riportati dai benchmark durante il solo `fit`: generazione dei dati, train/test split e valutazione non sono inclusi. Ogni configurazione è stata eseguita una volta; i tempi interi in millisecondi non sono medie e possono variare con carico, macchina e compilatore. Le misure inferiori a 1 ms sono troncate e compaiono come `0 ms`. L'accuracy e l'R² sono metriche sui dati di test sintetici usati dai benchmark.

Le dimensioni indicate sono quelle del dataset totale: l'80% è usato per il training e il 20% per il test. La loss neurale riportata è la loss di training dell'ultima epoca, non la loss di test. Le cifre pubblicate nella sezione Performance del README non sono state aggiornate: senza ambiente e protocollo di misura equivalenti non è corretto trattare questo run come un confronto diretto con quei valori.

## Verifica di adeguatezza

- I tre eseguibili sono stati compilati e lanciati con la libreria alla revisione indicata; ciò dimostra compatibilità delle API utilizzate, non copertura dell'intera libreria.
- I benchmark neurali esercitano solo reti Dense e gli ottimizzatori SGD, Momentum e Adam. Non misurano le nuove sequenze ricorrenti, la regolarizzazione neurale, il salvataggio/ripristino dello stato o RMSprop e Adagrad.
- Il generatore neurale somma termini con contributo mediamente positivo e classifica rispetto a zero. Riproducendo il suo split da 5.000 campioni, con 10, 20, 50 o 100 feature tutti i 4.000 campioni di training e i 1.000 di test appartengono alla classe positiva; con 5 feature sono positivi 3.988/4.000 nel training e 995/1.000 nel test. Su tali dati una previsione sempre positiva raggiungerebbe rispettivamente il 100% e il 99,5% di accuracy sul test.
- Ogni misura è una sola esecuzione, senza soglie automatiche di regressione; Gradient Descent lineare può fermarsi prima del massimo di iterazioni e il benchmark non riporta quante ne abbia eseguite. Le misure di regressione logistica riportano tutte 1.000 iterazioni, il massimo configurato.
- All'epoca di questo run lo script di generazione grafici cercava eseguibili in `build/tests/performance/`, mentre CMake li configurava sotto `build/bin/` e i benchmark erano disabilitati; inoltre segnalava come completata la suite anche se non trovava i binari. I risultati qui conservati sono stati raccolti con esecuzione diretta e parsing verificato, non tramite quel flusso standard. Il flusso è stato successivamente aggiornato.
- Per risultati presentabili come verifica della versione attuale occorrono dataset bilanciati e metriche rispetto a una baseline, scenari per le funzionalità mancanti, ripetizioni e statistiche, oltre a un flusso di esecuzione affidabile.

## Esecuzione e ambiente

- Revisione: `493b43a`
- Compilatore: GCC 14.4.1
- Opzioni dei benchmark: `-std=c++17 -O3 -march=native`. Libreria statica preesistente aggiornata con CMake: `-std=gnu++17 -O3 -DNDEBUG`; OpenMP disabilitato.
- CPU: Intel Core 5 120U; **1 CPU logica disponibile** durante il run
- Le misure sono state raccolte il 25 novembre 2026.
- Tutti e tre gli eseguibili sono terminati con codice di uscita 0.
- I benchmark non facevano parte del run CTest standard al momento di questa esecuzione.

## File

- `slide_ready_results.csv`: 51 misurazioni parsate, pronte per tabelle e grafici.
- `benchmark_linear_results.png`, `benchmark_logistic_results.png`, `benchmark_neural_results.png`: grafici ad alta risoluzione.
- `benchmark_linear.txt`, `benchmark_logistic.txt`, `benchmark_neural.txt`: output integrale dei benchmark.
- `metadata.json`: revisione, ambiente e note sulle misurazioni.
