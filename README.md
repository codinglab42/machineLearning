# C++ AI & Machine Learning Library

Una libreria di Machine Learning ad alte prestazioni scritta in **C++11/14**, basata sull'algebra lineare di **Eigen3** e integrata in **Python** tramite **pybind11**. 
Il progetto implementa gli algoritmi fondamentali trattati nei corsi di Andrew Ng, ottimizzati per l'efficienza computazionale.

## 🚀 Caratteristiche

* **Core in C++**: Implementazione nativa di Linear Regression, Logistic Regression e Neural Networks.
* **Algebra Lineare**: Utilizzo di **Eigen3** per operazioni matriciali veloci e vettorizzate.
* **Python Bindings**: Interfaccia Python fluida grazie a **pybind11**.
* **Preprocessing**: Scaler integrati (Standard, MinMax) basati su pattern Factory.
* **Robusta**: Gestione delle eccezioni custom per dimensioni non corrispondenti e problemi di fitting.

---

## 🛠 Struttura del Progetto

* `include/`: Header files (.h) organizzati per modelli, utility ed eccezioni.
* `src/`: Implementazione della logica C++.
* `pybinding/`: Codice sorgente per il bridging tra C++ e Python.
* `tests/`: Suite di test completa sia in C++ (GTest/Catch2) che in Python (pytest).

---

## 🏗 Requisiti

Prima di compilare, assicurati di avere:

* Un compilatore C++ (GCC >= 7 o Clang)
* **CMake** (>= 3.10)
* **Eigen3** libreria di algebra lineare
* **python3-dev** e **pybind11**

---

## ⚙️ Compilazione e Installazione

Il progetto include uno script `build.sh` per semplificare il processo:

```bash
# Rendi lo script eseguibile
chmod +x build.sh

# Compila il progetto
./build.sh

```

In alternativa, puoi usare la procedura standard CMake:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

```

Dopo la compilazione, il modulo Python (`machine_learning_module.cpython-...so`) sarà disponibile nella cartella `build/`.

---

## 🧪 Test

La libreria è validata attraverso test di integrazione e unitari.

### Test Python (Pytest)

```bash
pytest tests/python/unit

```

### Test C++

```bash
cd build
ctest --output-on-failure

```

---

## 💻 Esempio di Utilizzo in Python

Una volta compilato, puoi importare il modulo direttamente nel tuo script Python:

```python
import sys
sys.path.append('build/') # Punta alla cartella del modulo compilato
import machine_learning_module as ml

# Inizializza un modello di Regressione Lineare
model = ml.LinearRegression(learning_rate=0.01, iterations=1000)

# Training (utilizza matrici Eigen sotto il cofano)
model.fit(X_train, y_train)

# Predizione
predictions = model.predict(X_test)

```

---

## 📂 Organizzazione dei Modelli

| Modello                 | Header                   | Descrizione                                           |
| ---                     | ---                      | ---                                                   |
| **Linear Regression**   | `linear_regression.h`    | Modello di regressione classica per valori continui.  |
| **Logistic Regression** | `logistic_regression.h`  | Classificazione binaria con funzione sigmoide.        |
| **Neural Networks**     | `neural_network.h`       | Reti neurali multi-strato.                            |
| **Scalers**             | `scaler.h`               | Standardizzazione e normalizzazione dei dati.         |

---

## 📝 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per ulteriori dettagli.

---

### Prossimi Passi

* [ ] Implementazione di Regolarizzazione (L1/L2).
* [ ] Supporto per Multi-class Classification (Softmax).
* [ ] Ottimizzazione del caricamento dei dataset.

---
