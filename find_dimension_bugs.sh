#!/bin/bash
# find_dimension_bugs.sh

echo "Cercando bug di dimensioni nei modelli..."

# Trova tutti i fit() con il vecchio pattern
echo "=== Potenziali bug in fit() methods ==="
grep -l "ML_CHECK_DIMENSIONS.*X.cols(), 1" src/models/logistic_regression.cpp

# Trova tutti i predict() senza feature check
echo -e "\n=== Predict() senza feature check ==="
for file in src/models/*.cpp; do
    if grep -q "VectorXd.*predict.*MatrixXd.*X.*const" "$file"; then
        if ! grep -q "ML_CHECK_FEATURE" "$file" && ! grep -q "FeatureMismatchException" "$file"; then
            echo "$file: predict() senza feature dimension check"
        fi
    fi
done

# Mostra le linee problematiche
echo -e "\n=== Linee da correggere ==="
grep -n "ML_CHECK_DIMENSIONS.*X.cols(), 1" src/models/*.cpp