cd /home/devel/ai-lab/projects/ai-devel/machineLearning
# mkdir -p scripts

# Salva tutti gli script sopra e poi esegui:

echo "=== ANALISI COMPLETA CODICE ==="
echo ""

echo "1. Coerenza Header/Implementazione:"
python scripts/check_header_consistency.py

echo -e "\n2. Test Coverage:"
python scripts/check_test_coverage.py

echo -e "\n3. Include Mancanti:"
python scripts/find_missing_includes.py

echo -e "\n4. Codice Morto:"
python scripts/find_dead_code.py