cd /home/devel/ai-lab/projects/ai-devel/machineLearning

# Genera struttura completa dei file
echo "=== STRUTTURA FILE ===" > code_analysis_input.txt
echo "" >> code_analysis_input.txt
find . -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" \) \
    ! -path "./build/*" ! -path "./.git/*" ! -path "./.pytest_cache/*" \
    ! -path "./__pycache__/*" | sort >> code_analysis_input.txt

echo "" >> code_analysis_input.txt
echo "=== CONTENUTO HEADER IMPORTANTI ===" >> code_analysis_input.txt
echo "" >> code_analysis_input.txt

# Header principali
for header in include/models/*.h include/components/layers/*.h include/components/loss/*.h; do
    if [ -f "$header" ]; then
        echo "" >> code_analysis_input.txt
        echo "--- $header ---" >> code_analysis_input.txt
        cat "$header" >> code_analysis_input.txt
    fi
done

echo "" >> code_analysis_input.txt
echo "=== CONTENUTO SORGENTI IMPORTANTI ===" >> code_analysis_input.txt
echo "" >> code_analysis_input.txt

# Sorgenti principali
for src in src/models/*.cpp src/components/layers/*.cpp; do
    if [ -f "$src" ]; then
        echo "" >> code_analysis_input.txt
        echo "--- $src ---" >> code_analysis_input.txt
        cat "$src" >> code_analysis_input.txt
    fi
done