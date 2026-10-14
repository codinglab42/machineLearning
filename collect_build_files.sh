#!/bin/bash
# collect_build_files.sh - Raccoglie tutti i file di build in un unico file di testo

set -e

# Colori
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "   COLLECT BUILD FILES"
echo -e "==========================================${NC}"

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

OUTPUT_FILE="build_files_analysis.txt"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# File da cercare (patterns)
PATTERNS=(
    "CMakeLists.txt"
    "*.cmake"
    "build.sh"
    "*.sh"
    "setup.py"
    "pyproject.toml"
    "requirements.txt"
    "Makefile"
    "Doxyfile"
)

# Directory da escludere
EXCLUDE_DIRS=(
    "build"
    "build-*"
    ".git"
    "venv"
    "env"
    "__pycache__"
    ".pytest_cache"
    ".cache"
    "*.egg-info"
    "dist"
    ".idea"
    ".vscode"
    "cmake-build-*"
)

# ============================================================================
# PULIZIA OUTPUT PRECEDENTE
# ============================================================================

if [[ -f "$OUTPUT_FILE" ]]; then
    echo -e "${YELLOW}⚠️  Sovrascrivo $OUTPUT_FILE${NC}"
fi

# ============================================================================
# INIZIO OUTPUT
# ============================================================================

{
    echo "================================================================"
    echo "                  BUILD FILES ANALYSIS                          "
    echo "================================================================"
    echo ""
    echo "Project Root: $PROJECT_ROOT"
    echo "Generated:   $(date)"
    echo ""
    echo "================================================================"
    echo ""
} > "$OUTPUT_FILE"

# ============================================================================
# FUNZIONE PER TROVARE E STAMPARE FILE
# ============================================================================

find_and_print_files() {
    local pattern="$1"
    local header="$2"
    
    echo "---------------------------------------------------------------" >> "$OUTPUT_FILE"
    echo "📁 ${header} (pattern: ${pattern})" >> "$OUTPUT_FILE"
    echo "---------------------------------------------------------------" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Costruisci comando find con esclusioni
    local find_cmd="find . -type f"
    
    for exclude in "${EXCLUDE_DIRS[@]}"; do
        find_cmd="$find_cmd -not -path \"*/$exclude/*\""
    done
    
    # Aggiungi pattern (con -name o -iname)
    find_cmd="$find_cmd \( -name \"$pattern\""
    find_cmd="$find_cmd -o -iname \"$pattern\" \)"
    
    # Esegui find e stampa
    eval "$find_cmd" | while read -r file; do
        # Rimuovi il "./" iniziale
        local rel_path="${file#./}"
        
        echo "================================================================" >> "$OUTPUT_FILE"
        echo "FILE: $rel_path" >> "$OUTPUT_FILE"
        echo "================================================================" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        
        # Stampa il contenuto del file con numeri di riga
        if [[ -f "$file" ]]; then
            nl -ba "$file" >> "$OUTPUT_FILE" 2>/dev/null || cat "$file" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        else
            echo "❌ File non trovato: $file" >> "$OUTPUT_FILE"
        fi
    done
    
    echo "" >> "$OUTPUT_FILE"
}

# ============================================================================
# CERCA E STAMPA I FILE
# ============================================================================

echo -e "${BLUE}🔍 Ricerca file di build...${NC}"

# 1. CMakeLists.txt (tutti)
echo -e "${YELLOW}  - CMakeLists.txt...${NC}"
find_and_print_files "CMakeLists.txt" "CMakeLists.txt files"

# 2. File .cmake (moduli CMake)
echo -e "${YELLOW}  - .cmake files...${NC}"
find_and_print_files "*.cmake" ".cmake module files"

# 3. Script shell
echo -e "${YELLOW}  - .sh scripts...${NC}"
find_and_print_files "*.sh" "Shell scripts"

# 4. Specifico build.sh
echo -e "${YELLOW}  - build.sh...${NC}"
if [[ -f "build.sh" ]]; then
    find_and_print_files "build.sh" "build.sh"
else
    echo "⚠️  build.sh non trovato nella root" >> "$OUTPUT_FILE"
fi

# 5. Altri file di configurazione
echo -e "${YELLOW}  - Altri file di configurazione...${NC}"
{
    echo "---------------------------------------------------------------" >> "$OUTPUT_FILE"
    echo "📁 Altri file di configurazione" >> "$OUTPUT_FILE"
    echo "---------------------------------------------------------------" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    for pattern in "${PATTERNS[@]}"; do
        # Salta quelli già cercati
        if [[ "$pattern" == "CMakeLists.txt" ]] || [[ "$pattern" == "*.cmake" ]] || [[ "$pattern" == "*.sh" ]] || [[ "$pattern" == "build.sh" ]]; then
            continue
        fi
        
        find . -type f -name "$pattern" -not -path "*/build/*" -not -path "*/.git/*" 2>/dev/null | while read -r file; do
            rel_path="${file#./}"
            echo "================================================================" >> "$OUTPUT_FILE"
            echo "FILE: $rel_path" >> "$OUTPUT_FILE"
            echo "================================================================" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            nl -ba "$file" >> "$OUTPUT_FILE" 2>/dev/null || cat "$file" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        done
    done
} >> "$OUTPUT_FILE"

# ============================================================================
# RIEPILOGO
# ============================================================================

# Conta i file trovati
TOTAL_FILES=$(grep -c "^FILE:" "$OUTPUT_FILE" 2>/dev/null || echo 0)

echo -e "${BLUE}=========================================="
echo -e "${GREEN}✅ COMPLETATO!${NC}"
echo -e "${BLUE}=========================================="
echo ""
echo "📄 Output file: $OUTPUT_FILE"
echo "📊 File trovati: $TOTAL_FILES"
echo ""
echo -e "${YELLOW}Per visualizzare il contenuto:${NC}"
echo "  less $OUTPUT_FILE"
echo ""
echo -e "${YELLOW}Per inviarmelo:${NC}"
echo "  cat $OUTPUT_FILE | pbcopy    # macOS"
echo "  cat $OUTPUT_FILE | xclip     # Linux (con xclip installato)"
echo "  cat $OUTPUT_FILE             # Linux (per copiare manualmente)"
echo ""
echo -e "${BLUE}=========================================="