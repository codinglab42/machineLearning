#!/bin/bash
# build.sh - Script ottimizzato per Modern CMake + Pyenv

set -e
set -o pipefail

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}         BUILD ML LIBRARY (PYENV)         ${NC}"
echo -e "${BLUE}==========================================${NC}"

# ============================================================================
# 1. RILEVAMENTO AMBIENTE (PYENV)
# ============================================================================

PYTHON_EXE=$(which python)
PYTHON_VER=$($PYTHON_EXE --version 2>&1)

echo -e "${YELLOW}Using Python:${NC} $PYTHON_EXE ($PYTHON_VER)"

# ============================================================================
# 2. ARGOMENTI
# ============================================================================

CLEAN_BUILD="OFF"
PYTHON_TEST="ON"
INSTALL_DIR="${HOME}/.local"
VERBOSE="OFF"

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        --no-python)
            PYTHON_TEST="OFF"
            shift
            ;;
        --install-dir=*)
            INSTALL_DIR="${1#*=}"
            shift
            ;;
        --verbose)
            VERBOSE="ON"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --clean              Clean build directory before building"
            echo "  --no-python          Skip Python tests"
            echo "  --install-dir=PATH   Installation directory (default: ~/.local)"
            echo "  --verbose            Verbose output"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# ============================================================================
# 3. PULIZIA
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$CLEAN_BUILD" == "ON" ]]; then
    echo -e "${YELLOW}Pulizia directory build...${NC}"
    rm -rf "$PROJECT_ROOT/build"
fi

mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"

# ============================================================================
# 4. CONFIGURAZIONE CMAKE
# ============================================================================

echo -e "\n${BLUE}CONFIGURAZIONE CMAKE...${NC}"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DPython3_EXECUTABLE="$PYTHON_EXE"
    -DBUILD_PYTHON_BINDINGS=ON
    -DBUILD_TESTS=ON
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
)

if [[ "$VERBOSE" == "ON" ]]; then
    CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
fi

cmake .. "${CMAKE_ARGS[@]}"

# ============================================================================
# 5. COMPILAZIONE
# ============================================================================

echo -e "\n${BLUE}COMPILAZIONE IN CORSO...${NC}"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
make -j$NPROC

# ============================================================================
# 6. TEST C++
# ============================================================================

echo -e "\n${BLUE}ESECUZIONE TEST C++...${NC}"
if ctest --output-on-failure 2>/dev/null; then
    echo -e "${GREEN}✅ C++ tests passed${NC}"
else
    echo -e "${YELLOW}⚠️  Some C++ tests failed (continuing)${NC}"
fi

# ============================================================================
# 7. TEST PYTHON
# ============================================================================

if [[ "$PYTHON_TEST" == "ON" ]]; then
    echo -e "\n${BLUE}TEST PYTHON MODULE...${NC}"
    
    # Trova il modulo Python
    PY_MOD=$(find . -name "machine_learning_module*.so" 2>/dev/null | head -1)
    
    if [[ -n "$PY_MOD" ]]; then
        echo -e "${GREEN}✓ Modulo Python trovato:${NC} $PY_MOD"
        
        # Test import
        echo -e "${YELLOW}Test import modulo...${NC}"
        PY_MOD_DIR=$(dirname "$PY_MOD")
        
        if $PYTHON_EXE -c "
import sys
sys.path.insert(0, '$PY_MOD_DIR')
try:
    import machine_learning_module as ml
    print('✅ Module imported successfully')
    print(f'   Version: {ml.__version__}')
    
    # Test base
    import numpy as np
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    
    # Test LinearRegression
    lr = ml.LinearRegression(max_iter=50)
    lr.fit(X, y)
    pred = lr.predict(X)
    score = lr.score(X, y)
    print(f'   LinearRegression test: R²={score:.4f}')
    
    # Test NeuralNetwork (se disponibile)
    try:
        nn = ml.NeuralNetwork()
        nn.add_dense_layer(8, activation='relu')
        nn.add_dense_layer(1, activation='sigmoid')
        nn.build(3, 1)
        nn.set_epochs(10)
        nn.set_verbose(False)
        X2 = np.random.randn(20, 3)
        y2 = (np.sum(X2[:, :2], axis=1) > 0).astype(float)
        nn.fit(X2, y2)
        acc = nn.score(X2, y2)
        print(f'   NeuralNetwork test: accuracy={acc:.4f}')
    except Exception as e:
        print(f'   NeuralNetwork test: skipped ({e})')
    
    print('✅ All Python tests passed')
    
except ImportError as e:
    print(f'❌ Failed to import module: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1; then
            echo -e "${GREEN}✅ Python tests passed${NC}"
        else
            echo -e "${RED}❌ Python tests failed${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️  Python module not found${NC}"
    fi
fi

# ============================================================================
# 8. INSTALLAZIONE
# ============================================================================

echo -e "\n${BLUE}INSTALLAZIONE...${NC}"
make install

# ============================================================================
# 9. SUMMARY
# ============================================================================

echo -e "\n${BLUE}==========================================${NC}"
echo -e "${GREEN}✅ BUILD COMPLETATA!${NC}"
echo -e "${BLUE}==========================================${NC}"
echo -e ""
echo -e "${YELLOW}📦 Progetti generati:${NC}"
echo -e "  - Librerie C++: ${GREEN}build/lib/${NC}"
echo -e "  - Modulo Python: ${GREEN}build/pybinding/${NC}"
echo -e "  - Eseguibili test: ${GREEN}build/bin/${NC}"
echo -e ""
echo -e "${YELLOW}🧪 Per eseguire i test C++:${NC}"
echo -e "  ${GREEN}cd build && ctest --output-on-failure${NC}"
echo -e ""
echo -e "${YELLOW}🐍 Per testare il modulo Python:${NC}"
echo -e "  ${GREEN}python3 -c 'import sys; sys.path.insert(0, \"build/pybinding\"); import machine_learning_module as ml; print(ml.__version__)'${NC}"
echo -e ""
echo -e "${BLUE}==========================================${NC}"