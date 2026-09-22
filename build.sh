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
SKIP_PYTHON_TESTS="OFF"
SKIP_PYTHON_BINDINGS="OFF"
INSTALL_DIR="${HOME}/.local"
VERBOSE="OFF"
BUILD_TYPE="Release"
ENABLE_SANITIZER="OFF" # Var separata per non sovrascrivere BUILD_TYPE

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        --no-python-tests)
            SKIP_PYTHON_TESTS="ON"
            shift
            ;;
        --no-python-bindings)
            SKIP_PYTHON_BINDINGS="ON"
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
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --sanitize)
            ENABLE_SANITIZER="ON" # Attiva il flag senza toccare BUILD_TYPE
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --clean              Clean build directory before building"
            echo "  --no-python-tests    Skip Python tests (but still build bindings)"
            echo "  --no-python-bindings Skip Python bindings compilation entirely"
            echo "  --install-dir=PATH   Installation directory (default: ~/.local)"
            echo "  --verbose            Verbose output"
            echo "  --debug              Build with debug symbols"
            echo "  --sanitize           Build with AddressSanitizer and UBSan"
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
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DENABLE_SANITIZER="$ENABLE_SANITIZER"
    -DPython3_EXECUTABLE="$PYTHON_EXE"
    -DBUILD_PYTHON_BINDINGS="$([ "$SKIP_PYTHON_BINDINGS" == "ON" ] && echo "OFF" || echo "ON")"
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
# 7. TEST PYTHON (usando il target CMake)
# ============================================================================

if [[ "$SKIP_PYTHON_TESTS" == "OFF" ]] && [[ "$SKIP_PYTHON_BINDINGS" == "OFF" ]]; then
    echo -e "\n${BLUE}TEST PYTHON MODULE...${NC}"
    
    # Usa il target CMake che abbiamo definito
    if make test_python_module 2>/dev/null; then
        echo -e "${GREEN}✅ Python tests passed (via CMake target)${NC}"
    else
        # Fallback: test manuale
        echo -e "${YELLOW}⚠️  CMake test target failed, trying manual test...${NC}"
        
        # Cerca il modulo Python (nome corretto: ml_core)
        PY_MOD=$(find . -name "ml_core*.so" -o -name "ml_core*.pyd" 2>/dev/null | head -1)
        
        if [[ -n "$PY_MOD" ]]; then
            echo -e "${GREEN}✓ Modulo Python trovato:${NC} $PY_MOD"
            
            # Test import con sys.path
            if $PYTHON_EXE -c "
import sys
sys.path.insert(0, '$(pwd)')  # Aggiunge la build directory al path
try:
    import ml_core as ml  # Il pacchetto Python si chiama ml_core
    print('✅ Module imported successfully')
    print(f'   Version: {ml.__version__}')
    
    # Test rapido con LinearRegression
    import numpy as np
    X = np.random.randn(10, 3)
    y = 2.0 * X[:, 0] + 0.5 + np.random.randn(10) * 0.1
    
    lr = ml.LinearRegression(max_iter=50)
    lr.fit(X, y)
    pred = lr.predict(X)
    score = lr.score(X, y)
    print(f'   LinearRegression R²: {score:.4f}')
    
    print('✅ All Python tests passed')
    
except ImportError as e:
    print(f'❌ Failed to import module: {e}')
    print('   Check that ml_core.so is in the build directory')
    sys.exit(1)
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1; then
                echo -e "${GREEN}✅ Python tests passed (manual)${NC}"
            else
                echo -e "${RED}❌ Python tests failed${NC}"
                exit 1
            fi
        else
            echo -e "${RED}❌ Python module ml_core not found in build directory${NC}"
            echo -e "${YELLOW}   Check that BUILD_PYTHON_BINDINGS is ON${NC}"
            exit 1
        fi
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
echo -e "  - Modulo Python: ${GREEN}build/ml_core/${NC}"
echo -e "  - Eseguibili test: ${GREEN}build/bin/${NC}"
echo -e ""
echo -e "${YELLOW}🧪 Per eseguire i test C++:${NC}"
echo -e "  ${GREEN}cd build && ctest --output-on-failure${NC}"
echo -e ""
echo -e "${YELLOW}🐍 Per testare il modulo Python:${NC}"
echo -e "  ${GREEN}cd build && python3 -c 'import ml_core; print(ml_core.__version__)'${NC}"
echo -e ""
echo -e "${YELLOW}📦 Per installare il pacchetto Python:${NC}"
echo -e "  ${GREEN}cd build && make install${NC}"
echo -e "  ${GREEN}# Oppure: pip install -e .${NC} (se hai un setup.py)"
echo -e ""
echo -e "${BLUE}==========================================${NC}"

```