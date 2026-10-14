python3 -c "
import sys
import os
sys.path.insert(0, 'build/pybinding')
import numpy as np
import ml_core as ml

print('✅ Import OK')

# Test 1: StandardScaler
try:quit
    scaler = ml.StandardScaler()
    X = np.random.randn(10, 3)
    X_scaled = scaler.fit_transform(X)
    print('✅ StandardScaler OK')
except Exception as e:
    print(f'❌ StandardScaler: {e}')

# Test 2: NeuralNetwork construction
try:
    nn = ml.NeuralNetwork([10, 32, 16, 1], 'relu', 'sigmoid', ml.OptimizerType.ADAM, 0.001)
    print('✅ NeuralNetwork construction OK')
except Exception as e:
    print(f'❌ NeuralNetwork construction: {e}')

# Test 3: NeuralNetwork build
try:
    nn.build(10, 2)
    print('✅ NeuralNetwork build OK')
except Exception as e:
    print(f'❌ NeuralNetwork build: {e}')

# Test 4: NeuralNetwork fit (il più probabile candidato per il crash)
try:
    X = np.random.randn(100, 10)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(float)
    nn.set_loss_function('binary_crossentropy')
    nn.set_epochs(10)
    nn.set_verbose(False)
    nn.fit(X, y)
    print('✅ NeuralNetwork fit OK')
except Exception as e:
    print(f'❌ NeuralNetwork fit: {e}')
"