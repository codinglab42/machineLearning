#!/usr/bin/env python3
"""
Neural Network XOR Example
==========================
Demonstrates solving the XOR problem with a neural network.
"""

import machine_learning_modules as ml
import numpy as np

def main():
    print("=" * 40)
    print("  NEURAL NETWORK - XOR PROBLEM")
    print("=" * 40)
    print()
    
    # 1. XOR dataset
    print("1. XOR dataset:")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.float64)
    
    print("   Input  Output")
    for i in range(4):
        print(f"   {X[i,0]} {X[i,1]}  ->  {int(y[i])}")
    print()
    
    # 2. Create network
    print("2. Creating neural network:")
    print("   Architecture: 2 -> 4 -> 1")
    print("   Hidden activation: ReLU")
    print("   Output activation: Sigmoid")
    print("   Optimizer: Adam, Learning rate: 0.1")
    print("   Loss function: Binary Cross-Entropy")
    print()
    
    network = ml.NeuralNetwork([2, 4, 1], "relu", "sigmoid",
                               ml.OptimizerType.ADAM, 0.1)
    network.set_loss_function("binary_crossentropy")
    network.set_epochs(2000)
    network.set_batch_size(4)
    network.set_verbose(False)
    
    # 3. Train
    print("3. Training...")
    network.fit(X, y)
    print("   Training completed!\n")
    
    # 4. Predictions
    print("4. Predictions:")
    y_pred = network.predict(X)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()
    
    print("   Input    Predicted    True    Class")
    print("   " + "-" * 40)
    for i in range(4):
        print(f"   [{X[i,0]},{X[i,1]}]   {y_pred[i]:.6f}   {int(y[i])}     {y_pred_class[i]}")
    
    # 5. Training history
    print("\n5. Training history:")
    loss, val_loss, acc = network.get_training_history()
    print(f"   Initial loss: {loss[0]:.6f}")
    print(f"   Final loss:   {loss[-1]:.6f}")
    print(f"   Loss reduction: {((loss[0] - loss[-1]) / loss[0] * 100):.1f}%")
    
    # 6. Network summary
    print("\n6. Network summary:")
    network.summary()
    
    print("\n✅ XOR problem solved!")

if __name__ == "__main__":
    main()