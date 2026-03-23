#!/usr/bin/env python3
"""
Logistic Regression Example
===========================
Demonstrates binary classification using logistic regression.
"""

import machine_learning_module as ml
import numpy as np

def main():
    print("=" * 40)
    print("  LOGISTIC REGRESSION EXAMPLE")
    print("=" * 40)
    print()
    
    # 1. Generate synthetic data
    print("1. Generating synthetic classification data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Class 0: centered at (1,1)
    X0 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
    y0 = np.zeros(n_samples // 2)
    
    # Class 1: centered at (3,3)
    X1 = np.random.randn(n_samples // 2, 2) + np.array([3, 3])
    y1 = np.ones(n_samples // 2)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    print(f"   Generated {n_samples} samples with 2 features")
    print("   Class 0: centered at (1,1)")
    print("   Class 1: centered at (3,3)\n")
    
    # 2. Split data
    print("2. Splitting data (80% train, 20% test)...")
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}\n")
    
    # 3. Train model
    print("3. Training model...")
    model = ml.LogisticRegression(0.1, 1000, 0.001)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    print("\n4. Evaluation on test set:")
    accuracy = model.score(X_test, y_test)
    print(f"   Accuracy: {accuracy:.4f}")
    
    # 5. Confusion matrix
    print("\n5. Confusion Matrix:")
    cm = model.confusion_matrix(X_test, y_test)
    print("        Pred 0  Pred 1")
    print(f"  Act 0   {int(cm[0,0]):4d}   {int(cm[0,1]):4d}")
    print(f"  Act 1   {int(cm[1,0]):4d}   {int(cm[1,1]):4d}")
    
    # 6. Precision, Recall, F1
    print("\n6. Additional metrics:")
    metrics = model.precision_recall_f1(X_test, y_test)
    print(f"   Precision: {metrics[0]:.4f}")
    print(f"   Recall:    {metrics[1]:.4f}")
    print(f"   F1 Score:  {metrics[2]:.4f}")
    
    # 7. Coefficients
    print("\n7. Learned coefficients:")
    coef = model.coefficients
    intercept = model.intercept
    print(f"   Intercept: {intercept:.4f}")
    for i, c in enumerate(coef):
        print(f"   Coefficient x{i+1}: {c:.4f}")
    
    # 8. Probability predictions
    print("\n8. Probability predictions:")
    X_new = np.array([[2.0, 2.0]])
    proba = model.predict(X_new)
    print(f"   Input: x1=2.0, x2=2.0")
    print(f"   Probability of class 1: {proba[0]:.4f}")
    print(f"   Predicted class: {model.predict_class(X_new)[0]}")
    
    print("\n✅ Logistic Regression example completed!")

if __name__ == "__main__":
    main()