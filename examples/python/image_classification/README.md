# Image classification demo

This example classifies small synthetic grayscale images as containing a
vertical or horizontal bar. It uses an `ml_core.NeuralNetwork` multilayer
perceptron with binary cross-entropy. Each 10x10 image is flattened to a
100-element vector because the current Python model API accepts batches as
2D matrices (`samples x features`), not batches of 2D image tensors.

Images contain random bar positions, widths, and pixel noise. The generator is
seeded, and the data are split by class into training and test sets. No external
dataset or image-processing package is needed.

## Run

Use Python 3.11, NumPy, and the `ml_core` Python binding. From the repository
root, if the binding is available in `src/`:

```bash
PYTHONPATH=src python3.11 examples/python/image_classification/image_classification_demo.py
```

To change the generated dataset, random seed, or training duration:

```bash
PYTHONPATH=src python3.11 examples/python/image_classification/image_classification_demo.py --samples 600 --seed 7 --epochs 120
```

This is a small controlled classification task, not a benchmark for real
photographs. The reported metrics demonstrate the training and prediction
workflow only.
