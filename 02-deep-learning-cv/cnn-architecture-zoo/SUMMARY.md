# CNN Architecture Zoo — What We Learned (Part 1: LeNet vs. Fully-Connected)

## Status of the original files

`models/lenet.py` (and the other 13 architectures in `models/`) are correctly
implemented PyTorch CNNs, ready to use as-is. `FullyConnectedNeuralNetworkPyTorch.py`
is broken: old Python-2-era code (`Variable`, `.data[0]`), a tabs/spaces
indentation error, a missing `import torchvision`, and a network built for
flattened 28×28 (MNIST-shaped) input while the surrounding code downloads
CIFAR-10 (32×32×3) — a genuine shape mismatch. We wrote a clean, working
fully-connected network from scratch (`test_cnn_vs_fc.py`) instead of fixing
that file, to directly compare against the real `LeNet` class.

**Environment note**: PyTorch/torchvision weren't installed on this machine,
and the default pip install failed with a Windows long-path error (the Python
install lives under a deeply nested Microsoft Store path). Rather than
changing system settings, we created a separate virtual environment at
`C:\ml_venv` with torch/torchvision/numpy/matplotlib/scikit-learn installed.
Any script in `02-deep-learning-cv/` needs to be run with
`C:\ml_venv\Scripts\python.exe`, not the regular `python` command.

## The concept: why convolution matters for images

Every model in `01-classical-ml/` either treated features as independent
numbers or split on them one at a time. A plain fully-connected network does
the same thing to an image: it flattens a 32×32×3 image into 3,072 independent
numbers, discarding the fact that neighboring pixels are spatially related.

A convolution layer learns a small filter (e.g. 5×5) that slides across the
whole image, computing the same local pattern-detector at every position.
This gives two properties a flattened FC layer can't: **parameter sharing**
(one edge-detector filter works anywhere in the image) and **locality** (each
output depends only on a small neighborhood of input pixels). Pooling
(`max_pool2d`) then shrinks the spatial size, keeping only the strongest
response in each region, making the network care less about a feature's exact
position.

## Experiment: LeNet (CNN) vs. plain Fully-Connected net, same CIFAR-10 data

5,000 training images, 1,000 test images (subset, for a fast CPU-friendly
demo), identical optimizer (Adam) and training loop for both.

**3 epochs:**
```
Fully Connected Net: test accuracy = 0.4260, parameters = 820,874
LeNet (CNN):         test accuracy = 0.4030, parameters = 62,006
```
FC net's loss had already plateaued by epoch 1-2; LeNet's loss was still
dropping steadily — it hadn't finished learning yet at only 3 epochs.

**10 epochs:**
```
Fully Connected Net: test accuracy = 0.4080, parameters = 820,874
LeNet (CNN):         test accuracy = 0.4890, parameters = 62,006
```

## The two key findings

1. **LeNet overtook the FC net once given enough training time** (0.403 ->
   0.489 from 3 to 10 epochs), confirming CNNs typically need more epochs to
   converge than a large FC net, but reward that patience with a real
   architectural advantage on image data.

2. **The FC net's test accuracy got *worse* with more training** (0.426 ->
   0.408 from 3 to 10 epochs) even as its training loss kept dropping —
   textbook overfitting, the same signature seen with Decision Trees at
   `max_depth=None` in folder 1. Its 820,874 parameters (13x more than
   LeNet's 62,006) let it increasingly memorize the small 5,000-image
   training set rather than learn generalizable patterns. LeNet's far smaller
   parameter count (via weight-sharing in the convolutions) made it much more
   resistant to this, on the exact same data and training budget.

**Overall conclusion**: convolution's advantage over flattening isn't just
about final accuracy — it's about achieving comparable or better performance
with ~13x fewer parameters, which directly translates into better resistance
to overfitting on limited data.

## What's not yet done

This is one architecture (LeNet) out of the 14 in `models/`, compared only
against a plain FC baseline. The original ROADMAP suggestion — training
3-4 of the CNN zoo architectures (e.g. ResNet, VGG, MobileNet) with identical
hyperparameters to compare accuracy/parameter-count/speed against each other,
not just against a non-convolutional baseline — is still open.

## Files in this folder

- `models/*.py` — the 14 original CNN architectures (LeNet, VGG, ResNet,
  PreActResNet, ResNeXt, DenseNet, DPN, GoogLeNet, MobileNet/v2,
  ShuffleNet/v2, SENet, EfficientNet, PNASNet), untouched
- `FullyConnectedNeuralNetworkPyTorch.py` — original file, broken (see above), reference only
- `test_cnn_vs_fc.py` — our LeNet vs. FC net comparison, described above
- `main.py`, `utils.py` — original CS596 A2 training utilities
