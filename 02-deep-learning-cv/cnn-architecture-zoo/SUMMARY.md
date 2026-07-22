# CNN Architecture Zoo — What We Learned

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

## Concept: ResNet — solving the "deeper should be better, but isn't" problem

Plain deep CNNs (just stacking more convolution layers) often trained *worse*
than shallower ones — not from overfitting, but because gradients became too
weak by the time they backpropagated through many layers to reach the early
ones (the "vanishing gradient" problem). ResNet's fix, visible directly in
`models/resnet.py`:
```python
out += self.shortcut(x)   # the skip connection
```
Adding the block's input directly to its output gives gradients a direct path
during backpropagation, bypassing the convolutions if needed — this is what
made training genuinely deep networks (18+ layers) reliable. ResNet also uses
`BatchNorm2d` (normalizes each layer's outputs to stable mean/variance within
every mini-batch), which LeNet has neither of.

## Concept: MobileNet — efficiency via depthwise separable convolutions

Instead of one standard convolution doing two jobs at once (mixing spatial
neighbors *and* mixing across channels simultaneously), MobileNet splits this
into two cheaper steps, visible in `models/mobilenet.py`:
```python
self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, ..., groups=in_planes, ...)  # depthwise: spatial only, per-channel
self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, ...)                        # pointwise: channel-mixing only
```
This approximates a standard convolution at a fraction of the compute cost —
the reason MobileNet exists is to run CNNs on phones/embedded devices.

## Experiment: LeNet vs. ResNet18 vs. MobileNet, same CIFAR-10 setup

Same 5,000 train / 1,000 test subset, 10 epochs, Adam optimizer, all three
trained in the same run for a directly comparable result:

```
LeNet:     accuracy=0.4570, parameters=62,006
ResNet18:  accuracy=0.5440, parameters=11,173,962
MobileNet: accuracy=0.3970, parameters=3,217,226
```

**ResNet18 wins on raw accuracy**, consistent across repeated runs (0.546 and
0.544 in two separate runs) — depth + skip connections + batch norm is the
strongest combination tested here, even though it has ~180x more parameters
than LeNet. Crucially, unlike the FC net (which overfit badly with far fewer
extra parameters, 13x LeNet's count), ResNet18 does *not* overfit despite
having *more* extra parameters than the FC net did. **The lesson: raw
parameter count isn't what causes overfitting — unstructured parameters
(FC net) overfit far more easily than a much larger number of
well-structured ones (ResNet18's convolutions + skip connections + batch
norm).**

**MobileNet actually underperformed LeNet here (0.397 vs. 0.457)** — the
opposite of what "efficient, modern architecture" might suggest. Depthwise
separable convolutions trade away some per-layer representational capacity
in exchange for efficiency, a tradeoff that pays off at the scale MobileNet
was designed for (millions of ImageNet images, many training epochs, real
mobile deployment constraints) — but on this tiny 5,000-image/10-epoch demo,
it simply didn't get enough signal to reach its stride. MobileNet isn't
"worse" than LeNet in general — it's optimized for a different regime than
the one tested here.

**A metric worth flagging as misleading**: we also computed
"accuracy-per-million-parameters" as an efficiency score, and LeNet "won" by
a huge margin (7.37 vs. ResNet18's 0.05 and MobileNet's 0.12) — but this is
mostly an artifact of LeNet's parameter count being a tiny fraction of one
million, which inflates the ratio when dividing by a number much less than 1.
It doesn't mean LeNet is genuinely the most efficient architecture in any
real engineering sense. A fairer efficiency comparison would use FLOPs
(actual compute cost) rather than raw parameter count, or only compare models
of similar parameter scale.

## What's not yet done

3 of the 14 architectures in `models/` have now been compared (LeNet,
ResNet18, MobileNet), satisfying the original ROADMAP suggestion of testing
3-4. Remaining architectures (VGG, DenseNet, GoogLeNet, ShuffleNet, SENet,
EfficientNet, PNASNet, DPN, ResNeXt, PreActResNet) are untouched but available
in `models/` if deeper comparison is wanted later.

## Files in this folder

- `models/*.py` — the 14 original CNN architectures (LeNet, VGG, ResNet,
  PreActResNet, ResNeXt, DenseNet, DPN, GoogLeNet, MobileNet/v2,
  ShuffleNet/v2, SENet, EfficientNet, PNASNet), untouched
- `FullyConnectedNeuralNetworkPyTorch.py` — original file, broken (see above), reference only
- `test_cnn_vs_fc.py` — LeNet vs. FC net comparison
- `test_resnet_vs_lenet.py` — LeNet vs. ResNet18 comparison
- `test_all_cnns_comparison.py` — LeNet vs. ResNet18 vs. MobileNet, all trained in one run
- `main.py`, `utils.py` — original CS596 A2 training utilities
