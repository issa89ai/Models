# RNN Time Series (Stock Prediction) — What We Learned

## Status of the original file

`FinalProjectRNN.ipynb` is a complete, working project (not a stub) -- an
LSTM predicting next-day AMZN closing price from the previous 5 days,
using real data downloaded live via `yfinance`. Converted into a standalone
script (`test_rnn_stock.py`) with a shortened date range (2015-2024 instead
of 1997-2024) purely for faster download/training; same model and logic.

## Concept: RNN/LSTM -- sequential data

Every prior model treated input as a single snapshot (a flat feature vector,
or a whole image at once). Stock prices are a sequence where **order
matters** -- yesterday's price informs today's. An LSTM processes a sequence
step by step, carrying forward a hidden state (and a separate "cell state"
for longer-term memory) that it updates at each time step via internal
gates (forget/input/output) that control what to keep, discard, or pass
forward. This solves the same category of problem skip connections solved
for depth in ResNet, but along the time dimension instead.

The input shape is the key structural signature: `(batch, sequence_length=5,
features=1)` -- a genuine 3D tensor, not a flat vector like everything in
`01-classical-ml/`, and not a spatial grid like the CNN inputs.

## Experiment: LSTM on real AMZN data (2015-2024, 2,305 trading days)

```
Epoch 1/50:  train_loss=0.367276, val_loss=0.332468
Epoch 10/50: train_loss=0.001605, val_loss=0.002116
Epoch 50/50: train_loss=0.000732, val_loss=0.001106
```

Clean, fast convergence -- loss dropped ~300x by epoch 10 and kept improving
steadily, with train/val loss staying close throughout (no dramatic
overfitting gap like the CIFAR-10 FC net showed).

## The key finding: a naive baseline beats the LSTM

Before trusting that dropping-loss curve, we compared it against a **naive
"tomorrow's price = today's price" baseline** (zero learning, just copy the
most recent value):

```
Naive baseline ('no change') MSE: 0.000994
LSTM val_loss:                    0.001017
LSTM beats naive baseline by: -2.3%   (negative -- the LSTM is slightly WORSE)
```

**Despite an impressive-looking loss curve, the LSTM learned essentially
nothing beyond what a trivial copy-paste heuristic already captures** -- and
by this measure, ended up marginally worse. This is not a bug in the
implementation. Daily stock closing prices behave close to a "random walk":
today's price is already close to the best available predictor of
tomorrow's, and there's very little additional signal in the prior 5 days
beyond that single fact. This lines up with real quantitative finance theory
(the efficient market hypothesis, roughly: past price sequences don't
reliably predict future prices beyond the current price itself) -- a
genuine, honest result, not a failure of the code.

## Why this is the clearest example of a recurring theme in this repo

This is the starkest version yet of a pattern seen throughout: a model can
*look* like it's learning beautifully (dropping loss, small final numbers)
while contributing zero real predictive value. The same root issue showed up
as the CNN zoo's misleading "accuracy-per-million-params" ratio and the VOC
classifier's 92.86% accuracy that hid a complete miss on the one object that
mattered. The general lesson: **always compare a model's result against a
sensible, trivial baseline before trusting that a metric moving in the
"right" direction means real learning happened.**

## Files in this folder

- `FinalProjectRNN.ipynb` -- original CS596 final project notebook
- `test_rnn_stock.py` -- standalone script version, plus the naive-baseline comparison described above
