# Image Classifier Pipeline (VOC) — What We Learned

## Status of the original files

`classifier.py` has both a working `SimpleClassifier` (a small 3-conv-layer
CNN, 21 output classes: 20 VOC objects + background) and an incomplete stub
`Classifier` class (`# TODO: implement me`), similar to the stubs found in
folder 1. `voc_dataloader.py` had a real compatibility bug: `box_indices` is a
ragged list (each image has a different number of bounding boxes), and current
numpy versions refuse to silently convert that into an array the way older
numpy did -- fixed with an explicit `dtype=object`. This only affects
bounding-box data, unused by the classification path, so the fix doesn't
change classification behavior.

**Data**: Pascal VOC 2007 (auto-downloaded via `torchvision.datasets.VOCDetection`,
~460MB, 5,011 images) -- excluded from git via `.gitignore` (`data/`), same as CIFAR-10.

## Multi-label vs. single-label classification

Every classification task so far (Iris, CIFAR-10) was single-label: exactly
one correct class per example. VOC is **multi-label** -- a photo can contain
both a person *and* a dog simultaneously, so each label is a set of 21
independent 0/1 flags, not one class index. This needs `BCEWithLogitsLoss`
(independent yes/no per class) instead of `CrossEntropyLoss` (which assumes
exactly one correct answer), and `sigmoid` + per-class thresholding instead
of `argmax` at prediction time.

## Experiment: SimpleClassifier on real VOC data (500 train / 200 val subset, 5 epochs)

```
Per-class accuracy (0.5 threshold): 0.9286
Example -- predicted classes: []
Example -- true classes: ['chair']
```

**92.86% sounds great, but the real example right below it tells the true
story: the model predicted no classes at all for an image that actually
contains a chair.** With 21 possible classes and most images containing only
1-3 real objects, ~18-20 of the 21 flags are "absent" for any given image --
a model can score very high accuracy just by learning "predict absent for
everything," since it's trivially correct about the easy negatives while the
rare positive classes that actually matter get drowned out in the average.
This is the same category of problem as the misleading
"accuracy-per-million-params" metric from the CNN zoo comparison -- a metric
that looks reassuring in aggregate while hiding the thing that actually
matters. It's also exactly why real multi-label/detection tasks use
**mean Average Precision (mAP)** instead of raw per-flag accuracy -- mAP
specifically scores performance on the positive class predictions, not
diluted by a flood of easy true-negatives. Our threshold-accuracy was a
reasonable stand-in for demonstrating the training mechanics, but this result
is a concrete demonstration of why it isn't the metric anyone would trust for
this task in practice.

## See also

`../object-detection-yolo/SUMMARY.md` for the classifier-vs-YOLO output
structure comparison (this classifier's flat 21-value output vs. YOLO's
14x14x30 grid).

## Files in this folder

- `classifier.py` -- `SimpleClassifier` (working) + `Classifier` (incomplete stub, original)
- `voc_dataloader.py` -- original VOC data loader, one numpy compatibility fix applied
- `test_classifier.py` -- our training/evaluation experiment, described above
- `MP3_P1A_Introduction.ipynb`, `MP3_P1B_Develop_Classifier.ipynb`, `kaggle_submission.py` -- original CS596 A3 coursework
