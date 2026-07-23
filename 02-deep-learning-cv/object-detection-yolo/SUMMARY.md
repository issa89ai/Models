# YOLO Object Detection — What We Learned

## Why this wasn't fully trained

Fully training YOLO to convergence (correct bounding-box loss, non-max
suppression, mAP evaluation) is a much bigger undertaking than anything else
in this repo -- realistically hours, not minutes, even on a data subset.
Instead, we demonstrated the actual, concrete difference between
classification and detection using real code, real VOC data, and real
(untrained) model output shapes -- which directly answers "are these
comparable to the image classifier" without a multi-hour training run.

## Classification vs. detection: two fundamentally different questions

A classifier answers one question about the whole image: "which of these 21
classes are present?" A detector answers a much richer question per location:
"is there an object centered here, exactly where is its box, and what class
is it?"

`resnet_yolo.py`'s final layers make this concrete:
```python
self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)
...
x = torch.sigmoid(x)
x = x.permute(0, 2, 3, 1)
```
That `30` is not arbitrary: with 20 VOC classes (no background class here,
unlike the classifier's 21) and this YOLO variant predicting **2 bounding
boxes per grid cell** (`2 boxes x 5 values [x, y, w, h, confidence] = 10`,
plus `20 class probabilities` = 30 total), every grid cell makes its own
independent localized prediction.

## Experiment: real output shapes, same VOC image, both models

```
Image Classifier: input (3, 227, 227) -> output (1, 21)          [21 total values]
YOLO Detector:    input (3, 448, 448) -> output (1, 14, 14, 30)   [5,880 total values]
```

The classifier collapses an entire image into one flat 21-number answer.
YOLO divides the same-sized photo into a 14x14 grid (196 cells) and makes an
independent 30-value prediction *for each cell* -- a ~280x difference in
output size, from a similarly-sized input. This isn't "the same task done at
different accuracy levels" -- it's two structurally different kinds of
output entirely, which is the concrete, honest answer to whether
classification and detection results can be placed on the same
scale/metric: they can't.

## See also

`../image-classifier-pipeline/SUMMARY.md` for the classifier's own training
run (and a finding about misleading multi-label accuracy metrics that pairs
well with this comparison).

## Files in this folder

- `src/resnet_yolo.py` -- the ResNet50-based YOLO detector (original, untrained here)
- `src/config.py`, `src/dataset.py`, `src/eval_voc.py`, `src/predict.py` -- original CS596 final-exam utilities
- `yolo_loss.py` -- original YOLO loss function (bounding box + confidence + class loss), not exercised in this pass
- `test_output_shapes.py` -- our classifier-vs-YOLO output structure comparison, described above
- `MP3_P2.ipynb`, `kaggle_submission.py` -- original coursework files
