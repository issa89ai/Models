# Decision Trees — What We Learned

## The mechanism

A decision tree predicts by asking a sequence of yes/no questions about the
features — a flowchart. Unlike KNN, there's real "training": `.fit()` actually
searches for and builds the tree of splits, rather than just storing data.

**Data types:** works natively on numeric and categorical features, needs no
feature scaling (unlike KNN, which is sensitive to feature scale since it's
distance-based). Works for both classification and regression.

## How a split is chosen (the brute-force search)

At every node, for **every feature**, the algorithm:
1. Sorts the values present in the data reaching that node.
2. Generates candidate thresholds — the midpoints between consecutive sorted values.
3. For each candidate threshold, splits the data into "≤ threshold" and "> threshold"
   groups and computes the **weighted average Gini impurity** of the two groups.
4. After testing every threshold of every feature, picks whichever single
   (feature, threshold) pair gave the **lowest** weighted Gini — i.e. the split that
   most increases purity.
5. Repeats the entire process independently on each child node, using only the
   data that landed there, until a stopping rule is hit (`max_depth`, or a node
   is already pure).

**Gini impurity**: `Gini = 1 - Σ(p_i²)`, where `p_i` is each class's fraction in a
group. `Gini = 0` = perfectly pure (one class only, stop splitting). Maximum Gini
is `1 - 1/C` for `C` classes (e.g. 0.5 for 2 classes, 0.667 for 3) — reached when
classes are perfectly evenly mixed. Gini can never actually reach 1. Lower is
always better/purer; the tree's whole goal at every split is to **minimize** it.

**Crucially, the algorithm has zero domain knowledge.** It doesn't know "petal
length" is a flower measurement — it just sees a numeric column and mechanically
runs the identical brute-force search regardless of what the data represents
(flowers, loan applications, housing prices, anything). The column names only
matter for us reading the tree diagram afterward.

## max_depth = the bias-variance dial

Same role as KNN's `k`, opposite direction: **small max_depth → underfitting**
(too simple to capture real structure), **max_depth=None → the tree keeps
splitting until every leaf is pure**, memorizing the training set. That
memorization always shows up as train accuracy hitting 100% — whether it also
hurts *test* accuracy depends on the dataset (see results below).

## Experiment: Iris dataset, max_depth = 1, 2, 3, 4, 5, None

```
max_depth=1:    train=0.6750, test=0.6333
max_depth=2:    train=0.9500, test=0.9667
max_depth=3:    train=0.9583, test=1.0000
max_depth=4:    train=0.9750, test=1.0000
max_depth=5:    train=0.9917, test=1.0000
max_depth=None: train=1.0000, test=1.0000
```

- **depth=1**: bad on *both* train and test (~65%) — classic underfitting signature.
  One split can only isolate one class (setosa) from the rest; no budget left to
  separate the other two classes.
- **depth=2**: big jump to ~0.95-0.97 — a second question is enough to mostly
  separate the harder two classes too.
- **depth=None**: train accuracy hits exactly 1.0 — full memorization. Test
  accuracy also stayed perfect here, but that's a property of Iris being small
  and clean, not a general guarantee — on noisier/larger data, the same
  max_depth=None would show train=100% with visibly *worse* test accuracy. The
  original coursework notebook's `add_noise()` experiment demonstrates this
  directly by corrupting labels and rerunning.

## Visualizations

- **Accuracy vs. depth curve** (`accuracy_vs_depth.png`) — plots the table above;
  the underfitting flat-low region (depth=1) and the convergence toward 1.0 are
  directly visible as two lines closing the gap between them.
- **Tree structure diagram** (`tree_structure.png`) — the actual learned
  flowchart at max_depth=3. Root splits on `petal length ≤ 2.45` (perfectly
  isolates all 40 setosa training samples into a pure leaf, Gini=0). Every split
  in the whole tree uses petal length/width — sepal features were never selected,
  since they never reduced impurity enough to win the search. One leaf remains
  impure even at depth=3 (`Gini=0.5, samples=8, value=[0,4,4]`) — 4 versicolor
  and 4 virginica flowers indistinguishable by petal measurements at this depth.
- **Decision boundary** (`decision_boundary.png`) — same information spatially,
  using only petal length/width. Boundaries are axis-aligned rectangles (trees
  only ever split perpendicular to one feature at a time — never diagonal or
  curved). The teal/yellow border sits right at the tree's `petal width ≤ 1.75`
  split, and the handful of misclassified-looking points near that border are
  the same ambiguous 8 samples identified in the tree diagram.

## Files in this folder

- `Assignment_1_DecisionTrees.ipynb` — original CS505 coursework notebook (full
  worked assignment, including a noise-robustness experiment not repeated here)
- `test_tree.py` — our experiment script (Iris, varying `max_depth`, train vs test)
- `visualize_tree.py` — generates the three plots discussed above
- `accuracy_vs_depth.png`, `tree_structure.png`, `decision_boundary.png` — saved plots
