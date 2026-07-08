# Roadmap: Models Studied in the Master's

Full index of every model encountered in coursework, where the code lives now, and what's worth doing next to deepen each one. Status: `ported` = code copied as-is from coursework, `polished` = has its own full repo, `todo` = studied conceptually but not implemented standalone.

## 01 — Classical ML

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| KNN (loop + vectorized) | CS596 A1 | `01-classical-ml/knn/` | ported | Benchmark loop vs vectorized on a real dataset; add k-selection via cross-validation |
| Linear Classifier / Softmax / Perceptron / Logistic Regression | CS596 A1 | `01-classical-ml/linear-classifiers/` | ported | Consolidate into one sklearn-comparable benchmark script |
| SVM (from scratch + PyTorch) | CS596 A1 | `01-classical-ml/linear-classifiers/` | ported | Compare from-scratch gradient vs `sklearn.svm.SVC` on same data |
| Naive Bayes | CS566 (Pacman AI) | `01-classical-ml/naive-bayes-mira-pacman/` | ported | Re-run on a non-Pacman dataset (e.g. spam classification) to generalize the implementation |
| MIRA / Perceptron (Pacman) | CS566 | `01-classical-ml/naive-bayes-mira-pacman/` | ported | Document the margin-update rule; compare convergence vs plain perceptron |
| Decision Trees | CS505 A1 | `01-classical-ml/decision-trees/` | ported | Add pruning + compare gini vs entropy split criteria |
| Linear/LASSO/Logistic Regression | CS504 | `01-classical-ml/regression/` | ported | Cross-language comparison doc: same dataset in R vs Python vs PyTorch vs sklearn |
| SVM (theory only, CS596 lecture) | CS596 | — | todo | Kernel trick demo (RBF vs linear) on non-linearly separable toy data |
| Bayesian Decision Theory / MLE | CS509 lectures | — | todo | Small notebook deriving MLE for Gaussian, compare to sklearn estimate |
| HMM / Bayesian Networks | CS509 lectures | — | todo | Implement a simple HMM (Viterbi) — natural extension of the EM work already done |

## 02 — Deep Learning / Computer Vision

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| CNN architecture zoo (LeNet, VGG, ResNet, PreActResNet, ResNeXt, DenseNet, DPN, GoogLeNet, MobileNet/v2, ShuffleNet/v2, SENet, EfficientNet, PNASNet) | CS596 A2 | `02-deep-learning-cv/cnn-architecture-zoo/` | ported | Train 3-4 of these on CIFAR-10 with identical hyperparameters, compare accuracy/params/latency |
| Fully-connected NN (PyTorch) | CS596 A2 | `02-deep-learning-cv/cnn-architecture-zoo/` | ported | Baseline comparison against the CNNs above |
| Image classifier pipeline (VOC dataloader) | CS596 A3 | `02-deep-learning-cv/image-classifier-pipeline/` | ported | Swap in one CNN from the zoo above as backbone |
| YOLO object detection (ResNet-YOLO) | CS596 final exam / P3 Ayoub | `02-deep-learning-cv/object-detection-yolo/` | ported | Re-run eval_voc.py to get current mAP baseline; try a modern backbone |
| RNN (time series) | CS596 final project | `02-deep-learning-cv/rnn-time-series/`, also `cs596-deep-learning-rnn-project` | ported + polished | Already has a dedicated repo — use that for further iteration |
| GAN | CS596 (theory) | — | todo | Simple DCGAN on MNIST/CIFAR as a from-scratch addition |
| Dense prediction / segmentation | P3 Ayoub (slides) | — | todo | U-Net on a small segmentation dataset |
| Deep RL (MDP, Q-learning) | CS566 | — | todo | Q-learning on Pacman/Gridworld — natural extension of the CS566 AI course |

## 03 — Graph / Network Models

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| Girvan-Newman community detection | CS536 A1 | `03-graph-network/community-detection/` | ported | Compare against greedy-modularity on the same graph |
| Web mining project (PageRank-style) | CS536 | `03-graph-network/web-mining-project/`, also `cs536-web-mining-structure-analysis` | ported + polished | Use the polished repo for deeper analysis |
| Centrality measures | CS536 lectures | — | todo | Add betweenness/closeness centrality notebook |
| GNN (theory) | CS536 lectures | — | todo | Simple GCN node-classification demo on Karate Club or Cora |
| Graph traversal / MST / shortest path | CS571 | — | todo (proofs only) | CS571 was pure theory — could implement Kruskal/Dijkstra as a companion to the proofs |

## 04 — Probabilistic Models

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| EM algorithm — missing data imputation | CS509 final project | `04-probabilistic/missing-data-em/`, also `cs509-missing-data-imputation-em` | ported + polished | Use polished repo for further work |
| GMM estimation | CS509 PA A3 | `04-probabilistic/gmm-estimation/` | ported | Compare EM-based GMM fit vs k-means initialization |
| Hyperparameter tuning + dimensionality reduction (iris) | CS509 PR A4 | `04-probabilistic/hyperparameter-tuning-dimred/` | ported | Extend grid/randomized/halving search comparison to a larger dataset |
| HMM | CS509 lectures | — | todo | See classical ML section above |

## 05 — Unsupervised / Dimensionality Reduction

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| PCA (+ loading matrix) | CS504 | `05-unsupervised-dimred/pca-nmf/` | ported | Apply to a high-dimensional real dataset, visualize explained variance |
| NMF | CS504 | `05-unsupervised-dimred/pca-nmf/` | ported | Compare NMF vs PCA components for interpretability on same data |
| Topic Modeling (Python + R) | CS504 | `05-unsupervised-dimred/topic-modeling/` | ported | Try LDA vs NMF-based topic modeling on same text corpus |
| K-means / hierarchical clustering / DBSCAN | CS505 | — | todo | Referenced in data mining coursework — add standalone comparison notebook |
| SVD-based imputation | CS509 | — | todo | Compare against the EM-based imputation already ported |

## 06 — NLP / LLM Applications

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| PDF RAG chatbot (LLM API + retrieval) | P1 Rashid (CS590) | `06-nlp-llm-apps/pdf-rag-chatbot/` | ported | Compare against `doc-bot` (separate, more mature RAG project) — consider what doc-bot does better |
| Chatbot scenario design | P2 Yasir | (docs only, not code) | todo | Scenario doc exists in Master's folder but no implementation — could prototype it |
| Transformers / BERT (theory) | CS536 lectures | — | todo | Fine-tune a small BERT classifier as the natural "from theory to code" follow-up |

## Priority suggestions (highest learning value first)

1. **CNN zoo benchmark** — you already have 14 architectures implemented; running a real head-to-head comparison is the single highest-leverage next step.
2. **HMM implementation** — directly extends the EM work you already did for missing data; same probabilistic-modeling muscle.
3. **GCN/GNN demo** — you have graph theory (CS571) and web mining (CS536) covered conceptually; a working GNN closes the loop between classical graph algorithms and modern graph deep learning.
4. **GAN from scratch** — the one major deep generative model category not yet represented in code.
