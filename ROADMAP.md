# Roadmap: Models Studied in the Master's

Full index of every model encountered in coursework, where the code lives now, and what's worth doing next to deepen each one. Status: `ported` = code copied as-is from coursework, untouched; `in progress` = partially worked through hands-on, more remains; `completed` = worked through hands-on (run, visualized, and written up in that folder's `SUMMARY.md`); `polished` = has its own full repo; `todo` = studied conceptually but not implemented standalone.

## 01 — Classical ML

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| KNN (loop + vectorized) | CS596 A1 | `01-classical-ml/knn/` | completed | Try on a larger/higher-dimensional dataset to see how the best k scales with data size in practice |
| Perceptron / SVM / Softmax / Logistic Regression | CS596 A1 | `01-classical-ml/linear-classifiers/` | completed | Try a genuinely non-linearly-separable 2D toy dataset to visualize exactly where a straight-line boundary fails |
| Naive Bayes | CS566 (Pacman AI) | `01-classical-ml/naive-bayes/` | completed | Try on a text dataset (e.g. spam classification), where the independence assumption is a more natural fit than on correlated numeric features |
| Decision Trees | CS505 A1 | `01-classical-ml/decision-trees/` | completed | Add pruning + compare gini vs entropy split criteria |
| Linear/LASSO Regression | CS504 | `01-classical-ml/regression/` | completed | Cross-language comparison: same dataset in R vs Python vs PyTorch vs sklearn |
| SVM (theory only, CS596 lecture) | CS596 | — | todo | Kernel trick demo (RBF vs linear) on non-linearly separable toy data |
| Bayesian Decision Theory / MLE | CS509 lectures | — | todo | Small notebook deriving MLE for Gaussian, compare to sklearn estimate |
| HMM / Bayesian Networks | CS509 lectures | — | todo | Implement a simple HMM (Viterbi) — natural extension of the EM work already done |

## 02 — Deep Learning / Computer Vision

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| CNN architecture zoo (LeNet, VGG, ResNet, PreActResNet, ResNeXt, DenseNet, DPN, GoogLeNet, MobileNet/v2, ShuffleNet/v2, SENet, EfficientNet, PNASNet) | CS596 A2 | `02-deep-learning-cv/cnn-architecture-zoo/` | completed | LeNet, ResNet18, and MobileNet compared (see SUMMARY.md) -- ResNet18 wins on accuracy despite far more parameters (structure > raw count), MobileNet underperforms at this tiny scale (needs more data/epochs to show its efficiency advantage). Remaining 11 architectures (VGG, DenseNet, etc.) untouched if deeper comparison wanted later |
| Fully-connected NN (PyTorch) | CS596 A2 | `02-deep-learning-cv/cnn-architecture-zoo/` | completed | Original file was broken (Python 2 code, shape mismatch); rewrote from scratch and compared against LeNet -- FC net overfits faster despite 13x more parameters |
| Image classifier pipeline (VOC dataloader) | CS596 A3 | `02-deep-learning-cv/image-classifier-pipeline/` | completed | Real mAP evaluation instead of the simplified per-flag accuracy used here (see SUMMARY.md for why raw accuracy is misleading on this multi-label task) |
| YOLO object detection (ResNet-YOLO) | CS596 final exam / P3 Ayoub | `02-deep-learning-cv/object-detection-yolo/` | in progress | Output structure confirmed (see SUMMARY.md); full training + eval_voc.py mAP baseline still open -- a much larger undertaking (bounding box loss, NMS, real convergence) |
| RNN (time series) | CS596 final project | `02-deep-learning-cv/rnn-time-series/`, also `cs596-deep-learning-rnn-project` | completed | Trained LSTM on real AMZN data; found it performs *worse* than a naive "no change" baseline (see SUMMARY.md) -- worth testing whether adding more features (volume, other tickers) beats the naive baseline where lookback-only price history didn't |
| GAN | CS596 (theory) | — | todo | Simple DCGAN on MNIST/CIFAR as a from-scratch addition |
| Dense prediction / segmentation | P3 Ayoub (slides) | — | todo | U-Net on a small segmentation dataset |
| Deep RL (MDP, Q-learning) | CS566 | — | todo | Q-learning on Pacman/Gridworld — natural extension of the CS566 AI course |

## 03 — Graph / Network Models

| Model | Course | Location | Status | Next step |
|---|---|---|---|---|
| Girvan-Newman community detection | CS536 A1 | `03-graph-network/community-detection/` | completed | Ran on Zachary's Karate Club; recovers the club's real historical 2-faction split from pure graph structure (see SUMMARY.md) |
| Web mining project (PageRank vs. degree centrality, greedy-modularity) | CS536 | `03-graph-network/web-mining-project/`, also `cs536-web-mining-structure-analysis` | completed | Compared PageRank vs. degree centrality on a real Twitter ego-network (they diverge at rank 4); found this denser graph's achievable modularity is structurally lower than the karate club's (see SUMMARY.md) |
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
| PCA (+ loading matrix) | CS504 | `05-unsupervised-dimred/pca-nmf/` | completed | Applied to digits dataset (64 features); 21 components retain 90% of variance, 2 components alone visibly separate digit classes despite being unsupervised (see SUMMARY.md) |
| NMF | CS504 | `05-unsupervised-dimred/pca-nmf/` | completed | Compared vs. PCA on breast cancer (downstream accuracy: PCA=0.9825 matches raw, NMF=0.8860) and digits (component images: PCA diffuse/holistic, NMF sparse/parts-based) -- see SUMMARY.md |
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
