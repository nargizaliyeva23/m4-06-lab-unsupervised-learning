![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Unsupervised Learning

## Overview

Supervised learning needs labeled data — but in the real world, labels are expensive, incomplete, or simply unavailable. Unsupervised learning lets you discover hidden structure in data without any labels at all. Clustering algorithms group similar observations together, while dimensionality-reduction techniques compress high-dimensional data into something you can actually visualize and reason about.

In this lab you will apply several clustering algorithms — K-Means, Agglomerative Clustering, and DBSCAN — to the classic Wine dataset, then explore three different dimensionality-reduction methods on the Palmer Penguins dataset. By the end, you'll combine both families of techniques to see whether unsupervised clustering can "rediscover" known species labels that were deliberately withheld.

This hands-on practice mirrors a common real-world workflow: explore unlabeled data, reduce its dimensionality, cluster it, and then validate the clusters against ground truth (when you have it) to build confidence in your pipeline.

## Learning Goals

By the end of this lab, you should be able to:

- Apply K-Means clustering and select an optimal *k* using the elbow method and silhouette scores.
- Compare hierarchical and density-based clustering approaches on the same dataset.
- Reduce high-dimensional data to 2-D with PCA, t-SNE, and MDS, and critically compare the projections.
- Evaluate unsupervised cluster quality against known labels using adjusted Rand index and normalized mutual information.

## Setup and Context

You'll work inside a Jupyter Notebook for this lab. All analysis, code, and written interpretations should live in a single notebook so that your reasoning is visible alongside the output.

The two datasets you'll use are bundled with popular Python libraries — no downloads required. The Wine dataset (13 chemical features for 178 Italian wines from three cultivars) comes from scikit-learn, and the Palmer Penguins dataset (bill, flipper, and body-mass measurements for three penguin species) comes from seaborn.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Getting Started

1. Create a new Jupyter Notebook called **`m4-06-unsupervised-learning.ipynb`**.
2. In the first cell, import everything you'll need:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
```

3. Work through the tasks in order. Each task builds on the previous one.
4. Include markdown cells between code cells to explain your observations.

## Tasks

### Task 1: K-Means Clustering

1. Load the Wine dataset (`load_wine()`) and convert it to a DataFrame.
2. Scale the features using `StandardScaler`.
3. Run K-Means for *k* = 2 through 10. For each *k*, record the **inertia** and the **silhouette score**.
4. Plot an **elbow curve** (inertia vs. *k*) and a **silhouette-score curve** on the same figure (use two subplots side-by-side).
5. Choose the optimal *k* and justify your choice in a markdown cell.
6. Using your chosen *k*, fit a final K-Means model. Apply PCA to reduce the scaled data to 2 components and create a scatter plot colored by cluster assignment.

**Guiding questions:** Where does the elbow appear? Do the silhouette score and the elbow agree on the best *k*?

### Task 2: Hierarchical & Density-Based Clustering

Using the same scaled Wine data from Task 1:

1. Apply `AgglomerativeClustering` with the same *k* you selected in Task 1. Record the cluster labels.
2. Compute the **linkage matrix** using `scipy.cluster.hierarchy.linkage` (Ward method) and plot the **dendrogram**. Truncate it to the last 20 merges for readability.
3. Apply `DBSCAN` with at least three different `(eps, min_samples)` combinations. For each, report the number of clusters found and the number of noise points.
4. Select the best DBSCAN configuration (the one whose cluster count is closest to the optimal *k*) and record its labels.
5. Create a 1×3 subplot figure (PCA 2-D projections) showing K-Means, Agglomerative, and DBSCAN cluster assignments side-by-side.
6. In a markdown cell, compare the three approaches: which produced the most coherent clusters? Where did they disagree?

### Task 3: Dimensionality Reduction Comparison

1. Load the Palmer Penguins dataset (`sns.load_dataset("penguins")`) and drop rows with missing values.
2. Select the four numeric columns (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`) and scale them.
3. Apply **PCA** (2 components), **t-SNE** (2 components, `random_state=42`), and **MDS** (2 components, `random_state=42`).
4. Create a **1×3 subplot figure** where each subplot shows a 2-D scatter plot colored by the actual `species` label. Use consistent colors across all three panels.
5. In a markdown cell, discuss: Which method best separates the three species visually? Which one distorts inter-cluster distances the most? Why might t-SNE give different-looking results on repeated runs?

### Task 4: Putting It Together

1. Take the scaled Penguins numeric data from Task 3 — but **drop the species column** so the algorithm cannot see it.
2. Apply K-Means with *k* = 3 to the label-free data.
3. Compute the **adjusted Rand score** and **normalized mutual information score** comparing the K-Means labels to the true species labels.
4. Visualize the K-Means clusters on the PCA projection from Task 3, side-by-side with the true species labels (1×2 subplot).
5. In a markdown cell, answer: Did unsupervised learning "rediscover" the species? Where did it succeed and where did it struggle? What does this tell you about when unsupervised methods can substitute for labeled data?

## Submission

### What to submit
- `m4-06-unsupervised-learning.ipynb` — completed notebook with all four tasks.

### Definition of done (checklist)
- [ ] Elbow curve and silhouette scores are plotted for K-Means; optimal *k* is justified.
- [ ] Agglomerative, dendrogram, and DBSCAN results are compared with K-Means.
- [ ] PCA, t-SNE, and MDS projections are shown in a single figure with species coloring.
- [ ] Adjusted Rand and NMI scores are reported; cluster-vs-species comparison is visualized.
- [ ] Every task includes at least one markdown cell with interpretation.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete unsupervised learning"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
