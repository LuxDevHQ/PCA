#  Day 2: Dimensionality Reduction

### Topic: PCA and t-SNE

---

##  Summary

* Understand the **Curse of Dimensionality**
* Apply **Principal Component Analysis (PCA)** to reduce dimensions while preserving variance
* Use **t-SNE** for visualizing high-dimensional data in 2D/3D
* Explore **real-world use cases**: Visualization, speed-up model training, and removing redundancy

---

## 1. What is Dimensionality Reduction?

**Dimensionality Reduction** refers to techniques that transform data from a high-dimensional space into a lower-dimensional space **without losing important information**.

---

###  Analogy: Packing for a Trip

> Imagine you’re going on a trip and have to fit everything into a small suitcase. You want to **carry the essentials**, but not the bulk.
> Dimensionality reduction helps you **compress your dataset**, retaining just the “essentials.”

---

## 2. Curse of Dimensionality

As the number of features (dimensions) increases:

* Data becomes **sparse**
* Models **overfit** more easily
* Distance metrics become **less meaningful**
* Computation time increases
* **Visualization becomes impossible**

---

###  Analogy: Finding Friends in a City vs a Galaxy

> In a **2D city map**, it's easy to find people close to you. But if people were floating in **100D space**, everyone seems far apart!
> The more dimensions you have, the **less intuitive** relationships become.

---

## 3. Principal Component Analysis (PCA)

###  Goal:

Reduce the number of features while preserving as much **variance** (information) as possible.

---

###  Analogy 1: Compressing a Book

> Turning a 500-page novel into a 2-page summary that **captures the core plot** — that’s PCA.

---

###  Analogy 2: Shadows and Light

> Imagine casting shadows of a 3D object onto a wall.
> PCA finds the **most informative angle** from which to look at the data, and **projects it** onto a lower-dimensional surface.

---

## 4. How PCA Works (Step-by-Step)

1. **Standardize** the dataset
2. **Compute the covariance matrix**
3. **Calculate eigenvectors and eigenvalues**
4. **Select top k principal components** (based on largest variance)
5. **Project** data onto the new feature space

---

## 5. Visual Example of PCA

```plaintext
Before PCA (2D space):
   ●   ●
      ●
   ●       ●

After PCA (projected to 1D):
   ● ● ● ● ●   → All points aligned on a line (1D)
```

---

## 6. PCA in Python – Step-by-Step

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = pd.DataFrame({
    'Age': [25, 30, 45, 35, 50, 23, 40],
    'Income': [30000, 40000, 50000, 45000, 80000, 32000, 60000]
})

# 1. Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(data)

# 2. Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

# 3. Plot results
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Result")
plt.grid(True)
plt.show()
```

---

## 7. Choosing the Number of Components

Use the **explained variance ratio** to decide how many components to keep.

```python
pca = PCA().fit(scaled)
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs. Number of Components")
plt.grid(True)
plt.show()
```

---

## 8. Use Cases of PCA

| Use Case             | Why Use PCA                                    |
| -------------------- | ---------------------------------------------- |
|  Visualization     | Reduce 100D → 2D/3D for plotting               |
|  Speed Up Training  | Fewer features = faster models                 |
|  Remove Redundancy | Eliminate correlated or uninformative features |
|  Preprocessing     | Clean data before clustering or classification |
|  Genomics          | Reduce thousands of gene expression values     |

---

## 9. t-SNE – Visualizing High-Dimensional Data

###  What is t-SNE?

* **t-distributed Stochastic Neighbor Embedding**
* Unlike PCA (which captures variance), t-SNE captures **local relationships** between points
* Used **only for visualization**

---

### t-SNE vs PCA

| Feature        | PCA                      | t-SNE                     |
| -------------- | ------------------------ | ------------------------- |
| Type           | Linear                   | Non-linear                |
| Goal           | Preserve global variance | Preserve local similarity |
| Output         | Principal components     | 2D or 3D visual layout    |
| Use Case       | Preprocessing, speed     | Visualization             |
| Interpretable? | Yes                      | No (non-deterministic)    |

---

###  Analogy: Organizing a Library

* **PCA** = Reorganizing the entire library so that books with similar topics are on the same shelves.
* **t-SNE** = Creating a **map of the library** that shows **which books are closest**, regardless of their shelf.

---

## 10. Bonus: Visualizing with t-SNE (Python Example)

```python
from sklearn.manifold import TSNE

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(scaled)

# Plot
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title('t-SNE Visualization')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.grid(True)
plt.show()
```

---

## 11. Pros and Cons of PCA

###  Pros

* Simple and interpretable
* Improves model speed and performance
* Handles correlated features well

###  Cons

* Sensitive to feature scaling
* Components can be hard to interpret
* Captures **linear relationships** only

---

## 12. Final Analogy Recap

| Analogy                     | Concept                   |
| --------------------------- | ------------------------- |
| Compressing a Book          | Dimensionality reduction  |
| Casting Shadows             | Data projection           |
| Trip Packing                | Keep essentials only      |
| Library Map                 | t-SNE local relationships |
| Finding Friends in a Galaxy | Curse of Dimensionality   |

---
