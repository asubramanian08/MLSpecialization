# What is Clustering
* Clustering is a technique to group similar items together
* Supervised learning version:
    * We have a training set that contains that input and output
* Unsupervised learning version:
    * We have a training set that contains only the input
    * The points are not labeled
    * We can use clustering to group similar points together
    * Ex: Grouping news, market segmentation, DNA analysis, galaxy grouping

# K-Means Intuition
* Below are the steps to the algorithm
    1. Pick k random points as center of the clusters - "centroids"
    2. Assign each point to the closest centroid
    3. Move the centroids to the average of all points assigned to it
    4. Repeat steps 2 and 3 until convergence: The centroids don't move anymore

# K-Means Algorithm
* Randomly pick $K$ points as the centroids: $\mu_1, \mu_2, ..., \mu_K$
    * Note: every $\mu$ is an $n$-dimensional vector, where $n$ is the number of features in the input data
* Assign each point to the cluster centroids

    for $i = 1$ to $m$:
    * $c^{(i)}$ := index of the closest centroid to $x^{(i)}$ - $\min_k {|| x^{(i)} - \mu_k ||}^2$
* Move centroids

    for $k = 1$ to $K$:
    * $\mu_k$ := average of all points assigned to cluster $k$

        Example: $\mu_1 = \frac14 [x^{(1)} + x^{(5)} + x^{(6)} + x^{(10)}]$
    * If no points are assigned to a cluster, then move the centroid to a random point
* Repeat steps 2 and 3 until convergence

# Optimization Objective
* Notation
    * $c^{(i)}$: Index of the cluster that $x^{(i)}$ belongs to
    * $\mu_k$: Centroid of cluster $k$
    * $\mu_{c^{(i)}}$: Centroid of cluster $c^{(i)}$
* Cost function: (Distortion function)

    $$J(c, \mu) = \frac1m \sum_{i=1}^m \left( {||x^{(i)} - \mu_{c^{(i)}}||} ^ 2 \right)$$
* Both steps in the algorithm are both used to optimize the cost function

# Initializing K-Means
* How do we randomly pick the locations of the centroids?
* Randomly pick $K$ training examples and set $\mu_1$ through $\mu_K$ to those examples
* Some initializations might be stuck in local optima:
    * We could run the algorithm multiple times and pick the best one
    * "Best" is defined as the one that minimizes the cost function $J(c, \mu)$

# Choosing the Number of Clusters
* Given one dataset, it may seem there are multiple different number of clusters that can be used. See [page 33](Lecture.pdf)
* Elbow method:
    * Graph the number of clusters against the cost function $J(c, \mu)$
    * The elbow is the point where the cost function decreases the most
    * Pick the elbow as the number of clusters

# Quiz: 100%
Quiz [file](Quizzes.md#clustering)