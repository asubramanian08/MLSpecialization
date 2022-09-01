# Finding Unusual Events
* Example: Trying to find when aircraft engines are not working
    * $x_1$: heat generated
    * $x_2$: vibration intensity
    * Dataset: $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$
    * New engine $x_\text{test}$: Determine if it is faulty or not
* Density Estimation:
    * Define $p(x)$: The probability that an $x$ exists
    * If $p(x) < \epsilon$ then we will raise a flag - might be faulty
    * Uses: Fraud detection, manufacturing, monitoring computers to ensure they are working

# Gaussian (Normal) Distribution
* Gaussian Distribution:
    
    $$p(x) = \frac1{\sqrt{2\pi}\sigma}e^{\frac{-{(x - \mu)}^2}{2\sigma^2}}$$

    * Let $x$ be a random variable. Let $\mu$ be the mean and $\sigma$ be the standard deviation (and $\sigma^2$ is the variance)
    * The probability of $x$ will look like a bell curve, this function is the Gaussian distribution
    * If you make a histogram of the data (of infinite examples), you will see a bell curve
    * See [page 41](Lecture.pdf)
* Changing $\mu$ and $\sigma$
    * The bell curve will be centered at $\mu$
    * Increasing $\sigma$ will make the distribution more spread out
    * Decreasing $\sigma$ will make the bell curve more concentrated, taller
* Calculating $\mu$ and variance $\sigma^2$
    * $$\mu = \frac1m \sum_{i=1}^m x^{(i)}$$
    * $$\sigma^2 = \frac1m \sum_{i=1}^m \left( x^{(i)} - \mu \right)^2$$

# Anomaly Detection Algorithm
* Problem description:
    * Training set: $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$
    * Each example $x^{(i)}$ have $n$ features
    * $$p(x) = \prod_{j=1}^n p(x_j; \mu_j, \sigma_j^2)$$
* Algorithm:
    1. Choose $n$ features $x_i$ that might be indicative of an anomaly
    2. Fit parameters $\mu_i$ and $\sigma_i^2$ for each feature $x_i$
        
        $$\mu_j = \frac1m \sum_{i=1}^m x_j^{(i)}$$
        
        $$\sigma_j^2 = \frac1m \sum_{i=1}^m \left( x_j^{(i)} - \mu_j \right)^2$$
        
        Over the vectorized version:

        $$\vec\mu = \frac1m \sum_{i=1}^m \vec{x}^{(i)}$$
    3. Given a new example $x$, compute $p(x)$

        $$p(x) = \prod_{j=1}^n p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^n \frac1{\sqrt{2\pi}\sigma_j}e^{-\frac{{(x_j - \mu_j)}^2}{2\sigma_j^2}}$$
    4. If $p(x) < \epsilon$ we declare an anomaly

# Developing and Evaluating and Anomaly Detection System
* Practical tips for evaluating a system
* Assume we have some labeled data: $y=0$ if normal and $y=1$ otherwise
    * We need enough to create a cross-validation set and a test set that includes a few anomalies
    * Cross validation set: $\left( x_\text{cv}^{(1)}, y_\text{cv}^{(1)} \right), \dots \left( x_\text{cv}^{(m_\text{cv})}, y_\text{cv}^{(m_\text{cv})} \right)$
    * Test set: $\left( x_\text{test}^{(1)}, y_\text{test}^{(1)} \right), \dots \left( x_\text{test}^{(m_\text{test})}, y_\text{test}^{(m_\text{test})} \right)$
* Dividing the sets (using airplane engine example)
    * $10000$ good engine examples (its ok if there are a few anomalies in here)
    * $20$ anomalous engines (this can range from $2$ to $50$)
    * Training set: $6000$ good engines

        Train the algorithm on the training set
    * Cross validation set: $2000$ good engines ($y=0$), $10$ anomalies ($y=1$)

        Tune $\epsilon$ and added or subtracted features $x_j$
    * Test set: $2000$ good engines ($y=0$), $10$ anomalies ($y=1$)

        Evaluate the algorithm on the test set
    * There might not be enough data for the test set ($2$ anomalies), then we just create the cross validation set. The problem is that we don't know if the algorithm is overfitting or not.
* Since anomaly detection typically uses a skewed data set: see [my notes](../../2-AdvancedLearningAlgorithms/Week3/4-SkewedDatasets(Optional).md) on skewed data sets
    * True Positive, False Positive, True Negative, False Negative
    * Precision: True Positive / (True Positive + False Positive)
    * Recall: True Positive / (True Positive + False Negative)
    * F1 Score: (2 * Precision * Recall) / (Precision + Recall)
    * Use F1 Score to evaluate how good the algorithm is to choose $\epsilon$

# Anomaly Detection vs. Supervised Learning
* When should I use Anomaly Detection vs. Supervised Learning?
* Anomaly Detection
    * Small number of positive examples ($y=1$), $0-20$ is common
    * Large number of negative examples ($y=0$)
    * Many different type of anomalies: There are many ways things can go wrong. If something goes wrong that hasn't gone wrong before, this algorithm will pick up on it.
    * Example: Fraud detection, manufacturing (defects), monitoring machines in data centers
* Supervised Learning
    * Large number of positive and negative examples
    * Future anomalies are likely to be similar to the past positive examples
    * Example: Sorting spam emails, manufacturing (ex. scratches), weather (sunny, rainy, ...), disease classification

# Choosing what Features to use
* In anomaly detection its difficult for it to choose what features to ignore. Therefore it is important to choose the right features to use.
* Fixing Non-Gaussian Features:
    * Example: If the distribution of $x$ on a histogram is not Gaussian, then it needs to be changed
    * Note: use `plt.hist` to plot a histogram of the data
    * Option 1: $\log(x + c)$ for some constant $c$
    * Option 2: $x^c$ where $c$ is some constant. Ex: $\sqrt{x}$ or $x^{0.04}$
    * Just try out a bunch of different options and make it look Gaussian
    * See [page 56](Lecture.pdf)
* You can also use [error analysis](../../2-AdvancedLearningAlgorithms/Week3/4-SkewedDatasets(Optional).md#error-metrics-for-skewed-datasets): Sample random data to find out what types of things the data is getting wrong.
    * Try picking new features that might indicate an anomaly
    * See [page 58](Lecture.pdf)
* Example in a computer center:
    * Anomalies are having a high CPU load and low network traffic
    * Create a new feature that is the ratio of CPU load to network traffic
    * See [page 59](Lecture.pdf)

# Quiz: 100%
Quiz [file](Quizzes.md#anomaly-detection)