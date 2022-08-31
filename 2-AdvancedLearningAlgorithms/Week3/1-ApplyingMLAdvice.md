# Deciding what to try Next
* We have seen many different ML algorithms:
    * Linear Regression
    * Logistic Regression
    * Neural Networks
    * Next: Decision Trees
* Some ML projects take much faster than they need to take (6 Months -> 2 weeks)
* Tips for what to do next in an ML project: (saving time)
* Problem: linear regression for predicting housing prices
    $$J(\vec{w},b) = \frac1{2m} \sum_{i=1}^m \left(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}\right)^2 + \frac\lambda{2m} \sum_{j=1}^n w_j^2$$

    The model is giving large errors in the predictions:
    * Get more training examples
    * Try a smallest feature set
    * Get additional features
    * Adding polynomial features ($x_1^2$, $x_2^2$, $x_1x_2$, ...)
    * Decreasing $\lambda$
    * Increasing $\lambda$
* We will learn how to run diagnostics to find out what is working or not - save a lot of time

# Evaluating a Model
* How to evaluate the performance of a model
* Divide the data into **70% training and 30% testing**

    In an overfit model: $J_\text{test}(\vec{w},b)$ will be high when $J_\text{train}(\vec{w},b)$ will be low.
* Mathematical steps for linear regression:

    Fit the parameters by minimizing $J(\vec{w},b)$:
    $$J(\vec{w},b) = \min_{\vec{w,}, b} \left[ \frac1{2m_\text{train}} \sum_{i=1}^{m_\text{train}} \left(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)} \right)^2 + \frac\lambda{2m_\text{train}} \sum_{j=1}^n w_j^2 \right] $$

    Compute the test's error:
    $$J_\text{test}(\vec{w},b) = \frac1{2m_\text{test}} \left[ \sum_{i=1}^{m_\text{test}} \left( f_{\vec{w},b}(\vec{x}_\text{test}^{(i)})-y_\text{test}^{(i)} \right)^2 \right]$$

    Compute the training error:
    $$J_\text{train}(\vec{w},b) = \frac1{2m_\text{train}} \left[ \sum_{i=1}^{m_\text{train}} \left( f_{\vec{w},b}(\vec{x}_\text{train}^{(i)})-y_\text{train}^{(i)} \right)^2 \right]$$
* Mathematical steps for logistic regression:

    Fit the parameters by minimizing $J(\vec{w},b)$:
    $$J(\vec{w},b) = -\frac1m \sum_{i=1}^m \left[ y^{(i)}\log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) + (1 - y^{(i)})\log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right) \right] + \frac\lambda{2m} \sum_{j=1}^n w_j^2$$

    Compute the test's error:
    $$J_\text{test}(\vec{w},b) = -\frac1{m_\text{test}} \sum_{i=1}^{m_\text{test}} \left[ y_\text{test}^{(i)}\log\left(f_{\vec{w},b}(\vec{x}_\text{test}^{(i)})\right) + (1 - y_\text{test}^{(i)})\log\left(1 - f_{\vec{w},b}(\vec{x}_\text{test}^{(i)})\right) \right]$$

    Compute the training error:
    $$J_\text{train}(\vec{w},b) = -\frac1{m_\text{train}} \sum_{i=1}^{m_\text{train}} \left[ y_\text{train}^{(i)}\log\left(f_{\vec{w},b}(\vec{x}_\text{train}^{(i)})\right) + (1 - y_\text{train}^{(i)})\log\left(1 - f_{\vec{w},b}(\vec{x}_\text{train}^{(i)})\right) \right]$$

    **NOTE**: Another more practical way to define $J_\text{test}(\vec{w},b)$ and $J_\text{train}(\vec{w},b)$ is to count fraction misclassified. $ (\text{count of } \hat{y} \neq y) \div m$.
* Note: The term "$\frac\lambda{2m_\text{train}} \sum_{j=1}^n w_j^2$" is called a regularization term. It is there to help prevent overfitting and keep the $w_j$ terms small.

# Model Selection and Training / Cross Validation / Test Sets
* Problem: Given multiple different models, (say with different polynomial degrees), how do I choose what model is best to use?
* Split the data into:
    * Training set: 60%
    * Cross Validation (dev set): 20%
    * Test set: 20%
* Calculate the error using different sets:

    Let "$\text{set}$" be the name of the set's error we are checking:
    $$J_\text{set}(\vec{w}, b) = \frac1{2m_\text{set}} \left[ \sum_{i=1}^{m_\text{set}} \left( f_{\vec{w},b}(\vec{x}_\text{set}^{(i)}) - y_\text{set}^{(i)} \right)^2 \right]$$
* **Model selection**:

    $$d=1 \ \ \ \ f_{\vec{w},b}(\vec{x}) = w_1x_1 + b \\ d=2 \ \ \ \ f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x^2 + b \\ d=3 \ \ \ \ f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x^2 + w_3x^3 + b \\ \vdots \\ d=10 \ \ \ \ f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x^2 + \cdots + w_{10}x^{10} + b$$

    * Option 1: Calculate $J_\text{test}(w^{<d>}, b^{<d>})$ for all $d$. Choose the degree that minimizes $J_\text{test}(w^{<d>}, b^{<d>})$.
    * Option 2 (recommended - using dev set): Pick the one the minimizes $J_\text{dev}(w^{<d>}, b^{<d>})$. Test the error using $J_\text{test}(w^{<d>}, b^{<d>})$.
* The main idea is the not use the test set to make decisions about the model. The test set can then be used to see how good the model is, since it is fair (unbiased).

# Quiz: 100%
Quiz [file](Quizzes.md#advice-for-applying-machine-learning)