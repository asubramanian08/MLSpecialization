# The Problem of Overfitting
* Underfit ("high bias"): Does not fit the training set well "high bias" See [page 30](Lecture.pdf)
* Well fit: Matches the data and will do a good prediction
* Overfit ("high variance"): Fits the training set really well but won't work well for predictions.
    * Ex. The function has way to high a degree
    * The graph varies too much and won't give an accurate prediction
    * Doesn't **generalize** well
* In general the *higher polynomial* will increase the "fitness". This can be measures by the amount of edges / curves in terms of classification see [page 31](Lecture.pdf)
* Question

    Our goal when creating a model is to be able to use the model to predict outcomes correctly for **new examples**. A model which does this is said to **generalize** well.

    When a model fits the training data well but does not work well with new examples that are not in the training set, this is an example of:

    * [ ] Underfitting (high bias)
    * [ ] None of the above
    * [ ] A model that generalizes well (neither high variance nor high bias)
    * [x] Overfitting (high variance)

    This is when the model does not generalize well to new examples.

# Addressing Overfitting
* Option 1: Get more training data
* Option 2: Feature selection
    * Adding more features with not enough training data will cause overfitting
    * In that case: use only the most relevant features
* Option 3: [Regularization](#regularized-linear-regression)!
    * shrink the $\vec{w}$ values for high degrees features
* Question

    Applying regularization, increasing the number of training examples, or selecting a subset of the most relevant features are methods for …

    * [x] Addressing overfitting (high variance)
    * [ ] Addressing underfitting (high bias)

    These methods can help the model generalize better to new examples that are not in the training set.

# Optional Lab 8: Overfitting
Lab 8 Jupyter [file](Labs/C1_W3_Lab08_Overfitting_Soln.ipynb).

# Cost function with Regularization
* Let's say we have $n$ parameters
* Our original cost function:

    $$J(\vec{w},b)=\frac1{2m}\sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2$$
* To the cost function lets add a regularization term
    $$J(\vec{w},b)=\frac1{2m}\sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2 + \frac\lambda{2m} \sum_{j=1}^n w_j^2$$

    * $\lambda$: "lambda" the regularization parameter $\lambda > 0$
    * Regularization term: $\frac\lambda{2m} \sum_{j=1}^n w_j^2$
    * This basically "penalizes" the $\vec{w}$ values to be large
    * Note: Most programers don't "penalized" $b$
* Effects of $\lambda$
    * Too large: all the $\vec{w}$ values will $\rightarrow 0$ and $f(x) \approx b$
    * Too small: no regularization and overfitting will happen
* Question

    For a model that includes the regularization parameter $\lambda$ (lambda), increasing $\lambda$ will tend to …

    * [ ] Decrease the size of the parameter $b$.
    * [ ] Increase the size of the parameters $w_1, w_2, \dots, w_n$.
    * [ ] Increase the size of the parameter $b$.
    * [x] Decrease the size of the parameters $w_1, w_2, \dots, w_n$.

    Increasing the regularization parameter $\lambda$ reduces overfitting by reducing the size of the parameters. For some parameters that are near zero, this reduces the effect of the associated features.

# Regularized Linear Regression
* Cost function (squared error + regularization)


    $$J(\vec{w},b)=\frac1{2m}\sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2 + \frac\lambda{2m} \sum_{j=1}^n w_j^2$$
* Old gradient descent algorithm

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w_j=w_j-\alpha \frac\partial{\partial w_j}J(\vec{w},b) \space \space \forall j \le n $<br>
    $\space \space \space \space b=b-\alpha \frac\partial{\partial b}J(\vec{w},b) $<br>
    $\text{:: simultaneous updates}$
* Derivative Expanded + Regularized GD algorithm

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w_j=w_j-\alpha \left[\frac1m \sum_{i=1}^m \left[(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}\right] + \frac\lambda{m}w_j\right] \space \space \forall j \le n$<br>
    $\space \space \space \space b=b-\alpha \frac1m \sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) $<br>
    $\text{:: simultaneous updates}$
* Intuition of math and derivatives
    * How regularization shrinks $\vec{w}$ values: See [page 44](Lecture.pdf)
    * Derivative calculations (Calculus): See [page 45](Lecture.pdf)
* Question

    Recall the gradient descent algorithm utilizes the gradient calculation:

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w_j=w_j-\alpha \left[\frac1m \sum_{i=1}^m \left[(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}\right] + \frac\lambda{m}w_j\right] \space \space \forall j \le n$<br>
    $\space \space \space \space b=b-\alpha \frac1m \sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) $<br>
    $\text{:: simultaneous updates}$

    Where each iteration performs simultaneous updates on $w_j \space \forall j$.

    In lecture, this was rearranged to emphasize the impact of regularization:

    $$w_j=w_j-\alpha \left[\frac1m \sum_{i=1}^m \left[(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}\right] + \frac\lambda{m}w_j\right] \space \space \forall j \le n$$

    is rearranged to be:

    $$w_j=w_j\left(1- \alpha \frac\lambda{m}\right) -\alpha \frac1m \sum_{i=1}^m \left[(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}\right] \space \space \forall j \le n$$

    Assuming $\alpha$, the learning rate is a small number like $0.001$, $\lambda$ is 1, and $m = 50$, what is the effect of the 'new part' on updating $w_j$?

    * [x] The new part decreases the value of $w_j$ each iteration by a little bit.
    * [ ] The new part increases the value of $w_j$ each iteration by a little bit.
    * [ ] The new parts impact varies each iteration.

    the new term decreases $w_j$ each iteration

# Regularized Logistic Regression
* Similar to linear regression, logistic regression is prove to overfitting
* Original cost function

    $$J(\vec{w},b)=-\frac1m \sum_{i=1}^m\left[y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))\right]$$
* Adding the regularization section:

    $$J(\vec{w},b)=-\frac1m \sum_{i=1}^m\left[y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))\right] + \frac\lambda{2m} \sum_{j=1}^n w_j^2$$
* Old gradient descent algorithm

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w_j=w_j-\alpha \frac\partial{\partial w_j}J(\vec{w},b) \space \space \forall j \le n $<br>
    $\space \space \space \space b=b-\alpha \frac\partial{\partial b}J(\vec{w},b) $<br>
    $\text{:: simultaneous updates}$
* Derivative Expanded + Regularized GD algorithm

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w_j=w_j-\alpha \left[\frac1m \sum_{i=1}^m \left[(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}\right] + \frac\lambda{m}w_j\right] \space \space \forall j \le n$<br>
    $\space \space \space \space b=b-\alpha \frac1m \sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) $<br>
    $\text{:: simultaneous updates}$
* Question:

    See [page 48](Lecture.pdf). For regularized **logistic** regression, how do the gradient descent update steps compare to the steps for linear regression?

    * [x] They look very similar, but the $f(x)$ in not the same
    * [ ] They are identical

    For logistic regression, $f(x)$ is the sigmoid (logistic) function, whereas for linear regression, $f(x)$ is a linear function.

# Optional Lab 9: Regularization
Lab 9 Jupyter [file](Labs/C1_W3_Lab09_Regularization_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#the-problem-of-overfitting)