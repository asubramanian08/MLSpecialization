# Cost Function for Logistic Regression
* Training set:

    $x_1$ - tumor size (sm) | ... | $x_n$ - patient's age | $y$ - malignant?
    --- | --- | --- | ---
    10 | | 52 | 1
    2 | | 73 | 0
    5 | | 55 | 0
    12 | | 49 | 1
    ... | | ... | ...
* Notation
    * $m$: # training examples
    * $n$: # features
    * $y$: target (0 or 1)
    * $f_{\vec{w},b}(\vec{x}) = \frac1{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$: logistic regression model
* Squared error cost: $\frac1m \sum_{i=1}^m \frac12 (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$
    * With linear regression with produced a convex bowl shape
    * With our model the graph isn't so clean. That way it might not work for gradient descent
    * Note: Here the loss function is $L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}) = \frac12 (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$ (everything after the summation)
    * So, this code function doesn't work for us
* Logistic loss function

    $$L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right) = \begin{cases} -\log(f_{\vec{w},b}(\vec{x}^{(i)})) & y^{(i)}=1 \newline -\log(1-f_{\vec{w},b}(\vec{x}^{(i)})) & y^{(i)}=0 \end{cases}$$

    * Intuition: See [page 18-19](Lecture.pdf)
    * This graph produces a smooth curve when $y^{(i)}=0 \space \text{or} \space 1$
    * This will produce a convex graph and will work with gradient descent
* Question

    Why is the squared error cost not used in logistic regression?

    * [x] The non-linear nature of the model results in a “wiggly”, non-convex cost function with many potential local minima.
    * [ ] The mean squared error is used for logistic regression.

    If using the mean squared error for logistic regression, the cost function is "non-convex", so it's more difficult for gradient descent to find an optimal value for the parameters w and b.

# Optional Lab 4: Logistic Loss
Lab 4 Jupyter [file](Labs/C1_W3_Lab04_LogisticLoss_Soln.ipynb).

# Simplified Cost Function for Logistic Regression
* Non-piecewise Loss function:

    $$L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right) = -y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) - (1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))$$

    Previous loss function: $$L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right) = \begin{cases} -\log(f_{\vec{w},b}(\vec{x}^{(i)})) & y^{(i)}=1 \newline -\log(1-f_{\vec{w},b}(\vec{x}^{(i)})) & y^{(i)}=0 \end{cases}$$

    Note: This could only be done since $y$ can only be 0 or 1.
* New cost function:

    $$J(\vec{w},b)=\frac1m \sum_{i=1}^m\left[L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)})\right]$$

    Expanded:

    $$J(\vec{w},b)=\frac1m \sum_{i=1}^m\left[y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))\right]$$

    * People use this cost / loss function because of "maximum likelihood" estimation
* Question

    For the simplified loss function:

    $$L\left(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}\right) = -y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) - (1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))$$

    if the target $y^{(i)}=1$ then what does this expression simplify to?

    * [x] $-\log(f_{\vec{w},b}(\vec{x}^{(i)}))$
    * [ ] $-\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))$

    The second term of the expression is reduced to zero when the target equals 1.

# Optional Lab 5: Cost Function for Logistic Regression
Lab 5 Jupyter [file](Labs/C1_W3_Lab05_Cost_Function_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#cost-function-for-logistic-regression)