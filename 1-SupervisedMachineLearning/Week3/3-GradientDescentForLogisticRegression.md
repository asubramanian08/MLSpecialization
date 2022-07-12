# Gradient Descent Implementation
* Here we are trying to find $\vec{w},b$
* Model (given $\vec{x}$) outputs: $f_{\vec{w},b}(\vec{x}) = \frac1{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$
* Cost Function:

    $$J(\vec{w},b)=\frac1m \sum_{i=1}^m\left[y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1-y^{(i)})\log(1-f_{\vec{w},b}(\vec{x}^{(i)}))\right]$$
* Gradient descent algorithm:

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w_j=w_j-\alpha \frac\partial{\partial w_j}J(\vec{w},b) \space \space \forall j \le n $<br>
    $\space \space \space \space b=b-\alpha \frac\partial{\partial b}J(\vec{w},b) $<br>
    $\text{:: simultaneous updates}$
* We can then expand the derivatives:

    $\frac\partial{\partial w_j}J(\vec{w},b) = \frac1m \sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}$

    $\frac\partial{\partial b}J(\vec{w},b) = \frac1m \sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})$
* Final algorithm: (using the derivative expansion)

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w_j=w_j-\alpha \frac1m \sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)} \space \space \forall j \le n $<br>
    $\space \space \space \space b=b-\alpha \frac1m \sum_{i=1}^m(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)}) $<br>
    $\text{:: simultaneous updates}$
* Earlier concepts also apply
    * Graphing the learning curve to make sure GD converges
    * Vectorization
    * Feature Scaling

# Optional Lab 6: Gradient Descent for Logistic Regression
Lab 6 Jupyter [file](Labs/C1_W3_Lab06_Gradient_Descent_Soln.ipynb).

# Optional Lab 7: Logistic Regression with scikit-learn
Lab 7 Jupyter [file](Labs/C1_W3_Lab07_Scikit_Learn_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#gradient-descent-for-logistic-regression)