# Gradient Descent
* Gradient Descent is an algorithm we can use to determine the values of $w$ and $b$.
* We have some function $J(w,b)$, this could be any function not just linear regression
* Gradient Descent outline:
    * Pick some values for $w,b$
    * Change $w,b$ to find a smaller cost function
    * Repeat the above step until we are at a local min
* Intuition: Keep taking baby steps down hill until we are at the very bottom
* NOTE: GD only finds the *local min* and follows that path of least resistance

# Implementing Gradient Descent
* Algorithms:
    1. $tmp_w = w-\alpha \frac\partial{\partial w}J(w,b)$
    2. $tmp_b = b-\alpha \frac\partial{\partial b}J(w,b)$
    3. $w=tmp_w$
    4. $b=tmp_b$
    3. Repeat until convergence: not much of a difference in $J(w,b)$
* Understanding the algo
    * $\alpha$: The learning rate (how big of a step you take)
    * Simultaneously update values $w$ and $b$
* Question

    See [page 62](Lecture.pdf). Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J. What does this update statement do? (Assume $\alpha$ is small.) $$w=w-\alpha\frac{\partial J(w,b)}{\partial w}$$

    * [ ] Checks whether $w$ is equal to $w-\alpha\frac{\partial J(w,b)}{\partial w}$
    * [x] Updates parameter $w$ by a small amount

    This updates the parameter by a small amount, in order to reduce the cost $J$.

# Gradient Descent Intuition
* Simpler algo for GD: $w=w-\alpha \frac\partial{\partial w}J(w)$
* See [page 65](Lecture.pdf)
* Draw a tangent line to $J(w)$ and find its slope (derivative)
* If the slope is positive we end of decreasing $w$ and the same implies for the negative slope
* Question

    See [page 62](Lecture.pdf). Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J.

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w=w-\alpha \frac\partial{\partial w}J(w,b) $<br>
    $\space \space \space \space b=b-\alpha \frac\partial{\partial b}J(w,b) $

    Assume the learning rate  $\alpha $ is a small positive number. When $\frac{\partial J(w,b)}{\partial w} $ is a positive number (greater than zero) -- as in the example in the upper part of the slide shown above -- what happens to $w$ after one update step?

    * [ ] It is not possible to tell if $w$ will increase or decrease.
    * [ ] $w$ increases
    * [x] $w$ decreases
    * [ ] $w$ stays the same

    The learning rate  $\alpha $ is always a positive number, so if you take W minus a positive number, you end up with a new value for W that is smaller

# Learning Rate
* $w=w-\alpha \frac\partial{\partial w}J(w,b)$
* If the learning rate is too small: the algo will take too long
* If the learning rate is too large: we can overshoot the minimum (fail to converge)
* When we are exactly at the local min, the slope is 0 and our pos won't change
* Learning rate $\alpha$ is controls how big of a step we take in gradient descent

# Gradient Descent for Linear Regression
* LR model: $f_{w,b}(x)=wx+b$
* Squared error cost function: $J(w,b)=\frac1{2m}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})^2$
* Gradient Descent Algorithm:

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w=w-\alpha \frac\partial{\partial w}J(w,b) $<br>
    $\space \space \space \space b=b-\alpha \frac\partial{\partial b}J(w,b) $

    * $\frac\partial{\partial w}J(w,b) = \frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}$
    * $\frac\partial{\partial b}J(w,b) = \frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})$
* Calculus proof of the above formula - [page 72](Lecture.pdf)
* Final GD algo for LR:

    $ \text{repeat until convergence:} $<br>
    $\space \space \space \space w=w-\alpha\frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)} $<br>
    $\space \space \space \space b=b-\alpha\frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)}) $<br>
    $\text{NOTE: Updated simultaneously}$

# Running Gradient Descent
This video is just used to get an understanding of how gradient descent works using multiple examples. This is "Batch" Gradient Descent that looks at the entire training data set. Set [pages 77-86](Lecture.pdf),

# Optional Lab 5: Gradient Descent
Lab 5 Juypter [file](Labs/C1_W1_Lab05_Gradient_Descent_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#train-the-model-with-gradient-descent)