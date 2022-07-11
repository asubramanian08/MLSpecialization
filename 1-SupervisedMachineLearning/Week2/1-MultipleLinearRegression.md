# Multiple Features
* Before we used one feature (size of house) to predict the target (price of house)
* Example data table:

    Size - $x_1$ | # Bedrooms - $x_2$ | # Floors - $x_3$ | Age Of House - $x_4$
    --- | --- | --- | ---
    2104 | 5 | 1 | 45
    1416 | 3 | 2 | 40
    1534 | 3 | 2 | 30
     852 | 2 | 1 | 36
    ... | ... | ... | ...

* Notation:
    * $x_j$: $j^\text{th}$ feature
    * $n$: number of features
    * $\vec{x}^{(i)}$: $i^\text{th}$ training example (row vector)
    * $x_j^{(i)}$: feature $j$ in the $i^\text{th}$ training example
* Model: $$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$
    * This is the same as: $f_{\vec{w},b}(\vec{x}) = \sum_{i=1}^n (w_i x_i) + b$
    * $\vec{w} = [w_1, w_2, \dots, w_n]$ (row vector)
    * $b$: some base number
    * $\vec{x} = [x_1, x_2, \dots, x_n]$
    * Used to be: $f_{w,b}(x) = wx + b$
* Question

    In the training set below, what is $x_1^{(4)}$? Please type in the number below (this is an integer such as 123, no decimal points). See [page 4](../Lecture.pdf).

    Answer: $852$

    $x_1^{(4)}$ is the first feature (first column in the table) of the fourth training example (fourth row in the table).

# Vectorization Part 1
* Vectorization makes *code shorter* and *run more efficiently*
* NOTE: code is $0$ indexed and linear algebra is $1$ indexed
* Variables:
    * $\vec{w} = [w_1 \space w_2 \space w_3]$
    * $b$ is a number
    * $\vec{x} = [x_1 \space x_2 \space x_3]$

    ```python
    w = np.array([1.0, 2.5, -3.3])
    b = 4
    w = np.array([10, 20, 30])
    ```
* Not vectorized: (slow)

    $$f_{\vec{w},b}(\vec{x}) = \sum_{i=1}^n (w_i x_i) + b$$

    ```python
    f = 0
    for j in range(n):
        f = f + w[j] * w[j]
    f = f + b
    ```

* Vectorized code: (NumPy)

    $$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

    ```python
    f = np.dot(w,x) + b
    ```

    Faster because it runs in parallel hardware
* Question

    Which of the following is a vectorized implementation for computing a linear regression model’s prediction?

    * [x]
        ```python
        f = np.dot(w,x) + b
        ```
    * [ ]
        ```python
        f = 0
        for j in range(n):
            f = f + w[j] * w[j]
        f = f + b
        ```

    This numpy function uses parallel hardware to efficiently calculate the dot product.

# Vectorization Part 2
* What is happening behind the scenes for vectorization
    * No vectorization:
        ```python
        for j in range(0, 16):
            f = f + w[j] * x[j]
        ```

        timestep $i$: `f + w[i] * x[i]`
    * Vectorization:
        ```python
        np.dot(w,x)
        ```
        timestep ?: Multiple computations at once
* This can also help with Gradient Descent

    $$\vec{w} = (w_1 \space w_2 \cdots w_{16})$$
    $$\vec{d} = (d_1 \space d_2 \cdots d_{16})$$
    Goal: Compute $w_j = w_j - \alpha d_j$ for all $1\le j\le 16$

    * No vectorization:
        ```python
        for j in range(16):
            w[j] = w[j] - a * d[j]
        ```
    * Vectorization:
        ```python
        w = w - a * d
        ```
* Question

    Which of the following is a vectorized implementation for computing a linear regression model’s prediction?

    * [x]
        ```python
        f = np.dot(w,x) + b
        ```
    * [ ]
        ```python
        f = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b
        ```
    * [ ]
        ```python
        f = 0
        for j in range(n):
            f = f + w[j] * x[j]
        f = f + b
        ```

# Optional Lab 1: Python, NumPy, and Vectorization
Lab 1 Jupyter [file](Labs/C1_W2_Lab01_Python_Numpy_Vectorization_Soln.ipynb).

# Gradient Descent for Multiple Linear Regression
* Parameters: $\vec{w}$ and $b$
* Model: $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$
* Cost Function: $J(\vec{w}, b)$
* Gradient Descent general:

    $\text{repeat until convergence:}$<br>
    $\space \space \space \space w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(\vec{w}, b)$<br>
    $\space \space \space \space b = b - \alpha \frac{\partial}{\partial b} J(\vec{w}, b)$
* Expanded Gradient Descent:

    $\text{repeat until convergence:}$<br>
    $\space \space \space \space w_i = w_i - \alpha \frac1m \sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})x_i^{(i)} \space \space \forall i$<br>
    $\space \space \space \space b = b - \alpha \frac1m \sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})$<br>
    $\text{Simultaneously update!}$
* Using Normal Equation (instead of GD)
    * Gradient Descent is more general / applicable
    * NE only works for linear regression
    * Can be slow for $> 10,000$  features
    * GD is recommend but NE might be implements under the hood of some libraries

# Optional Lab 2: Multiple Linear Regression
Lab 2 Jupyter [file](Labs/C1_W2_Lab02_Multiple_Variable_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#multiple-linear-regression)