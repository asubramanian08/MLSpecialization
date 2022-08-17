# How Neural Networks are Implemented Efficiently
* Building large NNs is due to Vectorization (Matrix multiplication)
* Parallel computing hardware like GPUs and CPUs are good for this
* Looping implementation of a Layer (from [before](4-NeuralNetworkInPython.md#general-implementation-of-forward-propagation)):
    ```python
    # Create Data
    x = np.array([200, 17])
    W = np.array([[1, -3, 5],
                  [-2, 4, -6]])
    b = np.array([-1, 1, 2])

    # Create a layer
    def dense(a_in, W, b, g):
        units = W.shape[1]
        a_out = np.zeros(units)
        for j in range(units):
            w = W[:,j]
            z = np.dot(w, a_in) + b[j]
            a_out[j] = g(z)
        return a_out

    # Output -> [1, 0, 1]
    ```
* Vectorized Implementation:
    ```python
    # Create Data
    X = np.array([[200, 17]]) # 1x2 matrix
    W = np.array([[1, -3, 5], # Unchanged
                  [-2, 4, -6]])
    B = np.array([-1, 1, 2]) # 1x3 matrix


    # Create a layer
    def dense(a_in, W, B, g):
        units = W.shape[1] # Unchanged
        Z = np.matmul(A_in, W) + B # Matrix Multiplication
        A_out = g(Z) # Apply sigmoid on all Z
        return A_out

    # Output -> [[1, 0, 1]]
    ```

# Matrix Multiplication
* Vector dot product
    * Example:
        $$\begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} 3 \\ 4 \end{bmatrix} = (1 \times 3) + (2 \times 4)$$
    * In general:
        $$\begin{bmatrix} \uparrow \\ \vec{a} \\ \downarrow \end{bmatrix} \cdot \begin{bmatrix} \uparrow \\ \vec{w} \\ \downarrow \end{bmatrix} = \vec{a} \cdot \vec{w}$$
    * Transposing:
        $$ \vec{a} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \newline \vec{a}^T = \begin{bmatrix} 1 & 2 \end{bmatrix} $$
    * **Conclusion**: $\vec{a} \cdot \vec{w} = \vec{a}^T \times \vec{w}$ - Dot product is the matrix multiplication of the transposed vector.
* Vector matrix multiplication
    * General Form:
        $$ \vec{a} = \begin{bmatrix} \uparrow \\ \vec{a} \\ \downarrow \end{bmatrix} \space \space \space \space \vec{a}^T = \begin{bmatrix} \leftarrow & \vec{a}^T & \rightarrow \end{bmatrix} \space \space \space \space W = \begin{bmatrix} \uparrow & \uparrow \\ \vec{w_1} & \vec{w_2} \\ \downarrow & \downarrow \end{bmatrix} \newline Z = \vec{a} \cdot W = \vec{a}^TW = \begin{bmatrix} \vec{a}^T\vec{w_1} & \vec{a}^T\vec{w_2} \end{bmatrix} $$
    * Example:
        $$ \vec{a} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \space \space \space \space W = \begin{bmatrix} 3 & 5 \\ 4 & 6 \end{bmatrix} \newline \newline Z = \vec{a} \cdot W = \vec{a}^TW \newline Z = \begin{bmatrix} 1 & 2 \end{bmatrix} \times \begin{bmatrix} 3 & 5 \\ 4 & 6 \end{bmatrix} \newline z_1 = (1 \times 3) + (2 \times 4) \newline z_2 = (1 \times 5) + (2 \times 6) \newline Z= \begin{bmatrix} 11 & 17 \end{bmatrix} $$
* Matrix Matrix multiplication
    $$ A = \begin{bmatrix} 1 & -1 \\ 2 & -2 \end{bmatrix} \space \space \space \space A^T = \begin{bmatrix} 1 & 2 \\ -1 & -2 \end{bmatrix} \space \space \space \space W = \begin{bmatrix} 3 & 5 \\ 4 & 6 \end{bmatrix} \newline Z = A \cdot T = A^TW = \begin{bmatrix} \vec{a}_1^T\vec{w}_1 & \vec{a}_1^T\vec{w}_2 \\ \vec{a}_2^T\vec{w}_1 & \vec{a}_2^T\vec{w}_2 \end{bmatrix} = \begin{bmatrix} 11 & 17 \\ -11 & -17 \end{bmatrix} $$

# Matrix Multiplication Rules
$$ A = \begin{bmatrix} 1 & -1 & 0.1 \\ 2 & -2 & 0.2 \end{bmatrix} \space \space \space \space A^T = \begin{bmatrix} 1 & 2 \\ -1 & -2 \\ 0.1 & 0.2 \end{bmatrix} \space \space \space \space W = \begin{bmatrix} 3 & 5 & 7 & 9 \\ 4 & 6 & 8 & 0 \end{bmatrix} \newline Z = A \cdot W = A^TW = \begin{bmatrix} \vec{a}_1^T\vec{w}_1 & \vec{a}_1^T\vec{w}_2 & \vec{a}_1^T\vec{w}_3 & \vec{a}_1^T\vec{w}_4 \\ \vec{a}_2^T\vec{w}_1 & \vec{a}_2^T\vec{w}_2 & \vec{a}_2^T\vec{w}_3 & \vec{a}_2^T\vec{w}_4 \\ \vec{a}_3^T\vec{w}_1 & \vec{a}_3^T\vec{w}_2 & \vec{a}_3^T\vec{w}_3 & \vec{a}_3^T\vec{w}_4 \end{bmatrix}$$

**NOTE**: Each row of $A^T$ corresponds to a row in $Z$. Similarly, each column in $W$ corresponds to a column in $Z$.

Question: See [page 73](Lecture.pdf). Can you calculate the value at row 2, column 3?

* [x] $(-1 x 7) + (-2 x 8) = -23$
* [ ] $(0.1 x 5) + (0.2 x 6) = 1.7$
* [ ] $(1 x 3) + (2 x 4) = 11$

This is correct.  Take row 2 of $X^T$ and column 3 of $W$

**NOTE:** When multiplying matrices of shape $n_1 \times m_1$ by $n_2 \times m_2$, then $m_2 = n_1$.

# Matrix Multiplication Code
See the [above problem](#matrix-multiplication-rules) to see what the code is based off of. All the variables and their values are the same.

```python
# Declaring Variables

A = np.array([1, -1, 0.1],
             [2, -2, 0.2])
AT = np.array([1, 2],
              [-1, -2],
              [0.1, 0.2])
# AT = A.T # Transpose

W = np.array([3, 5, 7, 9],
             [4, 6, 8, 0])

Z = np.matmul(AT, W) # Z = AT @ W
```

Using vectorization to compute a dense layer: This layer will have 3 neurons taking in $A^T$ as an input and outputting A given matrix $W$ and vector $\vec{b}$ for each neuron. Note again that for first column of $W$ corresponds to the first neuron etc.

```python
# Declaring input variables
AT = np.array([[200, 17]])
W = np.array([[1, -3, 5],
              [-2, 4, -6]])
b = np.array([[-1, 1, 2]])

# Vectorized computing for layer
def dense(AT, W, b, g):
    z = np.matmul(AT, W) + b
    a_out = g(z)
    return a_out
```