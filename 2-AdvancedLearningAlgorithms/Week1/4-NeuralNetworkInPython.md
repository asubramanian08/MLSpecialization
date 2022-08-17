# Forward Prop in Single Layer
* Implementing forward propagation without and libraries (from scratch)
* See [page 55](Lecture.pdf) in the lecture or the [3rd optional lab](#optional-lab-3-coffee-roasting-numpy) for the exact code. Note that it is very similar to the math which I will write out.

**Mathematical version of the code:**

Note: This is for the Coffee Roasting example (Layer 1: 3, Layer 2: 1)
* Layer 1 - $\vec{a}^{[1]}$
    $$ \vec{x} \leftarrow \{200 \space\space 17\} $$
    $$ \vec{w}_1^{[1]} \leftarrow \{1 \space\space 2\}\\
    b_1^{[1]} \leftarrow \{-1\}\\
    a_1^{[1]} = g(\vec{w}_1^{[1]} \cdot \vec{x} + b_1^{[1]}) $$
    $$ \vec{w}_2^{[1]} \leftarrow \{-3 \space\space 4\}\\
    b_2^{[1]} \leftarrow \{1\}\\
    a_2^{[1]} = g(\vec{w}_2^{[1]} \cdot \vec{x} + b_2^{[1]}) $$
    $$ \vec{w}_3^{[1]} \leftarrow \{5 \space\space -6\}\\
    b_3^{[1]} \leftarrow \{2\}\\
    a_3^{[1]} = g(\vec{w}_3^{[1]} \cdot \vec{x} + b_3^{[1]}) $$
    $$ \vec{a}^{[1]} \leftarrow \{a_1^{[1]} \space a_2^{[1]} \space a_13^{[1]} \} $$
* Layer 2 - $\vec{a}^{[2]}$

    $$ \vec{w}_1^{[2]} \leftarrow \{-7 \space\space 8\}\\
    b_1^{[2]} \leftarrow \{3\}\\
    a_1^{[2]} = g(\vec{w}_1^{[1]} \cdot \vec{x} + b_1^{[1]}) $$

* General Code Implementation

    Note $w_j^{[l]} \space \forall j,l$ is represented as wl_j in the code.

    ```python
    # In this code we are going to use 1D arrays
    x = np.array([200, 17])

    # PARA 1,2,3 are specific to each j and l
    wl_j = np.array([PARA1, PARA2])
    bl_j = np.array([PARA3])
    zl_j = np.dot(wl_j, "a(l-1)") + b
    al_j = sigmoid(zl_j)
    ```

# General Implementation of Forward Propagation
* My own "Dense" function code:
    * $a_in$: The activation vector for the previous layer (could be $\vec{x}$).
    * $W$: A matrix where the $i^\text{th}$ *column* contains all the $w$ values for the $i^\text{th}$ neuron. A layer with $n$ neurons will have a shape `a_in.size x n`.
    * $b$: A 1D array with all the b values for each neuron
    ```python
    def dense(a_in, W, b, g):
        units = W.shape[1] # number of neurons
        a_out = np.zeros(units)
        for j in range(units):
            w = W[:,j] # jth neuron's w values
            z = np.dot(w, a_in) + b[j]
            a_out[j] = g(z)
        return a_out
    ```
* Now the "Sequential" function implementation:
    ```python
    def sequential(x):
        a1 = dense(x, W1, b1)
        a2 = dense(a1, W2, b2)
        a3 = dense(a2, W3, b3)
        a4 = dense(a3, W4, b4)
        f_x = a4
        return f_x
    ```

# Optional Lab 3: Coffee Roasting NumPy
Lab 3 Jupyter [file](Labs/C2_W1_Lab03_CoffeeRoasting_Numpy.ipynb).

# Quiz: 100%
Quiz [file](Quizzes.md#neural-network-implementation-in-python)