# Inference in Code
* TensorFlow is a popular framework for deep learning, another is PyTorch
* Task: Given a feature vector $\vec{x}$ with temperature and duration predict if the coffee will taste good.
* Creating / calculating layer 1:
    ```python
    # One test case with 200 celsius and 17 minutes
    x = np.array([[200.0, 17.0]])
    # Create a hidden layer of 3 neurons with the sigmoid activation function
    # "Dense" is a type of layer in NNs
    layer_1 = Dense(units=3, activation='sigmoid')
    # Compute the vector a1 by running the function "layer_1"
    a1 = layer_1(x)
    ```
* Creating / calculating layer 2:
    ```python
    layer_1 = Dense(units=1, activation='sigmoid')
    a2 = layer_1(a1)
    ```
* "Threshold" the output too see whether the coffee is predicted to be good.
    ```python
    if a2 >= 0.5:
        yhat = 1
    else:
        yhat = 0
    ```
* Look at the digit classification problem:
    ```python
    # Layer 1
    x = np.array([[0.0, ..., 245, ..., 240, ..., 0]])
    layer_1 = Dense(units=25, activation='sigmoid')
    a1 = layer_1(x)

    # Layer 2
    layer_2 = Dense(units=15, activation='sigmoid')
    a2 = layer_2(a1)

    # Layer 3
    layer_3 = Dense(units=1, activation='sigmoid')
    a3 = layer_3(a2)

    # Threshold
    if a3 >= 0.5:
        yhat = 1
    else
        yhat = 0
    ```

# Data in TensorFlow
* How data is represented in NumPy vs. TensorFlow
* **NumPy Feature Vectors**
    * 2x3 matrix
        $$\begin{bmatrix}
        1 & 2 & 3 \\
        4 & 5 & 6
        \end{bmatrix}$$
        ```python
        x = np.array([[1, 2, 3],
                      [4, 5, 6]])
        ```
    * Row vector (1x2)
        $$\begin{bmatrix}
        200 & 17
        \end{bmatrix}$$
        ```python
        x = np.array([[200, 17]])
        ```
    * Column vector (2x1)
        $$\begin{bmatrix}
        200\\
        17
        \end{bmatrix}$$
        ```python
        x = np.array([[200], [17]])
        ```
    * **1D vector**
        $$\begin{bmatrix}
        200\\
        17
        \end{bmatrix}$$
        ```python
        x = np.array([200, 17])
        ```
* We used to represent data in that *1D vector* form. In TensorFlow, though, everything should be a matrix.
* Back to the Coffee Roasting problem:
    ```python
    x = np.array([[200.0, 17.0]])
    layer_1 = Dense(units=3, activation='sigmoid')
    a1 = layer_1(x)

    # Here "a1" is a 1x3 matrix
    print(a1)
        # tf.Tensor([[0.2 0.7, 0.3]], shape=(1, 3), dtype=float32)
    print(a1.numpy())
        # array([[1.466, 1.125, 3.216]], dtype=float32)
    ```
* Note the different representation in NumPy vs. TensorFlow: NumPy has its array while TensorFlow has a Tensor

# Building a Neural Network
* See the creation of forward prop with the [above tensorflow code](#inference-in-code)
* New way to create a NN in TensorFlow: (Coffee Roasting Code)
    ```python
    # Create the model / data
    layer_1 = Dense(units=3, activation='sigmoid')
    layer_2 = Dense(units=1, activation='sigmoid')
    model = Sequential([layer_1, layer_2])
    # 4x2 training matrix
    x = np.array([[200.0, 17.0],
                  [120.0, 5.0],
                  [425.0, 20.0],
                  [212.0, 18.0]])
    y = np.array([1, 0, 0, 1])

    # Train the model (next week)
    model.compile(...)
    model.fit(x,y)

    # Forward Propagation / Inference
    model.predict(x_new)
    ```
* Note: The standard way to create the model without defining the layers:
    ```python
    # Create the model / data
    model = Sequential([
        Dense(units=3, activation='sigmoid'),
        Dense(units=1, activation='sigmoid')])
    ```
* See [page 41](Lecture.pdf) for digit classification NN code.

# Optional Lab 2: Coffee Roasting in TensorFlow
Lab 2 Jupyter [file](Labs/C2_W1_Lab02_CoffeeRoasting_TF.ipynb).

# Quiz: 100%
Quiz [file](Quizzes.md#tensorflow-implementation)