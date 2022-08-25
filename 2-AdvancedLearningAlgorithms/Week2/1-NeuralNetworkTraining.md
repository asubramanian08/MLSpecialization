# TensorFlow Implementation
* This week we are going to train the neural network
* Problem: Digit recognition
    * layer 1: 25 neurons
    * layer 2: 15 neurons
    * layer 3: 1 neuron
* Code to train the model
    ```python
    # Part 1: Create the NN (from last week)
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential([
        Dense(units=25, activation='sigmoid')
        Dense(units=15, activation='sigmoid')
        Dense(units=1, activation='sigmoid')
    ])

    # Part 2: What is the loss function
    from tensorflow.keras.losses import BinaryCrossentropy
    model.compile(loss=BinaryCrossentropy())

    # Part 3: Fit the model after running "epochs" iteration
    model.fit(X,Y,epochs=100)
    ```

# Training Details
* Logistic Regression:
    1. Define the output function: $f_{\vec{w},b}(\vec{x})$.
    2. Specify the loss and cost functions: $L(f_{\vec{w},b}(\vec{x}), y)$ and $J(\vec{w},b) = \frac1m \sum_{i = 1}^m L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)})$.
    3. Train using the data to minimize $J(\vec{w},b)$.
* Neural Network steps:
    1. Define the model: output given x and and parameters w,b
        ```python
        # Part 1: Create the NN (from last week)
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential([
            Dense(units=25, activation='sigmoid')
            Dense(units=15, activation='sigmoid')
            Dense(units=1, activation='sigmoid')
        ])
        ```
    2. Loss and cost function:

        * Cost Function

            $$J(W, B) = \frac1m \sum_{i=1}^m L(f(\vec{x}^{(i)}),y^{(i)})$$

        * Loss functions from binary classification problems - binary cross entropy:

            $$L(f(\vec{w}),y) = -y\log(f(\vec{x})) - (1 - y)\log(1 - f(\vec{x}))$$

            ```python
            # Part 2: What is the loss function
            from tensorflow.keras.losses import BinaryCrossentropy
            model.compile(loss=BinaryCrossentropy())
            ```

        * Another loss function: Squared error loss

            ```python
            # Part 2: What is the loss function
            from tensorflow.keras.losses.import MeanSquaredError
            model.compile(loss=MeanSquaredError())
            ```
    3. Train the model: Gradient Descent

        $$\begin{align*}
        &\text{repeat until convergence:} \; \lbrace \\
        &  \; \; \;w_j^{[l]} = w_j^{[l]} -  \alpha \frac{\partial}{\partial w_j} J(\vec{w},b) \\
        &  \; \; \;  \; \;b_j^{[l]} = b_j^{[l]} -  \alpha \frac{\partial}{\partial b} J(\vec{w},b) \\
        &\rbrace
        \end{align*}$$

        Take small steps until local minimum is achieved. **Back Propagation** is used to compute those derivatives.

# Quiz: 100%
Quiz [file](Quizzes.md#neural-network-training)