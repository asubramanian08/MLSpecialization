# Advanced Optimization
* What is a faster and better function to use, rather than gradient descent
* One step of gradient descent: $w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(\vec{w}, b)$
* The "Adam" algorithm (Adaptive Moment estimation): will automatically increase of decrease the learning rate $\alpha$ depending on if the algorithm is going too slow or too fast.
* Intuition - have different learning rates for each variable
    1. If $w_j$ keeps in the same direction - increase $\alpha_j$
    2. If $w_j$ keeps oscillating - decrease $\alpha_j$
* NOTE: The actual algorithm is beyond the scope of this class
* Code
    ```python
    # Model
    model = Sequential([
        tf.keras.layers.Dense(units=25, activation='sigmoid')
        tf.keras.layers.Dense(units=15, activation='sigmoid')
        tf.keras.layers.Dense(units=10, activation='linear')
    ])

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Fit
    model.fit(X, Y,epochs=100)
    ```
* The "Adam" algorithm is much faster than Gradient Descent and have become the standard

# Additional Layer Types
* We doing just need to use the "Dense" layer type
* In a "Dense": Each neuron get an input from every other neuron in the previous layer
* Convolutional Layer: Each neuron look at a part of the previous layer's output
    * Faster computation
    * Need less data -> less prone to overfitting
* Prime example: EKG signals
    * Have the first neuron look at the first 20 inputs
    * Second neuron might look at 11th - 30th
    * Third neuron might take the 21th - 40th section
    * ...
    * That layer will be a convolutional layer
    * Next the second layer's neuron will look at sections of the first layer ...
    * Finally there will be a sigmoid layer that states if a person might have heart disease

# Quiz: 100%
Quiz [file](Quizzes.md#additional-neural-network-concepts)