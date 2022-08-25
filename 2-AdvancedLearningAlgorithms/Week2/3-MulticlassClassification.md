# Multiclass
* Haves more than two possible output values
* Problem: Recognizing handwritten digits (not just 1 and 0)
* Next we will learn about a tool than can create multiple boundary

# Softmax
* Generalization of logistic regression for multiclass classification
* Logistic Regression (2 output classes):

    $$z = \vec{z} \cdot \vec{x} + b$$

    * $a_1 = g(z) = \frac1{1 + e^{-z}} = P(y = 1 \mid \vec{x})$
    * $a_2 = 1 - a_1 = P(y = 0 \mid \vec{x})$
* Softmax Regression (ex. 4 outputs)
    $$z_i = \vec{w}_i \cdot \vec{x} + b$$

    $$a_i = \frac{e^{z_i}}{\sum_{j=1}^4 e^{z_j}} = P(y = i \mid \vec{x})$$

    Note: $\sum_{i = 1}^n a_i$ must $ = 1$.
* Question:

    What do you think $a_4$ is equal to? See [page 25](Lecture.pdf)
    * [x] 0.35
    * [ ] 0.40
    * [ ] -0.40

    This is correct. Please continue the video to see why!
* If there are 2 features in softmax regression, it simplifies to logistic regression.
* Loss and Cost values

    Logistic Regression:
    $$loss = -y\log a_1 - (1-y) \log(1 - a_1)$$
    $$loss = -y\log a_1 - (1-y) \log a_2$$
    $$J(\vec{w},b) = \text{average loss}$$

    Softmax Regression:
    $$loss(a_1, ..., a_N, y) = \begin{cases}-\log a_1 & \text{if } y = 1 \\ -\log a_2 & \text{if } y = 2 \\
    & \vdots \\ -\log a_N & \text{if } y = N \end{cases} $$

# Neural Network with Softmax Output
* If there are $N$ output classes, then the output layer will have $N$ neurons with a "softmax" activation
* Softmax is a little different than other activation functions: In other functions $a_i$ is a function of $z_i$ but in softmax, $a_i$ is a function of all $z_j$.
* Code:
    1. Specify the model $f_{\vec{w},b}(\vec{x})$
        ```python
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential([
            Dense(units=25, activation='relu')
            Dense(units=15, activation='relu')
            Dense(units=10, activation='softmax')
        ])
        ```
    2. Specify the loss and cost function $L(f_{\vec{w},b}(\vec{x}), y)$
        ```python
        from tensorflow.keras.losses import SparseCategoricalCrossentropy
        model.compile(loss=SparseCategoricalCrossentropy())
        ```
    3. Train the model to minimize $J(\vec{w},b)$
        ```python
        model.fit(X, Y, epochs=100)
        ```
    * NOTE: Don't use the code as implemented here. There is another recommended way to implement softmax in tensorflow. (It is shown below.)

# Improved Implementation of Softmax
* Roundoff error: The above implementation of softmax is technically correct but this is very prone to roundoff errors.
* In logistic regression:

    ```python
    model = Sequential([
        Dense(units=25, activation='relu')
        Dense(units=15, activation='relu')
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(loss=BinaryCrossEntropy())
    ```
    In the above code, tensor flow will independently calculate $a$ value and then calculate the loss. It will lead to less round of errors if this intermediate step wasn't done. (Plugging the a value directly into the loss formula will allow TF to rearrange the numbers to reduce roundoff error) See the below code for how to fix that problem.

    ```python
    model.compile(loss=BinaryCrossEntropy(from_logits=True))
    ```
* Fixing roundoff error in softmax:
    ```python
    model = Sequential([
        Dense(units=25, activation='relu')
        Dense(units=15, activation='relu')
        Dense(units=10, activation='linear')
    ])
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))
    ```

    Notes: We are no longer using the softmax activation, instead a linear activation. So to get the output we have to map it to softmax, see the full implementation below:

```python
# Model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(units=25, activation='relu')
    Dense(units=15, activation='relu')
    Dense(units=10, activation='linear')
])

# Loss
from tensorflow.keras.losses import
SparseCategoricalCrossentropy
model.compile(...,loss=SparseCategoricalCrossentropy(from_logits=True) )

# Fit
model.fit(X,Y,epochs=100)

# Predict
logits = model(X)
f_x = tf.nn.softmax(logits)
```

# Classification with Multiple Outputs (Optional)
* In multilabel classification there are multiple outputs / labels for one given input

    For example: Is there a car, bus, pedestrian
    * Option 1: Make 3 different NNs for each of the three different problems
    * Option 2: Multilabel classification
* The final output layer would have $N$ neurons for the $N$ different labels

# Optional Lab 2: Softmax
Lab 2 Jupyter [file](Labs/C2_W2_SoftMax.ipynb).

# Optional Lab 3: Multiclass
Lab 3 Jupyter [file](Labs/C2_W2_Multiclass_TF.ipynb).

# Quiz: 100%
Quiz [file](Quizzes.md#multiclass-classification)