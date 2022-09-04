# Alternatives to the Sigmoid Function
* Problem: Tee-shirt prediction
    * Input: price, shipping, marketing, material
    * Layers: affordability, awareness, quality
    * Output: How well the product with do
* Awareness can be a non-negative number instead of just a 1/0

    Application: $a_2^{[1]} = g(\vec{w}_2^{[1]} \cdot \vec{x} + b_2^{[1]})$
    * Previous activation (sigmoid): $g(z) = \frac1{1 + e^{-z}}$ - here $0 < g(z) < 1$
    * Most common activation (ReLU): $g(z) = max(0, z)$ - here $g(z) >= 0$
    * Linear Activation function: $g(z) = z$ - here there is no bound on $g(z)$.

# Choosing Activation Functions
* Choosing $g(z)$ for the output layer:
    * Binary Classification: Sigmoid function
    * Regression (stock prices): Linear Activation
    * Regression (house prices): ReLU activation
* For the hidden layer: ReLU is a much more common choice
    * Reason one: Its easier to computer ReLU
    * Reason two: gradient descent will be slower in the flat regions
* Note: There are a few other activation functions that can be used when applicable

# Why do we need Activation Functions
* If every neuron used Linear Activation, then the entire NN will just be doing linear regression
* For example if we have 2 hidden layers with one neuron, this is the math:

    $$a^{[1]} = w_1^{[1]}x + b_1^{[1]}$$
    $$a^{[2]} = w_1^{[2]}a^{[1]} + b_1^{[2]}$$
    $$a^{[2]} = w_1^{[2]}(w_1^{[1]}x + b_1^{[1]}) + b_1^{[2]}$$
    $$a^{[2]} = (w_1^{[2]}w_1^{[1]})x + w_1^{[2]}b_1^{[1]} + b_1^{[2]}$$
    Since all w and b's are scalars this effectively becomes: $a^{[2]} = wx + b$

    This is just linear regression.
* In the general case:

    If you have linear activation for all the neurons in the hidden layers, the entire NN will just become whatever activation function was used in the output layer.

# Optional Lab 1: ReLU Activation
Lab 1 Jupyter [file](Labs/C2_W2_Relu.ipynb).

# Quiz: 100%
Quiz [file](Quizzes.md#additional-neural-network-concepts)