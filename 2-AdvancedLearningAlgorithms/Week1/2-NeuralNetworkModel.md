# Neural Network Layer
* Given an input $\vec{x}$ or activation values $\vec{a}$.
* Use the sigmoid function $g(z) = \frac1{1 + e^{-z}}$
* For our purposes let's use the first hidden layer as the example
* The "$x^{[i]}$" is associated with the $i^\text{th}$ layer
* For each neuron $i$ in the 1st layer, $a_i^{[1]} = g(\vec{w}_i^{[1]} \cdot \vec{x} + b_i^{[1]})$ where $\vec{w}_i^{[1]}$ and $b_i^{[1]}$ are value specific to the $i^\text{th}$ neuron in the 1st layer.
* For hidden layers $j$ other than the 1st and neuron $i$: $a_i^{[j]} = g(\vec{w}_i^{[j]} \cdot \vec{a}^{[j-1]} + b_i^{[j]})$

# More Complex Neural Networks
* General neural network:

    ![drawio now loading](Drawio/ComplexNN.drawio.svg)
* Layer 3:
    * $a_1^{[3]} = g(\vec{w}_1^{[3]} \cdot \vec{a}^{[2]} + b_1^{[3]})$
    * $a_2^{[3]} = g(\vec{w}_2^{[3]} \cdot \vec{a}^{[2]} + b_2^{[3]})$
    * $a_3^{[3]} = g(\vec{w}_3^{[3]} \cdot \vec{a}^{[2]} + b_3^{[3]})$
* Question

    Can you fill in the superscripts and subscripts for the second neuron?

    See [page 27](Lecture.pdf).

    * [x] $a_2^{[3]} = g(\vec{w}_2^{[3]} \cdot \vec{a}^{[2]} + b_2^{[3]})$
    * [ ] $a_2^{[3]} = g(\vec{w}_2^{[3]} \cdot \vec{a}^{[3]} + b_2^{[3]})$
    * [ ] $a_2^{[3]} = g(\vec{w}_2^{[3]} \cdot \vec{a}_2^{[2]} + b_2^{[3]})$

    This is correct. Please continue the lecture video to learn why!
* General form: $a_j^{[l]} = g(\vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]})$
* The sigmoid function $g$ is the "activation function"

# Inference: Making Predictions (Forward Propagation)
* Looking at the problem of digit regression: Given an image of a digit output if it is a 1 or a 0
* We will have:
    * Layer 1: 25 neurons
    * Layer 2: 15 neurons
    * Layer 3: 1 neuron (output)
* Forward propagation: Making the computation from left to right
* Backward propagation is used for learning (next week)

# Optional Lab 1: Neurons and Layers
Lab 1 Juypter [file](Labs/C2_W1_Lab01_Neurons_and_Layers.ipynb).

# Quiz: 100%
Quiz [file](Quizzes.md#neural-network-model)