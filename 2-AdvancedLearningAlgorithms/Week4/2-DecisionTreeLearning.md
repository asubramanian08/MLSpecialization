# Measuring Purity
* **Entropy**: Measure of impurity
* $p_1 = $ fraction of examples that are cats
* Let $H()$ be the entropy function
    
    Its highest at $H(0.5) = 1$ and lowest at $H(0) = H(1) = 0$.

    See [page 25](Lecture.pdf) for the graph of the entropy function.
* Examples:

    $$p_1 = 0/6 / H(p_1) = 0$$
    $$p_1 = 2/6 / H(p_1) = 0.92$$
    $$p_1 = 3/6 / H(p_1) = 1$$
    $$p_1 = 5/6 / H(p_1) = 0.65$$
    $$p_1 = 6/6 / H(p_1) = 0$$
* Let $p_0 = 1- p_1$, which is the fraction of examples that are not cats
* Entropy Function:

    $$H(p_1) = -p_1 \log_2(p_1) - p_0\log_2(p_0)$$
    $$H(p_1) = -p_1 \log_2(p_1) - (1 - p_1)\log_2(1 - p_1)$$

    Note: In this case let $0 \log(0) = 0$, even though it is actually undefined.

# Choosing a Split: Information Gain
* We want to pick the feature that reduces entropy ($H(p_1)$) the most
    * **Information Gain**: This is the values assigned by reduction of entropy
* Example: Splitting by face shape:
    
    * Face Shape -> Round: $p_1 = 4/7 / H(0.57) = 0.99$
    * Face Shape -> Not Round: $p_1 = 1/3 / H(0.33) = 0.92$
    * Information Gain of Face Shape: $H(0.5) - \left( \frac{7}{10} H(0.57) + \frac{3}{10} H(0.33) \right)$
* Calculating the "Information Gain":
    * Let $p_1^\text{root}$, $p_1^\text{left}$, and $p_1^\text{right}$ be the fraction of examples from the root, left, and right (respectively) that contains cats.
    * Let $w^\text{left}$ and $w^\text{right}$ be the fraction of examples that went to the left or right respectively.
    * Information Gain $ = H(p_1^\text{root}) - \left( w^\text{left} H(p_1^\text{left}) + w^\text{right} H(p_1^\text{right}) \right)$
* Pick the split that creates that **largest information gain**.

# Putting it Together
* Full Process of building a decision tree
    1. Start with all examples at the root node
    2. Calculate the information gain for all features. Pick the one with the highest information gain.
    3. Split dataset according to the selected feature. Create the left and right branches.
    4. Run step 2 on the left and right branches until the stopping criteria is met:
        * When the node 100% of a class
        * When the maximum depth is reached
        * When the information gain is below a threshold
        * When the number of examples if below a threshold
* See the examples of **recursive splitting** - see [pages 33-53](Lecture.pdf)
    * Notice: The way you build a decision tree from the root is the same as building it from the left or right.
    * This is a recursive algorithm!

# Using One-Hot Encoding for Categorical Features
* What do you do this there a multiple possible values a feature can take:
    * Option 1: Instead of creating a left and right sub-branches. Create $k$ different sub-branches, each representing a different value of the feature.
    * Option 2: **One hot encoding**. Divide the $k$ different feature values and into $k$ new features. Each will represent whether or not a certain feature value is present.

        For example: If the animal can have pointy, floppy, or oval ears. Create 3 new features the say whether or not the animal have pointy ear, floppy ears, and whether or not it has over ears.
* Note: This one hot encoding trick, can also work in **neural networks**!

# Continues Valued Features
* Let's say we have an additional feature: The weight of a cat.
* How do we transfer this continues feature into a decision problem

    We will **threshold** the value. For example if the weight $\ge 8$ lbs.
* How do you determine where the threshold should be:

    Pick whatever threshold **maximizes** the information gain.

# Regression Trees (Optional)
* Now we are going to transform decision trees to solve regression problems
* Problem: Given a bunch of information about an animal, predict its weight
* What values should be put in a leaf node:

    If the stopping criteria is met, a leaf node's value will be set to the average weights of all its given training examples.
* Choosing a split:
    * Rather than reducing entropy, we want to **reduce the variance**.
    * We can calculate the **information gain** $ = v^\text{root} \left( w^\text{left} \times v^\text{left} + w^\text{right} \times v^\text{right} \right)$.
    * Choose the split will the largest information gain.

# Quiz: 100%
Quiz [file](Quizzes.md#decision-tree-learning)