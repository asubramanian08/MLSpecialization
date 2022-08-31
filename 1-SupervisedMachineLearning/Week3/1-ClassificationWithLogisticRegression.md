# Motivations
* Linear Regression is not a good algorithm for classification
    * We could draw plot all the data points as (tumor size, 0/1).
    * Then apply linear regression
    * Set the decision boundary to the $x$ value when the height is $0.5$
    * *Problem*: If we add more examples that obviously malignant, the regression line will drastically change. See [page 5](Lecture.pdf).
* Binary classification examples:
    * Email Spam
    * Fraudulent transactions
    * Malignant tumors
* *class* is a category:
    * *positive class*: the false ex. malignant
    * *negative class*: the false ex. benign
* **Decision Boundary**: The line where we change the classification of query
* Question

    Which of the following is an example of a classification task?

    * [ ] Estimate the weight of a cat based on its height.
    * [x] Decide if an animal is a cat or not a cat.

    Correct: This is an example of *binary classification* where there are two possible classes (True/False or Yes/No or 1/0).

# Optional Lab 1: Classification
Lab 1 Jupyter [file](Labs/C1_W3_Lab01_Classification_Soln.ipynb).

# Logistic Regression
* Remember Logistic Regression is a *Classification* algorithm, don't get confused with the name
* Logistic Regression is the largest classification algorithm
* Sigmoid function:
    * Denoted $g(z)$ outputs $0 < g(z) < 1$
    * $g(z) = \frac1{1 + e^{-z}}$
    * See [page 7](Lecture.pdf) for how the function's graph looks
    * As $z$ get larger $g(z)$ approaches 1
    * When $z=0, g(z)=0.5$
* Determining the model: $f_{\vec{w},b}(\vec{x})$
    * Original: $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$
    * Let $z = \vec{w} \cdot \vec{x} + b$
    * Plug this into the sigmoid function: $g(z) = \frac1{1 + e^{-z}}$

    $$f_{\vec{w},b}(\vec{x}) = \frac1{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$
* The model outputs the "probability" that the class is $1$
    * Ex. $f_{\vec{w},b}(\vec{x}) = 0.7$ means $70$% probability
* **Notation** example
    * $x$: "tumor size"
    * $y$: 0/1

        $0$: not malignant

        $1$: malignant
    * $f_{\vec{w},b}(\vec{x}) = P(y=1 | \vec{x}; \vec{w},b)$: $f$ is the probability that $y$ is $1$
        * **Note**: $P(y=0) + P(y=1) = 1$
* Question

    See [page 8](Lecture.pdf). Recall the sigmoid function is $g(z) = \frac1{1 + e^{-z}}$. If z is a large negative number then:

    * [x] $g(z)$ in near zero
    * [ ] $g(z)$ in near negative one (-1)

    Say $z=-100$. $e^{-z}$ is then $e^{100}$, a really big positive number. So, $g(z) = \frac1{1 + \text{a big positive number}}$ or about $0$

# Optional Lab 2: Sigmoid function and Logistic Regression
Lab 2 Jupyter [file](Labs/C1_W3_Lab02_Sigmoid_function_Soln.ipynb).

# Decision Boundary
* Recap $f_{\vec{w},b}(\vec{x})$
    1. Step 1: $z = \vec{w} \cdot \vec{x} + b$
    2. Step 2: $g(z) = \frac1{1 + e^{-z}}$

    So: $f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) = \frac1{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$
    * Keep in mind: $f_{\vec{w},b}(\vec{x}) = P(x=1|\vec{x};\vec{w},b)$
* We can put some kind of threshold here for the decision boundary
    * If $f_{\vec{w},b}(\vec{x}) \ge 0.5$ then $\hat{y}=1$ otherwise $\hat{y}=0$
    * This happens when $g(z) \ge 0.5$ or $z \ge 0$
    * In conclusion if $\vec{w} \cdot \vec{x} + b \ge 0$ the model predicts YES or $1$
* Visualizing the decision boundary:
    * **Decision boundary**: The line $z = \vec{w} \cdot \vec{x} + b = 0$
    * Linear boundary: See [page 12](Lecture.pdf)
    * Non-linear boundary: See [page 13-14](Lecture.pdf)

        This for example could look like a circle or ellipse.
    * **Take away**: Since the decision boundary can take many shapes it can fit a lot of different data
* Question

    Let’s say you are creating a tumor detection algorithm. Your algorithm will be used to flag potential tumors for future inspection by a specialist. What value should you use for a threshold?

    * [ ] High, say a threshold of 0.9?
    * [x] Low, say a threshold of 0.2?

    You would not want to miss a potential tumor, so you will want a low threshold. A specialist will review the output of the algorithm which reduces the possibility of a ‘false positive’. The key point of this question is to note that the threshold value does not need to be 0.5.

# Optional Lab 3: Decision Boundary
Lab 3 Jupyter [file](Labs/C1_W3_Lab03_Decision_Boundary_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#classification-with-logistic-regression)