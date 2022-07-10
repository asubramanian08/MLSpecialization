# Linear Regression Model Part 1
* The first model: Linear Regression (fitting a straight line to data)
* Ex. Predicting a price based on the size of a house
* This is supervised learning since we train with the price and size
* Specifically the type of supervised learning is a regression and not a classification
* The data table might look like:

    size in feet $^2$ | price in $1000's
    ------------------|--------------------
    2104 | 400
    1416 | 232
    1534 | 315
    852 | 178
    ... | ...
    3210 | 870

    Each row in the table is one data point
* Terminology
    * **Training Set**: Data used to train the model

        row # | $x$: size in feet $^2$ | $y$: price in $1000's
        ------|------------------------|-------------------------
        (1) | 2104 | 400
        (2) | 1416 | 232
        (3) | 1534 | 315
        (4) | 852 | 178
        ... | ... | ...
        (47) | 3210 | 870
    * **$x$**: "input" variable / feature
    * **$y$**: "output" variable / "target" variable
    * **$m$**: number of training examples
    * **$(x, y)$**: one training example
    * **$(x^{(i)}, y^{(i)})$**: ith training example
    * ex. $x^{(1)} = 2104$ and $y^{(1)} = 400$

        $(x^{(1)}, y^{(1)}) = (2104, 400)$

# Linear Regression Model Part 2
* Training set has:
    * "input features" - $x$
    * "output features" - $y$
* Process of generating a supervised learning model: (see [page 34](../Lecture.pdf))
    1. training set (features and targets)
    2. learning algorithm
    3. $x \rightarrow f \rightarrow \hat{y}$
* Where the above values correspond with:
    * $x$: feature
    * $f$: hypothesis
    * $\hat{y}$: prediction
* The $f$ hypothesis function:
    * $f_{w,b}(x)=wx+b$
    * One variable linear regression
    * Univariate linear regression
* Question:

    See [page 32](../Lecture.pdf) or Terminology section of the [first notes](#linear-regression-model-part-1).
    For linear regression, the model is represented by $f_{w,b}(x)=wx+b$. Which of the following is the output or "target" variable?

    * [ ] $\hat{y}.$
    * [ ] $x$
    * [x] $y$
    * [ ] $m$

    y is the true value for that training example, referred to as the output variable, or “target”.

# Optional Lab 3: Model Representation
Lab 3 Juypter [file](Labs/C1_W1_Lab03_Model_Representation_Soln.ipynb).

# Cost Function Formula
* Model: $f_{w,b}(x)=wx+b$
* $w,b$: parameters/coefficients
    * This is what we need to edit when training the model
    * These variables are like the slope and y-intercept respectively
* When a line "fits" the training set, the regression is close data points
* Determining squared error cost function
    * Error: $\hat{y}^{(i)}-y^{(i)}$
    * Cost function: $J(w,b)=\frac{\sum_{i=1}^m(\hat{y}^{(i)}-y^{(i)})^2}{2m}$
    * NOTE: It is divided by $2m$ instead of $m$ to make later calculations easier
    * Expanding $\hat{y}^{(i)}$: $J(w,b)=\frac{\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})^2}{2m}$
* Question:

    See [page 38](../Lecture.pdf). The cost function used for linear regression is: $$J(w,b)=\frac{1}{{2m}}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})^2$$ Which of these are the parameters of the model that can be adjusted?

    * [x] $w$ and $b$
    * [ ] $f_{w,b}(x^{(i)})$
    * [ ] $w$ only, because we should choose b=0
    * [ ] $\hat{y}$

    w and b are parameters of the model, adjusted as the model learns from the data. They’re also referred to as “coefficients” or “weights”

# Cost Function Intuition
* model: $f_{w,b}(x)=wx+b$
* parameters: $w,b$
* cost function: $J(w,b)=\frac{1}{2m}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})^2$
* *Our goal*: $\min_{w,b} J(s,b)$
* Understanding how cost function acts:
    * See [page 41-44](../Lecture.pdf)
    * We simplify our cost function to only be a factor of $w$ (so $f_w(x)=wx$)
    * Using training set  $\{(1,1), (2,2), (3,3)\}$
    * By graphing $w$ vs. $J(w)$, the graph is a parabola - [page 43](../Lecture.pdf)
* Question:

    See [page 43](../Lecture.pdf). When does the model fit the data relatively well, compared to other choices for parameter w?

    * [ ] When w is close to zero.
    * [ ] When $f_w(x)$ is at or near a minimum for all the values of x in the training set.
    * [ ] When $x$ is at or near a minimum.
    * [x] When the cost $J$ is at or near a minimum.

    When the cost is relatively small, closer to zero, it means the model fits the data better compared to other choices for w and b.

# Visualizing the Cost Function
* Recap
    * model: $f_{w,b}(x)=wx+b$
    * parameters: $w,b$
    * cost function: $J(w,b)=\frac{1}{2m}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})^2$
    * Objective: $\min_{w,b} J(s,b)$
* See the 2D "soup bowl looking" group of $J(w,b)$ on [page 48](../Lecture.pdf)
* We can use a contour plot to visualize the 2D graph. Each ring in the contour has equivalent cost function values. See [page 52](../Lecture.pdf)

# Visualization Examples
In this video Mr. Ng gives us examples of points on the cost function contour graph. He locates a point and graphs its line onto the training data. This is just to gain a better understanding of contour graphs. See [page 54-57](../Lecture.pdf)

# Optional Lab 4: Cost Function
Lab 4 Jupyter [file](Labs/C1_W1_Lab04_Cost_function_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#regression-model)