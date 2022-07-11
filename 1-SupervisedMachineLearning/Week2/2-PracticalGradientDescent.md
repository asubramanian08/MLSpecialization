# Feature Scaling Part 1
* Let's look at an example scenario:
    * $x_1$: size (feet$^2$), $300 \le x_1 \le 2000$
    * $x_2$: # bedrooms, $0 \le x_1 \le 5$
    * The  $\vec{w} $ might be $[w_1 = 0.1 \space \space w_2 = 50]$
    * **Takeaway**: When the size of the feature is small its weight is larger.
* See [page 18](Lecture.pdf): The contour plot will be elliptical and might cause GD to run slowly
* By scaling our features we get more circle like figures on the contour plot

# Feature Scaling Part 2
* Using our examples:

    $300 \le x_1 \le 2000$

    $0 \le x_2 \le 5$
* Divide by the upper bound
* Mean normalization:
    1. Find the average of each feature ($\mu$)
    2. Let $x_i = \frac{x_i - \mu_i}{\max_{x_i} - \min_{x_i}}$
    3. This will center all new data points around the origin
* Z-score normalization
    * $\sigma$: standard divination
    * $\mu$: average value
    * $x_i = \frac{x_i - \mu_i}{\sigma_i}$
* Feature Scaling goal
    * We typically want $-1 \le x_j \le 1$ for all $x_j$
    * $-100 \le x_j \le 100$ is too large
    * $-0.00001 \le x_j \le 0.00001$ is too small
* Question

    See [page 21](Lecture.pdf). Which of the following is a valid step used during feature scaling?

    * [ ] Multiply each value by the maximum value for that feature
    * [x] Divide each value by the maximum value for that feature

    By dividing all values by the maximum, the new maximum range of the rescaled features is now 1 (and all other rescaled values are less than 1).

# Checking Gradient Descent for Convergence
* We can plot the cost function $J(\vec{w},b)$ to the # of iterations
* Looking at this graph shows how $J$ is changing. That way we can see when GD coverages
* If the graph increases likely $\alpha$ is too large
* Automatic convergence test: If $\Delta J(\vec{w},b) < \epsilon$ the GD has converged
* See [page 27](Lecture.pdf)

# Choosing the Learning Rate
* If $\alpha$ is too large it might not coverage, $J(\vec{w}, b)$ might increase
* If $\alpha$ is too small the program might take a long time to run
* There could be a bug in the code causing the graph of  $J(\vec{w}, b) $ too look weird. To test this slowly increase  $\alpha $ (start with 0.01, 0.03, 0.1, ...) until GD diverges. Pick an $\alpha$ that is just small then the value it diverges at.
* Question

    See [page 29](Lecture.pdf). You run gradient descent for 15 iterations with  $\alpha = 0.3 $ and compute  $J(\vec{w}) $ after each iteration. You find that the value of $J(w) $ increases over time.  How do you think you should adjust the learning rate  $\alpha $?

    * [ ] Try running it for only 10 iterations so $J(w)$ doesn't increase much.
    * [x] Try a small value of  $\alpha $ (say  $\alpha = 0.1$)
    * [ ] Keep running it for additional iterations
    * [ ] Try a large value of  $\alpha $ (say  $\alpha = 1.0$)

    Since the cost function is increasing, we know that gradient descent is diverging, so we need a lower learning rate.

# Optional Lab 3: Feature Scaling and Learning Rate
Lab 3 Jupyter [file](Labs/C1_W2_Lab03_Feature_Scaling_and_Learning_Rate_Soln.ipynb).

# Feature Engineering
* Example: Prediction house price

    $x_1$: Frontage

    $x_2$: Depth

    * We can design another feature  $x_3 $ that is the area - frontage $\times$ depth
* **Concept**: We can add new features by combining other features and getting a more accurate prediction

    Look at the above example of adding $x_3$ as the area.
* Question

    See [page 32](Lecture.pdf). If you have measurements for the dimensions of a swimming pool (length, width, height), which of the following two would be a more useful engineered feature?

    * [ ] $\text{length} + \text{width} + \text{height}$
    * [x] $\text{length} + \text{width} \times \text{height}$

    The volume of the swimming pool could be a useful feature to use.  This is the more useful engineered feature of the two.

# Polynomial Regression
* This is a form of [Feature Engineering](#feature-engineering) where we can add features  $x_2 $ as  $(x_1)^2 $.
* We can also use $\sqrt{x_1}$ as a feature
* **Note**: [Feature scaling](#feature-scaling-part-2) becomes increasingly important here

# Optional Lab 4: Feature Engineering and Polynomial Regression
Lab 4 Jupyter [file](Labs/C1_W2_Lab04_FeatEng_PolyReg_Soln.ipynb).

# Optional Lab 5: Linear Regression with scikit-learn
Lab 5 Jupyter [file](Labs/C1_W2_Lab05_Sklearn_GD_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#gradient-descent-in-practice)