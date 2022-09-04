# Mean Normalization
* Mean normalization: Changing the values of the ratings to be centered around 0
* How to do it:
    * $\mu$: Vector of average ratings for each movie
    * Editing Ratings: $R_{norm} = R - \mu$ (subtract the \mu value from each rating)
    * Prediction: $w^{(j)} \cdot x^{(i)} + b^{(j)} - \mu_i$, instead of $w^{(j)} \cdot x^{(i)} + b^{(j)}$
* This helps the model to converge faster
* In addition: It helps predict what a user would like, who has not rated anything yet
* Note: Another option is to normalize for the users rather than the movies

# TensorFlow Implementation of Collaborative Filtering
* Gradient Decent Algorithm:
    
    $$\begin{align*}
    &\text{repeat until convergence:} \; \lbrace \\
    & \; \; \;w = w - \alpha \frac{\partial}{\partial w} J(w,b) \\
    & \; \; \; \; \;b = b - \alpha \frac{\partial}{\partial b} J(w,b) \\
    &\rbrace
    \end{align*}$$
* Benefit of TensorFlow: Computing the partial derivative term can be difficult but TF implement is for you
* TF code: Implementing $J = (wx - 1)^2$
    ```python
    # Setup 
    w = tf.Variable(3.0) # Meaning w is a value to be optimized
    x = 1.0
    y = 1.0 # target value
    alpha = 0.01 # learning rate

    iterations = 30
    for iter in range(iterations):
        # Record the history of w values on a tape (auto differentiation)
        with tf.GradientTape() as tape:
            f_wb = w*x
            costJ = (f_wb - y) ** 2
        
        # Compute the gradient of J with respect to w
        [dJdw] = tape.gradient(costJ, [w])
        
        # Update w (one step of gradient descent)
        w.assign_add(-alpha * dJdw)
    ```

    See [page 25](Lecture.pdf) for more on auto differentiation (Auto Diff)
* TF code: Adam optimizer
    ```python
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)
    iterations = 200

    for iter in range(iterations):
        # Record operation for computing the cost
        with tf.GradientTape() as tape:
            cost_value = cofiCostFuncV(X, W, b, Ynorm, R, num_users, num_movies, lambda)
    
    # Automatically compute the gradients of the trainable variables
    grads = tape.gradient( cost_value, [X,W,b] )

    # Update the variables (one step of gradient descent)
    optimizer.apply_gradients( zip(grads, [X,W,b]) )
    ```

    See [page 26](Lecture.pdf) for more on using Adam optimizer.

# Finding Related Items
* When looking at one item, we want to show the user other related items
* Features $x^{(i)}$ of item $i$ are quite hard to interpret. They don't directly correspond to a genre.
* To find items related to item $i$: Find and item $k$ so $x^{(k)}$ is similar to $x^{(i)}$

    $$\sum_{i=1}^n \left( x_l^{(k)} - x_l^{(i)} \right)^2$$

    or in other words:

    $${||x^{(k)} - x^{(i)}||}^2$$
* Problems with Collaborative Filtering:
    * Cold start problem:
        * Ranking new movies that few users have seen
        * Showing reasonable movies to new users
    * Doesn't give an easy way to use side information:
        * Side information (User): Demographic, Age, Gender, etc. (IP address, web-browser, ...)
        * Side information (Item): Genre, Director, Actors, etc.
* [Content based algorithms](3-ContentBasedFiltering.md) solve a lot of these problems

# Quiz: 100%
Quiz [file](Quizzes.md#recommender-systems-implementation-detail)