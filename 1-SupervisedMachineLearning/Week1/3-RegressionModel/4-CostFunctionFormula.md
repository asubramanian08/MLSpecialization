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
