* Training set has:
    * "input features" - $x$
    * "output features" - $y$
* Process of generating a supervised learning model: (see [page 34](../Lecture.pdf))

    training set (features and targets) $$\downarrow$$
    learning algorithm $$\downarrow$$
    $x \rightarrow f \rightarrow \hat{y}$
* Where the above values correspond with:
    * $x$: feature
    * $f$: hypothesis
    * $\hat{y}$: prediction
* The $f$ hypothesis function:
    * $f_{w,b}(x)=wx+b$
    * One variable linear regression
    * Univariate linear regression
* Question:

    See [page 32](../Lecture.pdf) or Terminology section of [Notes 1](1-LinearRegressionP1.md).
    For linear regression, the model is represented by $f_{w,b}(x)=wx+b$. Which of the following is the output or "target" variable?

    * [ ] $\hat{y}.$
    * [ ] $x$
    * [x] $y$
    * [ ] $m$

    y is the true value for that training example, referred to as the output variable, or “target”.
