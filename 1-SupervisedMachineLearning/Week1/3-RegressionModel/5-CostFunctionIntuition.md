* model: $f_{w,b}(x)=wx+b$
* parameters: $w,b$
* cost function: $J(w,b)=\frac{1}{{2m}}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})^2$
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