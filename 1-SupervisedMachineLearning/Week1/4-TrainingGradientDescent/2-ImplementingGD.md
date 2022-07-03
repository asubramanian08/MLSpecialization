* Algorithms:
    1. $tmp_w = w-\alpha \frac\partial{\partial w}J(w,b)$
    2. $tmp_b = b-\alpha \frac\partial{\partial b}J(w,b)$
    3. $w=tmp_w$
    4. $b=tmp_b$
    3. Repeat until convergence: not much of a difference in $J(w,b)$
* Understanding the algo
    * $\alpha$: The learning rate (how big of a step you take)
    * Simultaneously update values $w$ and $b$
* Question:

    See [page 62](../Lecture.pdf). Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J. What does this update statement do? (Assume $\alpha$ is small.) $$w=w-\alpha\frac{\partial J(w,b)}{\partial w}$$

    * [ ] Checks whether $w$ is equal to $w-\alpha\frac{\partial J(w,b)}{\partial w}$
    * [x] Updates parameter $w$ by a small amount

    This updates the parameter by a small amount, in order to reduce the cost $J$.