* Simpler algo for GD: $w=w-\alpha \frac\partial{\partial w}J(w)$
* See [page 65](../Lecture.pdf)
* Draw a tangent line to $J(w)$ and find its slope (derivative)
* If the slope is positive we end of decreasing $w$ and the same implies for the negative slope
* Question:

    See [page 62](../Lecture.pdf). Gradient descent is an algorithm for finding values of parameters w and b that minimize the cost function J. 
    
    $$
    \text{repeat until convergence: } \{ \\
    w=w-\alpha \frac\partial{\partial w}J(w,b) \\
    b=b-\alpha \frac\partial{\partial b}J(w,b) \\\}
    $$

    Assume the learning rate $\alpha$ is a small positive number. When $\frac{\partial J(w,b)}{\partial w}$ is a positive number (greater than zero) -- as in the example in the upper part of the slide shown above -- what happens to $w$ after one update step?

    * [ ] It is not possible to tell if $w$ will increase or decrease. 
    * [ ] $w$ increases
    * [x] $w$ decreases
    * [ ] $w$ stays the same

    The learning rate $\alpha$ is always a positive number, so if you take W minus a positive number, you end up with a new value for W that is smaller