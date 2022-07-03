* $w=w-\alpha \frac\partial{\partial w}J(w,b)$
* If the learning rate is too small: the algo will take too long
* If the learning rate is too large: we can overshoot the minimum (fail to converge)
* When we are exactly at the local min, the slope is 0 and our pos won't change
* Learning rate $\alpha$ is controls how big of a step we take in gradient descent