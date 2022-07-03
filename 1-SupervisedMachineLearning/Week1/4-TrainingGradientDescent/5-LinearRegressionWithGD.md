* LR model: $f_{w,b}(x)=wx+b$
* Squared error cost function: $J(w,b)=\frac1{2m}\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})^2$
* Gradient Descent Algorithm:
    $$
    \text{repeat until convergence: } \{ \\
    w=w-\alpha \frac\partial{\partial w}J(w,b) \\
    b=b-\alpha \frac\partial{\partial b}J(w,b) \\\}
    $$
    * $\frac\partial{\partial w}J(w,b) = \frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}$
    * $\frac\partial{\partial b}J(w,b) = \frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})$
* Calculus proof of the above formula - [page 72](../Lecture.pdf)
* Final GD algo for LR:
    $$
    \text{repeat until convergence: } \{ \\
    w=w-\alpha\frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)} \\
    b=b-\alpha\frac1m\sum_{i=1}^m(f_{w,b}(x^{(i)})-y^{(i)}) \\\} \\
    \text{NOTE: Updated simultaneously}
    $$
