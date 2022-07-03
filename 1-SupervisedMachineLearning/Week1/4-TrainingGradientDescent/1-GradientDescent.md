* Gradient Descent is an algorithm we can use to determine the values of $w$ and $b$.
* We have some function $J(w,b)$, this could be any function not just linear regression
* Gradient Descent outline:
    * Pick some values for $w,b$
    * Change $w,b$ to find a smaller cost function
    * Repeat the above step until we are at a local min
* Intuition: Keep taking baby steps down hill until we are at the very bottom
* NOTE: GD only finds the *local min* and follows that path of least resistance