* The first model: Linear Regression (fitting a straight line to data)
* Ex. Predicting a price based on the size of a house
* This is supervised learning since we train with the price and size
* Specifically the type of supervised learning is a regression and not a classification
* The data table might look like:
    
    size in feet $^2$ | price in $\$1000$'s
    ------------------|--------------------
    2104 | 400
    1416 | 232
    1534 | 315
    852 | 178
    ... | ...
    3210 | 870

    Each row in the table is one data point
* Terminology
    * **Training Set**: Data used to train the model

        row # | $x$: size in feet $^2$ | $y$: price in $\$1000$'s
        ------|------------------------|-------------------------
        (1) | 2104 | 400
        (2) | 1416 | 232
        (3) | 1534 | 315
        (4) | 852 | 178
        ... | ... | ...
        (47) | 3210 | 870
    * **$x$**: "input" variable / feature
    * **$y$**: "output" variable / "target" variable
    * **$m$**: number of training examples
    * **$(x, y)$**: one training example
    * **$(x^{(i)}, y^{(i)})$**: ith training example
    * ex. $x^{(1)} = 2104$ and $y^{(1)} = 400$

        $(x^{(1)}, y^{(1)}) = (2104, 400)$
    