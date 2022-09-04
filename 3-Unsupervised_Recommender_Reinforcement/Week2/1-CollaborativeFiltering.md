# Making Recommendations
* We are going to discuss recommender systems
    * Studied a lot in academia
    * The number of use cases is very vast
    * Aka: This is a field that can grow a lot
* Example: Netflix recommending movies for you to watch
* Problem: Predicting movie ratings
    * $n_u$: number of users (people given ratings)
    * $n_m$: number of items (movies)
    * $r(i, j)$: If user $j$ rated movie $i$ (1 if yes, 0 if no)
    * $y^{(i,j)}$: Rating of user $j$ to movie $i$ (from 0 to 5)
        
        $y^{(i,j)}$ is only defined if $r(i,j) = 1$
    * Goal: Determine what rating a certain user would rate a movie

        That way we can determine what movies to recommend to a user

# Using Per-Item Features
* Additional Data:
    * We have features about each movie
    * There are ratings for have much a movies fits into a genre: (0-1)

        Example: Romance, Action, Comedy, etc.
    * $n$: Number of genres features we have
    * $x^{(i)}$ Vector of features for movie $i$, one for each genre
* Linear Regression (ish) model:
    * Predict rating of movie $i$ for user $j$ as: $w^{(j)} \cdot x^{(i)} + b^{(j)}$
    * Learn the values of $w^{(j)}$ and $b^{(j)}$
    * Note this is similar to linear regression. The difference is we are fitting a different model for each user.
    * $m^{(j)}$: Number of movies user $j$ has rated
    * Cost function:
        $$\min_{w^{(j)}b^{(j)}} J(w^{(j)}, b^{(j)}) = \frac12 \sum_{i:r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac\lambda2 \sum_{k=1}^n \left( w_k^{(j)} \right)^2$$
    
        Note: For recommender systems we don't need to divide by $m^{(j)}$ since it is just a constant
    * To learn parameters $w^{(j)}, b^{(j)}$ for all $1 \le j \le n_u$:
        $$J(w, b) = \frac12 \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac\lambda2 \sum_{j=1}^{n_u} \sum_{k=1}^n \left( w_k^{(j)} \right)^2$$
        
        We can use gradient descent to learn the parameters.

# Collaborative Filtering Algorithm
* Now let's try and predict the ratings without "knowing" the genre of each movie
* If we had the values for $w^{(j)}, b^{(j)} \forall j$, then we could guess what the genre ratings of a movie would be.

    For example let:
    
    $$w^{(1)} = \begin{bmatrix} 5 \\ 0 \end{bmatrix}, \ \ \ \ w^{(2)} = \begin{bmatrix} 5 \\ 0 \end{bmatrix} \ \ \ \ w^{(3)} = \begin{bmatrix} 0 \\ 5 \end{bmatrix} \ \ \ \ w^{(4)} = \begin{bmatrix} 0 \\ 5 \end{bmatrix} \\ b^{(1)} = 0, \ \ \ \ \ \ b^{(2)} = 0, \ \ \ \ \ \ b^{(3)} = 0, \ \ \ \ \ \ b^{(4)} = 0 $$

    Since $w^{(j)} \cdot x^{(i)} + b^{(j)}$:
    $$w^{(1)} \cdot x^{(1)} \approx 5 \\ w^{(2)} \cdot x^{(1)} \approx 5 \\ w^{(3)} \cdot x^{(1)} \approx 5 \\ w^{(4)} \cdot x^{(1)} \approx 5 \\ \ \ \ \ \ \ \ \ \ \ \implies x^{(1)} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$
* This means we can "guess" the features $x^{(i)} \forall i$
* Cost function

    $$J(x^{(i)}) = \frac12 \sum_{j:r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac\lambda2 \sum_{k=1}^n \left( w_k^{(i)} \right)^2$$
* Collaborative Filtering Algorithms:
    * Cost functions:

        Learn parameters $w^{(j)}, b^{(j)}$ for all $1 \le j \le n_u$:
        $$\min_{w,b} \frac12 \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac\lambda2 \sum_{j=1}^{n_u} \sum_{k=1}^n \left( w_k^{(j)} \right)^2$$

        Learn parameters $x^{(i)}$ for all $1 \le i \le n_m$:
        $$\min_x \frac12 \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac\lambda2 \sum_{i=1}^{n_m} \sum_{k=1}^n \left( x_k^{(i)} \right)^2$$

        Combine the two:
        $$\min_{\substack{w^{(1)}, \dots, w^{(n_u)} \\ b^{(1)}, \dots, b^{(n_u)} \\ x^{(1)}, \dots, x^{(n_m)}}} J(w,b,x) = \frac12 \sum_{(i,j):r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac\lambda2 \sum_{j=1}^{n_u} \sum_{k=1}^n \left( w_k^{(j)} \right)^2 + \frac\lambda2 \sum_{i=1}^{n_m} \sum_{k=1}^n \left( x_k^{(i)} \right)^2$$
    * Use gradient descent to minimize the cost function $J(w, b, x)$. See [page 14](Lecture.pdf).

# Binary Labels: Favs, Likes, and Clicks
* Change the collaborative filtering algorithm to work with binary labels
    * Most input come in this form: 
        * "I like this movie" or "I don't like this movie"
        * fav/like an item
        * click/didn't click an item
        * Seen and items for more than 30 seconds

    * Goal: Predict whether a user will like a movie that they have not yet watched
* The change is to use a logistic regression model instead of a linear regression model

    This is similar to the switch between regression and binary classification
    * Previously: predict $y^{(i,j)}$ as $w^{(j)} \cdot x^{(i)} + b^{(j)}$
    * For binary labels: predict $P(y^{(i,j) = 1})$, given by $g(w^{(j)} \cdot x^{(i)} + b^{(j)})$ where $g(z) = \frac1{1 + e^{-z}}$
* Cost function:

    * Previous cost function:
        $$J(w,b,x) = \frac12 \sum_{(i,j):r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac\lambda2 \sum_{j=1}^{n_u} \sum_{k=1}^n \left( w_k^{(j)} \right)^2 + \frac\lambda2 \sum_{i=1}^{n_m} \sum_{k=1}^n \left( x_k^{(i)} \right)^2$$

    * Loss for binary labels $y^{(i,j)}$: $f_{w,b,x}(x) = g(w^{(j)} \cdot x^{(i)} + b^{(j)})$
        $$L(f_{w,b,x}(x), y^{(i,j)}) = -y^{(i,j)}\log \left( f_{w,b,x}(x) \right) - (1-y^{(i,j)})\log \left( 1-f_{w,b,x}(x) \right)$$

    * Cost function for binary labels:
        $$J(w, b, x) = \sum_{(i,j):r(i,j)=1} L(f_{w,b,x}(x), y^{(i,j)})$$

    * Note: $f_{w,b,x}(x) = g(w^{(j)} \cdot x^{(i)} + b^{(j)})$

# Quiz: 100%
Quiz [file](Quizzes.md#collaborative-filtering)