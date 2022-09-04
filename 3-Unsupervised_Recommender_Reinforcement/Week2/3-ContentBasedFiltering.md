# Collaborative Filtering vs Content-Based Filtering
* How either algorithm recommends movies:
    * Collaborative Filtering: "People who liked this also liked..."
    * Content-Based Filtering: "Movies similar to this one are...", taking into account the features of the movie and user
* Example Features
    * User Features: ($x_u^{(j)}$ for user $j$) - Age, Gender, Location, Movies Watched, Rating per Genre, ...
    * Movie Features: ($x_m^{(i)}$ for user $i$) - Year, Genre, Reviews, Average Rating, ...
    * Note: For Location, Genre, ... we can use **one-hot encoding**
* How content-based filtering makes matches:
    * Previous version of predicting the rating for a movie $i$ by a user $j$: $w^{(j)} \cdot x^{(i)}$
    * $V_u^{(j)} = w^{(j)}$, where $V_u^{(j)}$ is computed from $x_u^{(j)}$
    * $V_m^{(i)} = x^{(i)}$, where $V_m^{(i)}$ is computed from $x_m^{(i)}$
    * Predicted rating for movie $i$ by user $j$: $V_u^{(j)} \cdot V_m^{(i)}$
    * NOTE: $V_u^{(j)}$ and $V_m^{(i)}$ are vectors of the same length

# Deep Learning for Content-Based Filtering
* User network:
    * Takes input $X_u$
    * Outputs $V_u$ (might have length 32)
* Movie network:
    * Takes input $X_m$
    * Outputs $V_m$ (same length as $V_u$)
* Prediction: $g(V_u^{(j)} \cdot V_m^{(i)})$ as the probability of a user liking a movie
* We can combine the two networks into one network: By joining the outputs of the two networks, we can get a single vector $V_u \cdot V_m$ that we can use to predict the rating

    See [page 36](Lecture.pdf)
* Cost function
    $$J = \sum_{(i,j):r(i,j)=1} \left( V_u^{(j)} \cdot V_m^{(i)} - y^{(i,j)} \right)^2 + \text{NN regularization term}$$
* $V$ vectors and their meanings:
    * $V_u^{(j)}$ is a vector of length 32 that describes the user $j$ with features $x_u^{(j)}$
    * $V_m^{(i)}$ is a vector of length 32 that describes the movie $i$ with features $x_m^{(i)}$
* Finding similar movies to movie $i$: ${||V_m^{(i)} - V_m^{(k)}||}^2$

    Note: This can be precomputed ahead of time, so it doesn't need to be computed at the time of a user query

# Recommending from a Large Catalogue
* This section about how to effectively use content-based filtering if we have many movies. (Since the it requires precomputing the similarity between all movies $O(N^2)$)
* Two step process: Retrieval and Ranking
* Retrieval: Create a general list of movies that might be what the user is looking for
    * General large plausible list
        * Find the 10 most similar movies to the 10 movies the user last watched (${||V_m^{(i)} - V_m^{(k)}||}^2$)
        * Add the top 10 movies for the 3 most viewed genres of the user
        * Top 20 movies in the country
    * Remove duplicates / movies the user has already watched
* Ranking: Rank the movies in the list
    * Given the retrieved list, rank the movies by how much the user would like them
    * Rank the movies by feeding it into the Neural Network and sorting based on the "predicted rating"
    * Display ranked list to user
    * One more optimization: Precompute $V_m$ and calculate $V_u$ on the fly. Take the dot product and that is the prediction
* How many items to retrieve?
    * Retrieving more items will increase the accuracy of the ranking but decrease the speed
    * To analyze the tradeoff: Carry offline experiments with different numbers of retrieved items to see if we get more relevant recommendations

# Ethical use of Recommender Systems
* Ex. Advertizing: A recommender system my promote a product that is not good for the user
* Ex. Promotion: A recommender system may display the most profitable items to the user rather than the most relevant items
* Maximizing user engagement: This tends to spread conspiracy theories and fake news
* See more examples on [page 46](Lecture.pdf)

# TensorFlow Implementation of Content-Based Filtering
```python
# Create the user and movie networks
user_NN = tf.keras.models.Sequential([
tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(32)
])
item_NN = tf.keras.models.Sequential([
tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(32)
])

# Setup the user network: Define the input, and the output
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# Setup the movie network: Define the input, and the output
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# Define additional variables to later use
output = tf.keras.layers.Dot(axes=1)([vu, vm])
model = Model([input_user, input_item], output)
cost_fn = tf.keras.losses.MeanSquaredError()
```

See [page 48](Lecture.pdf).

# Quiz: 100%
Quiz [file](Quizzes.md#content-based-filtering)