# Using Multiple Decision Trees
* Using one decision tree will make the algorithm highly sensitive to changes in the data
* One solution is to build multiple trees: A tree ensemble
* Prediction using a tree ensemble:
    * Have each tree "vote" on whether it thinks the item is a cat or not
    * After every tree have voted, pick the answer with the most number of votes

# Sampling with Replacement
* Our goal: How do we create a tree ensemble. How do we generate multiple different trees?
* Sampling with replacement:
    * Pick a random sample
    * Put it back in the bag (so I might pick it again)
    * Repeat the above steps
* Using "Sampling with replacement" we can generate multiple random training sets
    * "Sampling with replacement" $m$ times to general a random training training set
    * This training set might have repeat elements but that is okay
    * Note that $m = $ number of training examples that I have

# Random Forest Algorithm
* **Bagged decision tree**: Steps for generating an ensemble of trees
    * Given training set of size $m$
    1. For $b = 1$ to $B$: ($B$ might be $64 \le B \le 128$)
        1. Use sampling with replacement to generate a new training set with $m$ examples
        2. Train a decision tree on the new dataset
* **Random forest algorithm**: Steps for generating an ensemble of trees
    * Given training set of size $m$
    1. For $b = 1$ to $B$: ($B$ might be $64 \le B \le 128$)
        1. Use sampling with replacement to generate a new training set with $m$ examples
        2. Train a decision tree on the new dataset

            At each node (when trying to pick what feature to split by): Pick a random subset of $k$ features that I am allowed to split by. If there are $n$ total features then set $k = \sqrt{n}$.

# XGBoost
* **XGBoost** is the most commonly used algorithm for decision tree ensembles.
* Boosted Trees Intuition: (Deliberate Practice - Focus on what is not working)
    * Given training set of size $m$
    1. For $b = 1$ to $B$: ($B$ might be $64 \le B \le 128$)
        1. Use sampling with replacement to generate a new training set with $m$ examples

            Instead of picking all examples with the same probability. Pick the examples that the previous decision trees have not been doing well on.
        2. Train a decision tree on the new dataset
* XGBoost (eXtreme Gradient Boosting):
    * The most popular approach to boosting
    * Open source
    * Fast
    * Good choice of splitting and stop splitting criteria
    * Build in regularization (prevent overfitting)
    * Using in many ML competitions (Kaggle)
* XGBoost code

    Classification:
    ```python
    from xgboost import XGBClassifier

    model = XGBClassifier()

    model.fit(X_train, y_train)
    y_pred = mode.predict(X_test)
    ```

    Regression:
    ```python
    from xgboost import XGBRegressor

    model = XGBRegressor()

    model.fit(X_train, y_train)
    y_pred = mode.predict(X_test)
    ```

# When to use Decision Trees
* Decision Trees and Tree Ensembles
    * Work well on tabular (structured) data - data looks like a spread sheet
    * Not recommended on unstructured data (images, audio, text, ...)
    * Fast to Train
    * May be understandable to a human
* Neural Networks
    * Works well on all data: tabular and unstructured
    * Slower than a decision tree
    * Works with transfer learning
    * Its easier to string together multiple neural networks

        Train multiple neural networks are once

# Quiz: 100%
Quiz [file](Quizzes.md#tree-ensembles)