# What is Machine Learning
* Arthur Samuel: Machine learning is "the field of study that givens computers the ability to learn without being explicitly programmed."
    * Arthur created a checkers program train using many game plays
    * Eventually the program was able to beat Arthur himself
* Question

    If Arthur Samuel's checkers-playing program had been allowed to play only 10 games (instead of tens of thousands games) against itself, how would this have affected its performance?

    * [ ] Would have made it better
    * [x] Would have made it worse

    That's right! Please continue the video to learn more about why.
* Machine Learning Algorithms
    * Supervised learning : course 1, 2

        Used most in real world applications (+ improvements)
    * Unsupervised learning : course 3
    * Recommender systems : course 3
    * Reinforcement learning : course 3

    This course will also teach how to apply these tools. What are the best practices?

# Supervised Learning Part 1
* Supervised learning algorithms
    * $x \to y$ where $x$ is input and $y$ is the output label
    * Learns from being given the correct answers
* For example:

    Input(X) | Output(Y) | Application
    ---------|-----------|------------
    email | spam?(0/1) | spam filtering
    audio | text transcript | speech recognition
    English | Spanish | Translator
    ad,user_info | click?(0/1) | ads
    image | loc of other cars | self-driving cars
    image of phone | defect?(0/1) | visual inspection
* Regression can use supervised learning
    * Regression data is the $(x,y)$ point
    * $x$ is the input
    * $y$ is the labeled output

# Supervised Learning Part 2
* Classification algorithms
    * Ex. building a system to determine breast cancer ?(0/1)
    * Plot as: $x$ is the tumor size and $y$ is 0/1 if the tumor is malignant
    * NOTE: Can predict more than one catagories or classes of items
    * Classes or groups a limited number of catagories unlike regression
    * NOTE: We can also use multiple inputs

        For example: Age and Tumor size could be two factors
* 2 Major types of supervised learning
    1. Regression:
        * Predict a number
        * Infinite possible outputs
    2. Classification
        * Predicts catagories
        * Limited possible outputs
* Question

    Supervised learning is when we give our learning algorithm the right answer $y$  for each example to learn from.  Which is an example of supervised learning?

    * [x] Spam filtering
    * [ ] Calculating the average age of a group of customers.

    For instance, emails labeled as "spam" or "not spam" are examples used for training a supervised learning algorithm. The trained algorithm will then be able to predict with some degree of accuracy whether an unseen email is spam or not.

# Unsupervised Learning Part 1
* Unsupervised learning is not given labels
    * We are not trying to supervise the algo into saying what is right
* For tumors: unsupervised learning would cluster groups of tumor size
* Google news for example groups related articles or clusters them
* We can group people with similar DNA and label them at type 1, 2, ...
* Can you group customers for your businesses and see how to market to them

# Unsupervised Learning Part 2
* **Unsupervised learning**: Data only come at input $x$ and the algorithm tries to find *structure* in the data
* 3 main types of unsupervised learning
    1. Clustering: Grouping similar data points
    2. Anomaly Detection: Finding unusual activity (credit card stealing)
    3. Compressing data: Reduce the amount of storage needed to store something
* Question

    Of the following examples, which would you address using an unsupervised learning algorithm?  (Check all that apply.)

    * [x] Given a set of news articles found on the web, group them into sets of articles about the same stories.
    * [ ] Given email labeled as spam/not spam, learn a spam filter.
    * [x] Given a database of customer data, automatically discover market segments and group customers into different market segments.
    * [ ] Given a dataset of patients diagnosed as either having diabetes or not, learn to classify new patients as having diabetes or not.

    This a type of unsupervised learning called clustering

# Jupyter Notebooks
* Jupyter notebook is a widely used tool for machine learning
* The tools we are using are not a simplified version of what developers use
* In optional labs: just run the code and understand what is happening
* Cells can be markdown or code blocks, JNs are filled with a row of cells
* shift-enter is the way to run a code block

# Optional Lab 1: Python and Jupyter Notebooks
Lab 1 Jupyter [file](Labs/C1_W1_Lab01_Python_Jupyter_Soln.ipynb).

# Quiz: 100%
Quiz [file](./Quizzes.md#supervised-vs-unsupervised-machine-learning)