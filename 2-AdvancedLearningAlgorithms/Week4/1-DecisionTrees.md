# Decision Tree Model
* Problem (used for the whole week): Cat classification
* Input features: Only take up a few discrete values
    * Ear shape: Floppy / Pointy
    * Face shape: Round / Not Round
    * Whiskers: Present / Absent
* Output: Predicting whether it is a cat or not
* Decision Tree structure, see [page 4](Lecture.pdf).
    * This is a CS tree - has a root node and branches down into several leaf nodes
    * At each (non leaf) node we will be asked a question: Ex. Ear shape?
    * A discrete number of options will be given. Whichever option matches, we will enter into that node. Then be asked another question ...
    * After all series of questions, the program will end in a leaf node. This node will tell whether or not the decision tree predicts the input is a cat.

# Learning Process
* Abstract Learning Process:
    * We are given a set of all the different training examples
    * Need to pick a feature (Whiskers, etc). After that, create edges from this node representing each value for that feature.
    * Divide the given examples into different sets, based on that feature.
    * If these sets contain only cats or no cat, make is a leaf. Otherwise, repeat the previous steps using, the "newly divided" set of training examples.
* Choices we have in creating a decision tree:
    1. Choose what feature to split on:
    
        Pick whatever one maximized purity: Leads to a set being all cats or all dogs.
    2. When to stop splitting:
        * When a node has 100% of one class
        * When the node exceeds the "maximum" depth
        * When the purity score is below a threshold
        * When the number of examples is below a threshold

# Quiz: 100%
Quiz [file](Quizzes.md#decision-trees)