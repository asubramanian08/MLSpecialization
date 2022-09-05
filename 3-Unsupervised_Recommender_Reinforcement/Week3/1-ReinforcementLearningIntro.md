# What is Reinforcement Learning?
* Not widely used in industry, but is a pillar of ML
* Example: Autonomous Helicopter
    * Learn how to fly the helicopter
    * Input: Position of the helicopter
    * Output: What action to take (How to move control stick)
    * Want to map a state $s$ to an action $a$
    * Main idea: Set up a reward/punishment systems for training the helicopter

        Kind of like training a dog, say what to do instead of what to do wrong
* Applications:
    * Controlling robots
    * Factory optimization
    * Stock trading
    * Game playing

# Mars Rover Example
* At every time step the rover at a certain position or state
    * The rover can move left or right
    * Each position has a reward
    * Have the model learn where it should move
    * **State-Action-Reward-Next State**

# The Return in Reinforcement Learning
* Example: We have a $5 bill or have to walk 30 mins for a $10 bill

    The return helps show that a faster solution is better that a slower solution
* Calculating the return:
    * $r$: Discount factor (ex. 0.9)
    * $R_i$: The reward at time step $i$
    * Return = $\sum_{i=1}^{|R|} r^{i-1} \times R_i$
* In finance: The discount factor represents inflation/interest rates
* Negative rewards incentive the algorithm to push out the reward into the future

# Making Decisions: Policies in Reinforcement Learning
**Policy**: A function that maps states to actions

This function decides what action to take given a state ($\pi(s) = a$)

# Review of Key Concepts
* States: The current situation (6 states - the position of the rover)
* Actions: The possible moves (left or right)
* Rewards: The reward for each state (100, 0, 40)
* Return: The total reward for a sequence of actions ($\sum_{i=1}^{|R|} r^{i-1} \times R_i$)
* Policy $\pi$: The function that maps states to actions ($\pi(s) = a$)
* Markov Decision Process (**MDP**): The framework for reinforcement learning

    The future only depends on the present state - See [page 18](Lecture.pdf)

# Quiz: 100%
Quiz [file](Quizzes.md#reinforcement-learning-introduction)