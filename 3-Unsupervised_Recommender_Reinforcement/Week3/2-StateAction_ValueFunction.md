# State-Action Value Function Definition
* State-Action Value Function ($Q(s,a)$): The return is you take action $a$ at state $s$ and behave optimally thereafter.
    * See several examples of $Q(s,a)$ on [page 20](Lecture.pdf).
    * Note: Also called the $Q$ function, $Q^*$, or optimal $Q$ function
* The best return/action for a state $a$: $\max_a Q(s,a)$

    In other words if $\max_a Q(s,a)$, then $\pi(s) = a$ 

# State-Action Value Function Examples
* This video demonstrates what is shown in the [optional lab](#optional-lab-1-state-action-value-function) after this video
* See what changes happen to $Q(s,a)$ and $\pi(s)$ when ... is changed:
    * Rewards are each state
    * Gamma $\gamma$, the discount factor

# Bellman Equations
* How to compute $Q(s,a)$?

    Use the Bellman equations
* Notation
    * $s$: Current state
    * $a$: Current action
    * $s'$: Next state (after action $a$)
    * $Q(s,a)$: Return is you take action $a$ at state $s$ and behave optimally thereafter
    * $R(s)$: Reward at state $s$
    * $a'$ Action you take after $s'$
* **Bellman Equation$** for $Q(s,a)
    $$Q(s,a) = R(s) + \gamma \max_{a'} Q(s',a')$$
* See [page 26](Lecture.pdf) for an example
* Explanation
    * $R(s)$: The immediate reward for being at state $s$
    * $\max_{a'} Q(s',a')$: The optimal return from $s'$
    * $\gamma$: The discount factor (Will multiply recursively to account for the exponential decay of future rewards)

# Random (stochastic) Environment (Optional)
* In a random environment, the next state is not deterministic

    There might be wind or rocky terrain so we don't exactly know where the robot will end up
* Specifically, if we instruct the robot the go left: There is a 90% chance is goes left, but a 10% chance it goes right
* **Expected Return**: Rather then maximizing the "return", we maximize the expected return
    * The expected return is the sum of all possible returns weighted by the probability of that return
    * Expected return = $E(R_1 + \gamma R_2 + \gamma^2 R_3 + \cdots)$ where $E()$ is the expected value of a random variable
    * **New** bellman equation: $Q(s,a) = R(s) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q(s',a')$
* Note: Test this out by changing the `misstep_prob` parameter in the [optional lab](#optional-lab-1-state-action-value-function)

# Optional Lab 1: State-Action Value Function
Lab 1 Jupyter [file](Labs/State-action%20value%20function%20example.ipynb).

# Quiz: 100%
Quiz [file](Quizzes.md#state-action-value-function)