# Example of Continuous State Space Application
* Problem: What if the state space is a point on a grid rather then discrete states?
* Example: Model that state of a truck, that can drive in a 2D plane, as a point on a grid. The state variables are:
    * $x$: x-coordinate
    * $y$: y-coordinate
    * $\theta$: Angle of the truck
    * $\dot{x}$: x-velocity
    * $\dot{y}$: y-velocity
    * $\dot{\theta}$: change in angle
* See [page 36](Lecture.pdf) for an autonomous helicopter example

# Lunar Lander
* Lunar Lander is a classic/well studied example of reinforcement learning
* Goal: Land a rocket between two flags in an upwards orientation
* Actions: Do nothing, left thruster, right thruster, and main thruster(up)
* State:
    * $x$: x-coordinate
    * $y$: y-coordinate
    * $\dot{x}$: x-velocity
    * $\dot{y}$: y-velocity
    * $\theta$: angle (orientation)
    * $\dot{\theta}$: $angular velocity
    * $l$: Whether the left leg is touching the ground (0/1)
    * $r$: Whether the right leg is touching the ground (0/1)
* Reward:
    * $+100-+140$ for landing (depending on how centered it is)
    * $-100$ for crashing
    * $+10$ for one leg touching the ground
    * $-0.3$ for firing the main engine
    * $-0.03$ for firing the left/right thrusters
    * Additional reward for moving to/away from the flags
    * Note: This is a complex reward function just like most will be. This would still be better then coding the decision making process in a traditional way
* Goal: Learn a policy $\pi$ that maximizes given a state $s$ that picks action $a = \pi(s)$ to maximize the return

# Learning the State-Value Function
* Goal: Learn or approximate the function $Q(s,a)$
* Solution: Train an NN to approximate $Q(s,a)$
    * Input $\begin{bmatrix} s \\ a \end{bmatrix}$: The action and current state

        Note: action $a$ will use one-hot encoding!
    * Output $Q(s,a)$: The (approximated) best return for each action
* The NN's notation / functions:
    * $\vec{x}$: $\begin{bmatrix} s \\ a \end{bmatrix}$
    * $\hat{y}$: $Q(s,a)$
    * $y$: $R(s) + \gamma \max_{a'} Q(s',a')$
    * $f_{W,B}(x) \approx y$
* Train the NN: Generate a bunch of data and train the NN on it
    1. Generate a bunch of data:
        * Pick a random $s$ and $a$
        * Get the reward $R(s)$
        * Get the next state $s'$ by applying the action $a$ to $s$
        * Store the data: $(s, a, R(s), s')$ tuple
    2. Divide the data into $x$ and $y$ (That the NN can train on)
        * $x = (s, a)$
        * $y = R(s) + \gamma \max_{a'} Q(s',a')$ (see below on how to calculate this)
    3. Train the NN on the data
* **Full** learning algorithm (DQN algorithm):
    1. Initialize the NN with a random guess for $Q(s,a)$
    2. Take random actions to get $(s, a, R(s), s')$
    3. Store the 10,000 recent examples of these tuples (**Replay Buffer**)
    4. Train the NN
        * Create training set of 10,000 examples where ...
        
            $x = (s,a)$ and $y = R(s) + \gamma \max_{a'} Q(s',a')$
        * Train $Q_\text{new}$ such that $Q_\text{new}(s,a) \approx y$
    5. Set $Q = Q_\text{new}$
    6. Repeat steps 2-5 until $Q$ converges
    * Problem: This algorithm might take a while to converge. See [below](#algorithm-refinement-improved-neural-network-architecture) for how to speed it up.

# Algorithm Refinement: Improved Neural Network Architecture
* This architecture will the DQN algorithm converge faster (**more efficient**)
* Updated NN:
    * Input: just $s$ (not $a$)
    * Output: $Q(s,a)$ for all $a$

        There will be one output for each action.

# Algorithm Refinement: $\epsilon$-Greedy Policy
* Problem: When generating the data, we don't want to randomly pick an action
    * Option 1: Pick the action that maximizes $Q(s,a)$
    * Option 2: **$\epsilon$-Greedy Policy**
        * "Exploration" step: Pick a random action with probability $\epsilon$
        * "Greedy"/"Exploitation" step: Pick the action that maximizes $Q(s,a)$ with probability $1-\epsilon$
* Note: Start $\epsilon$ high and decrease it over time

    At first you don't trust you algorithm, then gain more and more trust.
* Note: Unlike in supervised learning, picking the right parameters make a large effect on performance / speed of convergence

# Algorithm Refinement: Mini-Batch and Soft Updates (Optional)
* Mini-batch Gradient Descent:
    * A dataset might be too large to train on for every step of gradient descent
    * Solution: Train on a random subset of the data
    * Though the improvements may not be consistent, this will speed up training by a lot
* Soft Updates:
    * Problem: What if the calculated $Q_\text{new}$ is not very good?
    * We want to update $Q$ slowly so that we don't make large changes to $Q$ that might make it worse
    * The weights are updated by:
    
        $W_{\text{new}} = \tau W_{\text{new}} + (1-\tau) W$
    * $\tau$ is a small number (e.g. $0.01$)
    * This will make the NN converge more reliably

# The State of Reinforcement Learning
* Limitations of RL:
    * Easier to make RL work on a simulation than on a real robot
    * Far less applications than supervised and unsupervised learning
* Still: There is a lot of research on RL that might be applicable in the future

# Quiz: 100%
Quiz [file](Quizzes.md#continues-state-spaces)