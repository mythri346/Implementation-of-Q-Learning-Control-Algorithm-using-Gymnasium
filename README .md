# Implementation of Q-Learning Control Algorithm using Gymnasium
# Name:Kolluru Pujitha
# Reg.no:212223240074

## Aim

To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

## Problem Statement

To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The agent must learn an optimal action-value function through interaction with the environment and select suitable actions to reach the goal while avoiding the holes.

## Software Requirements

* **Programming Language:** Python
* **Library:** Gymnasium
* **Numerical Computation:** NumPy
* **Visualization:** Matplotlib
* **Environment:** FrozenLake-v1
* **Platform:** Jupyter Notebook / Google Colab

## Environment Description

The FrozenLake-v1 environment is a grid-world reinforcement learning environment provided by Gymnasium.

The environment consists of a **4 × 4 grid** containing:

* **S** – Starting state
* **F** – Frozen/safe surface
* **H** – Hole, which terminates the episode
* **G** – Goal state

The agent starts from the initial state and must navigate through the frozen surface to reach the goal without falling into a hole.

The environment contains **16 states** and **4 possible actions**:

| Action | Direction |
| ------ | --------- |
| 0      | Left      |
| 1      | Down      |
| 2      | Right     |
| 3      | Up        |

The agent receives a reward when it successfully reaches the goal. Falling into a hole terminates the episode.

## Theory

Q-Learning is an **off-policy reinforcement learning control algorithm** that learns the optimal action-value function directly.

The action-value function \(Q(s,a)\) represents the expected return obtained when the agent takes action \(a\) in state \(s\) and subsequently follows the best possible policy.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) +
\alpha
\left[
R_{t+1}+
\gamma\max_a Q(S_{t+1},a)
-Q(S_t,A_t)
\right]
$$

Where:

| Symbol                  | Meaning                                |
| ----------------------- | -------------------------------------- |
| \(S_t\)                 | Current state                          |
| \(A_t\)                 | Current action                         |
| \(R_{t+1}\)             | Reward received after taking action    |
| \(S_{t+1}\)             | Next state                             |
| \(\alpha\)              | Learning rate                          |
| \(\gamma\)              | Discount factor                        |
| \(Q(s,a)\)              | Action-value function                  |
| \(\max_a Q(S_{t+1},a)\) | Maximum action value in the next state |

The important feature of Q-Learning is that it uses the **maximum Q-value of the next state** when calculating the target:

$$
Target = R+\gamma\max_a Q(S',a)
$$

Therefore, Q-Learning learns the optimal policy independently of the action actually selected during exploration.

## Epsilon-Greedy Action Selection

During training, the agent uses an **epsilon-greedy strategy** to balance exploration and exploitation.

With probability \(\epsilon\), the agent explores by selecting a random action.

With probability \(1-\epsilon\), the agent exploits the learned Q-table by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon\\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

Initially, epsilon is high, allowing the agent to explore different actions. As training progresses, epsilon is gradually reduced so that the agent increasingly exploits the learned Q-values.

## Algorithm

1. Initialize the FrozenLake-v1 environment.
2. Determine the number of states and actions.
3. Initialize the Q-table with zeros for all state-action pairs.
4. Set the learning rate \(\alpha\), discount factor \(\gamma\), initial epsilon, minimum epsilon, and epsilon decay rate.
5. Reset the environment at the beginning of each episode.
6. Select an action using the epsilon-greedy strategy.
7. Execute the selected action in the environment.
8. Observe the next state, reward, and termination status.
9. Calculate the Q-Learning target:

$$
Target = R+\gamma\max_a Q(S',a)
$$

10. Update the Q-value using:

$$
Q(S,A) \leftarrow Q(S,A)+
\alpha[Target-Q(S,A)]
$$

11. Move to the next state.
12. Continue until the episode terminates.
13. Store the reward obtained in the episode.
14. Reduce epsilon gradually to decrease exploration and increase exploitation.
15. Repeat the process for the specified number of episodes.
16. Obtain the state-value function using:

$$
V(S)=\max_a Q(S,a)
$$

17. Obtain the learned policy by selecting the action with the maximum Q-value for each state.
18. Plot the learning curve.
19. Calculate the average reward over the last 1000 episodes.

## Python Program

```python
# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
episode_rewards = []

for episode in range(num_episodes):

    state, _ = env.reset()

    total_reward = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Select action using epsilon-greedy strategy
        action = choose_action(state, epsilon)

        # Take action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Calculate Q-Learning target
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        # Update Q-value
        Q[state, action] += learning_rate * (
            target - Q[state, action]
        )

        # Move to next state
        state = next_state

        total_reward += reward

    # Store total reward of the episode
    episode_rewards.append(total_reward)

    # Reduce exploration gradually
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )
# -------------------------------------------------
# State-Value Function and Learned Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)
```


### Estimated State-Value Function

The state-value function is obtained using:

$$
V(S)=\max_a Q(S,a)
$$

It represents the highest estimated value for each state.

### Learned Policy

The learned policy selects the action having the maximum Q-value for each state.

The actions are represented as:

* **L** – Left
* **D** – Down
* **R** – Right
* **U** – Up

### Average Reward

The average reward is calculated over the **last 1000 episodes** to evaluate the final performance of the learned agent.

### Learning Curve

The learning curve shows how the average reward changes during training.
## Output
<img width="675" height="570" alt="image" src="https://github.com/user-attachments/assets/4fee94d4-fc0f-4b68-9ccf-e788b9ae5459" />

<img width="995" height="492" alt="image" src="https://github.com/user-attachments/assets/32858e8e-99ed-439f-b667-d91049548247" />

## Result
Q-Learning was successfully implemented using the Gymnasium FrozenLake-v1 environment.The agent learned Q-values for each state-action pair through repeated interaction.The state-value function and policy were derived by selecting the highest Q-value for each state.The learning curve and average reward over the last 1000 episodes show the agent’s final performance.
## Inference
Q-Learning learns the best action-selection policy through trial and error by balancing exploration and exploitation.The Q-table gradually learns action values, while the state-value function and policy are derived from the highest Q-values.Thus, Q-Learning effectively learns a suitable policy for the FrozenLake environment.
