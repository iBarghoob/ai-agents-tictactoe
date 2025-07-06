# AI Agents for Tic-Tac-Toe: MDPs and Reinforcement Learning
Java-based AI project implementing three intelligent agents using Value Iteration, Policy Iteration, and Q-Learning to learn and play 3x3 Tic-Tac-Toe against rule-based opponents. Developed as part of academic coursework to understand Markov Decision Processes and Reinforcement Learning. 


This project is a Java-based implementation of three AI agents designed to play Tic-Tac-Toe. The agents use **Value Iteration**, **Policy Iteration**, and **Q-Learning** to make intelligent decisions based on a Markov Decision Process (MDP) model or reinforcement learning environment. Completed as an academic project as part of a coursework based on Markov Decision Processes and Reinforcement Learning.

## Agents Implemented

- **ValueIterationAgent.java**  
  Solves the Tic-Tac-Toe game using value iteration on a known MDP model.

- **PolicyIterationAgent.java**  
  Implements policy iteration to derive optimal strategies in the MDP.

- **QLearningAgent.java**  
  Learns strategies through reinforcement learning without prior knowledge of the transition model.

---

## Opponent Agents 

The implemented agents were tested by playing multiple games 50 games each against the following rule based agents:

- **RandomAgent** – Chooses moves randomly
- **AggressiveAgent** – Prioritizes winning moves
- **DefensiveAgent** – Focuses on blocking the opponent
- **HumanAgent** – Allows user to manually play against the AI

---

## Project includes

| File | Description |
|------|-------------|
| `Game.java` | Core implementation of 3x3 Tic-Tac-Toe game logic |
| `TTTMDP.java` | Defines the MDP model of the Tic-Tac-Toe environment |
| `TTTEnvironment.java` | Defines the Reinforcement Learning environment for Q-learning |
| `Agent.java` | Base class for all agents |
| `Policy.java` | Base class for agent policies |
| `Outcome.java`, `Move.java`, `TransitionProb.java` | Supporting classes for state transitions and actions |

---

The PDF reports for the performance evluation of each implemented agent and short descriptions breaking down the methods implemented.

---