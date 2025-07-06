package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=40000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 40000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train()
	{	
		for (int episode = 0; episode < numEpisodes; episode++) {
			while (!env.isTerminal()) {
				// get state for current game
				Game state = env.getCurrentGameState();
		
				Random random = new Random();
				Move chosenMove = null;		
				// get list of all possible moves for state
				List<Move> possibleMoves = state.getPossibleMoves();
				double qValue = 0;
				double maxQValue = Double.NEGATIVE_INFINITY;
				
				// choose move based on epsilon greedy policy (exploration or exploitation) 
				if(random.nextDouble() < epsilon) {	
					// if random value below given epsilon value
					// exploration, choose a random move
					chosenMove = possibleMoves.get(random.nextInt(possibleMoves.size()));
				} else {
					// exploitation, choose the best move based on the q values (s,a)
					for (Move move : possibleMoves) {
						// get q value for each state action pair
						qValue = qTable.getQValue(state, move);
						// update max Q value and optimal move accordingly
						if (qValue >= maxQValue) {
							maxQValue = qValue;
							chosenMove = move;
						}
					}
				}
				// use chosen move then observe outcome sample
				Outcome outcome;
				try {
					outcome = env.executeMove(chosenMove);
				} catch (IllegalMoveException e) {
					System.out.println("Illegal move attempted " + e.getMessage());
					break;
				}
				// sample (s, a, s', r)
				Game sourceState = outcome.s;
				Move action = outcome.move;
				Game nextState = outcome.sPrime;
				double reward = outcome.localReward;
				
				// q value for state action pair Q(s,a)
				double currentQValue = qTable.getQValue(sourceState, action);
				
				// get max q value for next state
				double nextMaxQValue = Double.NEGATIVE_INFINITY;
				if (nextState.isTerminal()) {
					nextMaxQValue = 0.0;
				} else {
					for (Move move : nextState.getPossibleMoves()){
						double nextQValue = qTable.getQValue(nextState, move);
						if (nextQValue > nextMaxQValue) {
							nextMaxQValue = nextQValue;
						}
					}
				}
				
				// update q value for state action pair 
				double updatedQValue = (1 - alpha) * currentQValue + alpha * (reward + discount * nextMaxQValue);
				// update q table
				qTable.addQValue(sourceState, action, updatedQValue);
			}
			// reset environment for next episode
			env.reset();
		}
        
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}


	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		// map to store updated policy
		HashMap<Game, Move> policyMap = new HashMap<>();
		
		// loop through all states in the q table
		for (Game state : qTable.keySet()) {
			if (state.isTerminal()) {
				continue;
			}
			
			// get all possible moves for state
			List<Move> possibleMoves = state.getPossibleMoves();
			// initialise max q value
			double maxQValue = Double.NEGATIVE_INFINITY;
			Move bestMove = null;
			
			for (Move move : possibleMoves) {
                double qValue = qTable.getQValue(state, move);
                if (qValue > maxQValue) {
                    maxQValue = qValue;
                    bestMove = move;
                }
            }
			// update policy
			// store best move for the current state in the policy map
			policyMap.put(state, bestMove);
		}
		
		// return updated policy map
		return new Policy(policyMap);
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
