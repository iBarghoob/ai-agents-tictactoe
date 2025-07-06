package ticTacToe;


import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		Random random = new Random();
		// policy initialisation with random moves for each state
		for (Game state : policyValues.keySet()) {
			if (state.isTerminal()) {
				continue;
			}
			// get a list of all valid moves for current state
			List<Move> possibleMoves = state.getPossibleMoves();
			// select a random move from list of possible moves	
			if (!possibleMoves.isEmpty()) {
				Move randomMove = possibleMoves.get(random.nextInt(possibleMoves.size()));
				// store random mvoe in curPolicy as action for current state
				curPolicy.put(state, randomMove);
			}
		}
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		while (true) {
			// new map to store updated values
			HashMap<Game, Double> newPolicyValues = new HashMap<>();
			boolean converged = true;

			// loop over each state in current policy
			for (Game state : policyValues.keySet()) {
				if (state.isTerminal()) {
					newPolicyValues.put(state, 0.0);
					continue;
				}

				// get action set by current policy pi(s) 
				Move action = curPolicy.get(state);
				// get all possible transitions for action
				List<TransitionProb> transitions = mdp.generateTransitions(state, action);
				double newVal = 0.0;

				// calculate new value V_pi(s) using the Bellman equation 
				for (TransitionProb transition : transitions) {
					Game nextState = transition.outcome.sPrime; // s'
					double probability = transition.prob; // T(s,pi(s),s')
					double reward = transition.outcome.localReward; // R(s,pi(s),s')

					// sum up expected rewards for taking that action 
					newVal += probability * (reward + discount * policyValues.get(nextState));
				}
				// update value for state
				newPolicyValues.put(state, newVal);

				// check if change in value exceeds delta 
				// if not, then policy has converged
				if (Math.abs(newVal - policyValues.get(state)) > delta) {
					converged = false;
				}
			}
			// update policy values with new values 
			policyValues = newPolicyValues;
			// break if converged
			if (converged) {
				break;
			}
		}
	}

		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		boolean policyImproved = false;

		// loop through each game state in current policy
		for (Game state : curPolicy.keySet()) {
			if (state.isTerminal()) {
				continue;
			}

			Move bestAction = null;
			// value initialised for best action q-value
			double maxQValue = Double.NEGATIVE_INFINITY;

			// evaluate all possible actions for current state
			for (Move action : state.getPossibleMoves()) {
				double actionValue = 0.0;
				// generate all possible transitions for action
				// update the action value Q*(s,a) using the Bellman equation
				List<TransitionProb> transitions = mdp.generateTransitions(state, action);
				for (TransitionProb transition : transitions) {
					Game nextState = transition.outcome.sPrime; // s'
					double probability = transition.prob; // T(s,a,s')
					double reward = transition.outcome.localReward; // R(s,a,s') 
					
					// sum up expected rewards of that action
					actionValue += probability * (reward + discount * policyValues.get(nextState));
				}

				// use new action value to update optimal action if it is better 
				if (actionValue > maxQValue) {
					maxQValue = actionValue;
					bestAction = action; // argmax
				}
			}
			// update policy if a different action found
			if (bestAction != null && !bestAction.equals(curPolicy.get(state))) {
				curPolicy.put(state, bestAction);
				policyImproved = true;
			}
		}
		return policyImproved;
	}
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		while (true) {
			// evaluate policy until convergence
	        evaluatePolicy(delta);
	        // improve policy using state values
	        boolean policyImproved = improvePolicy();
	        // break once policy stops improving, policy stable
	        if (!policyImproved) {
	            break;
	        }
	    }
	    // store the final stable policy
	    this.policy = new Policy(curPolicy);
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
