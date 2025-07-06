package ticTacToe;


import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=50;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate()
	{
		for (int i = 0; i < k; i++) {
			// new map to store updated values
			Map<Game, Double> newValueFunction = new HashMap<>();
			
			for (Game state : valueFunction.keySet()) {
				// if state is terminal, set its value to 0 and continue to next state
				if (state.isTerminal()) {
					newValueFunction.put(state, 0.0);
					continue;
				}
				
				// value initialised
				double maxQValue = Double.NEGATIVE_INFINITY;

				// compute values for all possible actions for the state
				// get optimal value for state by the maximum of Q*(s,a) over all actions
				for (Move action : state.getPossibleMoves()) {
					double actionValue = 0.0;
					
					// get all possible transitions for this action
		 			List<TransitionProb> transitions = mdp.generateTransitions(state, action);
		 			
		 			// get expected value of action for state, summing the expected rewards over all 
		 			// possible next states
					for (TransitionProb transition : transitions) {
						Game nextState = transition.outcome.sPrime; // s'
						double probability = transition.prob; // T(s,a,s')
						double reward = transition.outcome.localReward; // R(s,a,s')

						// sum expected rewards using Bellman's equation Q*(s,a)
						actionValue += probability * (reward + discount * valueFunction.get(nextState));
					}

					// update value of action with max utility for the state
					// V*(s) = maxQ*(s,a)
					if (actionValue > maxQValue) {
						maxQValue = actionValue;
					}
				}
				// update the optimal action value for state in value function V*(s)
	            newValueFunction.put(state, maxQValue); 
			}
			// replace old value function with updated one
			valueFunction = newValueFunction;
		}
	}
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		HashMap<Game, Move> policyMap = new HashMap<>();

		// iterate over all states in the value function, which has been updated by the iterate() method
		for (Game state : valueFunction.keySet()) {
			// skip terminal states
			if (state.isTerminal()) {
				continue;
			}
			
			// value initialised
			double maxQValue = Double.NEGATIVE_INFINITY;
			// optimal move for state
			Move bestAction = null;
 
			// loop through all possible actions for current state
			for (Move action : state.getPossibleMoves()) {
				double actionValue = 0.0;

				// get all possible transitions for the current action
				List<TransitionProb> transitions = mdp.generateTransitions(state, action);
				
				// get expected value of action for state, summing the expected rewards over all 
	 			// possible next states
				for (TransitionProb transition : transitions) {
					Game nextState = transition.outcome.sPrime; // s'
					double probability = transition.prob; // T(s,a,s')
					double reward = transition.outcome.localReward; // R(s,a,s')
					
					// Q*(s,a)
					actionValue += probability * (reward + discount * valueFunction.get(nextState));
				}

				// update maximum action value and the optimal action for state
				if (actionValue > maxQValue) {
					// value updated
					maxQValue = actionValue; 
					bestAction = action; // argmax Q*(s,a)
				}
			}

			// store best action for the current state in the policy map
			policyMap.put(state, bestAction);
		}
		
		return new Policy(policyMap);
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
