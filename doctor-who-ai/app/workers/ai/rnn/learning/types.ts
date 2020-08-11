export interface Config {
  /**
   * number of actions
   *
   * @example
   *  4 actions for up, down, left, and right
   */
  numActions: number;
  /**
   * number of input states
   *
   * @example
   *  16 inputs for each of a 4x4 grid
   */
  numInputs: number;

  // TODO: document
  inputShape: number[];

  /**
   * (starting) chance of random action. Range: [0, 1]
   */
  epsilon: number;

  /**
   * minimum chance of random action. Range: [0, 1)
   * @default 0.001
   */
  minEpsilon?: number;

  /**
   * maximum chance of random action. Range: (0, 1]
   * @default 0.9
   */
  maxEpsilon?: number;

  /**
   * How fast epsilon decays.
   * @default 0.00001
   */
  epsilonDecaySpeed?: number;

  /**
   * How many games to keep in memory
   */
  gameMemorySize: number;

  /**
   * How many moves per game to keep in memory
   */
  moveMemorySize: number;

  /**
   * How much future rewards are weighted
   */
  learningDiscount: number;

  /**
   * How much to weigh the "next" reward
   */
  learningRate: number;
}
