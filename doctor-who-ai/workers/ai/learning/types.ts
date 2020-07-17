import tf from '@tensorflow/tfjs';
import { DirectionKey } from 'doctor-who-ai/services/ai/consts';

export interface Config {
  /**
   * number of actions
   *
   * @example
   *  4 actions for up, down, left, and right
   */
  numActions: number;

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
   * @default 0.1
   */
  maxEpsilon?: number;

  /**
   * How fast epsilon decays.
   * @default 0.1
   */
  epsilonDecaySpeed?: number;

  /**
   * How many games to keep in memory
   */
  memorySize: number;
}

export type SampleMemory = {
  state: tf.Tensor1D;
  action: DirectionKey;
  reward: number;
  nextState: tf.Tensor1D | null;
};
