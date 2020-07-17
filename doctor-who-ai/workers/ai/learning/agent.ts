import tf from '@tensorflow/tfjs';
import random from 'random';

import type { Config } from './types';

export class Agent {
  declare config: Config;
  declare model: tf.LayersModel;

  constructor(model: tf.LayersModel, config: Config) {
    this.config = config;
    this.model = model;
  }

  fit(gameState: tf.Tensor1D, rankedMoves: tf.Tensor1D) {
    return this.model.fit(gameState, rankedMoves, { batchSize: 32, epochs: 1 });
  }

  act(inputs: tf.Tensor1D, epsilon: number) {
    let { numActions } = this.config;

    if (Math.random() < epsilon) {
      return random.int(0, numActions - 1); // [0, numActions]
    }

    // ranked outputs for each of numActions
    // if numActions = 4, then there will be 4 elements in the returned array
    return tf.tidy(() => this.model.predict(inputs));
  }
}
