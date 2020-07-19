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

  fit(gameState: tf.Tensor, rankedMoves: tf.Tensor) {
    return this.model.fit(gameState, rankedMoves);
  }

  act(inputs: tf.Tensor, epsilon: number = Infinity) {
    let { numActions } = this.config;

    if (Math.random() < epsilon) {
      return random.int(0, numActions - 1); // [0, numActions]
    }

    // ranked outputs for each of numActions
    // if numActions = 4, then there will be 4 elements in the returned array
    // expandDims converts regular inputs into batch inputs
    let inputData =  inputs.expandDims();
    let output = tf.tidy(() => this.model.predict(inputData));

    let moves = output.dataSync();

    return highestIndex(moves);
  }

  predict(inputs: tf.Tensor1D) {
    let inputData = inputs.expandDims();
    let output = tf.tidy(() => this.model.predict(inputData));

    return output;
  }
}

function highestIndex(arr: number[]) {
  let highestIndex = 0;
  let highest = 0;

  for (let i = 0; i < arr.length; i++) {
    let value = arr[i];

    if (highest < value) {
      highest = value;
      highestIndex = i;
    }
  }

  return highestIndex;
}
