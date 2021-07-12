import * as tf from '@tensorflow/tfjs';

import type { Config } from './types';

import { ALL_INTERNAL_MOVES } from '../../consts';
import { guidedMove } from '../../a-star';

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

  act(inputs: tf.Tensor, epsilon = -Infinity, gameManger: Game2048) {
    let { numActions } = this.config;

    if (Math.random() < epsilon) {
      return guidedMove(numActions, gameManger);
    }

    // ranked outputs for each of numActions
    // if numActions = 4, then there will be 4 elements in the returned array
    // expandDims converts regular inputs into batch inputs
    let inputData = inputs.expandDims();
    let result;

    tf.tidy(() => {
      let output = this.model.predict(inputData);

      let moveWeights = output.dataSync();
      result = moveInfoFor(moveWeights);
    });

    return result;
  }

  predict(inputs: tf.Tensor1D) {
    let inputData = inputs.expandDims();
    let output = tf.tidy(() => this.model.predict(inputData));

    return output;
  }
}

function moveInfoFor(weights: number[]) {
  let sorted = sortedMoves(weights);

  return {
    weights,
    sorted,
  };
}

function sortedMoves(weights: number[]) {
  let moves = ALL_INTERNAL_MOVES.sort((a, b) => {
    return weights[b] - weights[a];
  });

  return moves;
}
