import tf from '@tensorflow/tfjs';
import random from 'random';

import type { Config } from './types';
import type { InternalMove } from '../consts';

import { ALL_INTERNAL_MOVES } from '../consts';

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

  act(inputs: tf.Tensor, epsilon: number = -Infinity) {
    let { numActions } = this.config;

    if (Math.random() < epsilon) {
      return randomMove(numActions);
    }

    // ranked outputs for each of numActions
    // if numActions = 4, then there will be 4 elements in the returned array
    // expandDims converts regular inputs into batch inputs
    let inputData = inputs.expandDims();
    let moveWeights: number[];

    tf.tidy(() => {
      let output = this.model.predict(inputData);

      moveWeights = output.dataSync();
    });

    return moveInfoFor(moveWeights);
  }

  predict(inputs: tf.Tensor1D) {
    let inputData = inputs.expandDims();
    let output = tf.tidy(() => this.model.predict(inputData));

    return output;
  }
}

function randomMove(numActions: number) {
  let result: InternalMove[] = [];

  // [0, numActions]
  let generateMove = () => random.int(0, numActions - 1);

  while (result.length < 4) {
    let move = generateMove();

    if (!result.includes(move)) {
      result.push(move);
    }
  }

  return { sorted: result };
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
