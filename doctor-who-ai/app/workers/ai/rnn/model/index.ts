import * as tf from '@tensorflow/tfjs';

import { ALL_INTERNAL_MOVES } from 'ai/consts';

type NumberArray = Float32Array | Int32Array | Uint8Array;

/**
 * Helper class that wraps a built TensorFlow model but
 * also takes care of cleanup and, in our case for 2048,
 * sorts the ranked prediction output
 */
export class Model {
  declare model: tf.LayersModel;

  constructor(model: tf.LayersModel) {
    this.model = model;
  }

  fit(gameState: tf.Tensor, rankedMoves: tf.Tensor) {
    return this.model.fit(gameState, rankedMoves);
  }

  /**
   * ranked outputs for each of numActions
   * if numActions = 4, then there will be 4 elements in the returned array
   *
   * @param inputs
   * @param gameManger
   */
  act(inputs: tf.Tensor) {
    let prediction = this.predict(inputs);
    let ranked = (prediction as tf.Tensor<tf.Rank>).dataSync();

    return moveInfoFor(ranked);
  }

  predict(inputs: tf.Tensor) {
    inputs.print();
    let output = tf.tidy(() => this.model.predict(inputs));

    return output;
  }
}

export function act(network: tf.LayersModel, inputs: tf.Tensor) {
  let ranked;

  tf.tidy(() => {
    ranked = network.predict(inputs).argMax(-1).dataSync();
  });

  return moveInfoFor(ranked);
}

function moveInfoFor(weights: NumberArray) {
  let sorted = sortedMoves(weights);

  return {
    weights,
    sorted,
  };
}

function sortedMoves(weights: NumberArray) {
  let moves = ALL_INTERNAL_MOVES.sort((a, b) => {
    return weights[b] - weights[a];
  });

  return moves;
}
