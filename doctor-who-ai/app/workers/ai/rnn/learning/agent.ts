import * as tf from '@tensorflow/tfjs';
import * as random from 'random';

import { imitateMove, fakeGameFrom } from '../game';
import { clone } from '../utils';

import type { Config } from './types';
import type { InternalMove } from '../consts';

import { ALL_INTERNAL_MOVES, ALL_MOVES, MOVE_KEY_MAP } from '../consts';

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

  act(inputs: tf.Tensor, epsilon: number = -Infinity, gameManger: Game2048) {
    let { numActions } = this.config;

    if (Math.random() < epsilon) {
      return guidedMove(numActions, gameManger);
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

type SearchNode = {
  value: {
    model: Game2048;
  };
  children: SearchNode[];
  move?: number;
  parent?: SearchNode;
  weightedScore?: number;
};

function guidedMove(numActions: number, gameManager: Game2048) {
  let result: InternalMove[] = [];

  let bestNode: any;
  let bestScore = 0;
  let bestHops = 1000;

  let rootNode: SearchNode = {
    value: { model: gameManager },
    children: [],
  };

  function updateBest(childNode: SearchNode) {
    if (childNode === rootNode) {
      return;
    }

    if (childNode.weightedScore < bestScore) {
      return;
    }

    // if the score is equal, let's choose the least hops
    let root = childNode;
    let hops = 0;

    while (root.parent !== undefined && root.parent.move !== undefined) {
      root = root.parent;
      hops++;
    }

    // if (hops < bestHops) {
    //   if (hops === 0) {
        if (childNode.weightedScore > bestScore) {
          bestNode = root;
          bestScore = childNode.weightedScore;
        }

        // return;
      // }

      // bestHops = hops;
      // bestNode = root;
      // bestScore = childNode.weightedScore;
    // }
  }

  function expandTree(node: SearchNode, level: number) {
    updateBest(node);

    if (level >= 3) {
      return;
    }

    const enumerateMoves = () => {
      for (let move of ALL_MOVES) {
        let copyOfModel = clone(node.value);
        let moveData = imitateMove(copyOfModel.model, move);

        if (!moveData.wasMoved) {
          continue;
        }

        let scoreChange = moveData.score - gameManager.score;
        let weightedScore = level === 0 ? scoreChange : scoreChange / level;

        node.children.push({
          // penalize scores with higher depth
          weightedScore,

          value: moveData,
          children: [],
          move: MOVE_KEY_MAP[move],
          parent: node,
        });
      }
    };

    enumerateMoves();

    for (let childNode of node.children) {
      expandTree(childNode, level + 1);
    }
  }

  expandTree(rootNode, 0);

  if (bestNode && bestNode.move) {
    result.push(bestNode.move);
  }

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
