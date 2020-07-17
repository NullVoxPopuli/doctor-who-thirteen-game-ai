import tf from '@tensorflow/tfjs';

import { Agent } from './learning/agent';
import { Memory } from './learning/memory';

import type { SampleMemory, Config } from './learning/types';
import type { DirectionKey, InternalMove } from './consts';

import { ALL_MOVES, MOVE_KEY_MAP } from './consts';
import { clone, groupByValue, gameTo1DArray, isEqual } from './utils';
import { imitateMove, executeMove, fakeGameFrom } from './game';

const defaultConfig = {
  minEpsilon: 0.001,
  maxEpsilon: 0.1,
  epsilonDecaySpeed: 0.1,
};

export class GameTrainer {
  declare memory: Memory<SampleMemory>;
  declare config: Required<Config>;
  declare agent: Agent;
  declare model: tf.LayersModel;

  trainingStats = {
    totalGames: 0,
    totalMoves: 0,
    averageMovesPerGame: 0,
  };

  constructor(network: tf.LayersModel, config: Config) {
    this.config = { ...defaultConfig, ...config };
    this.model = network;
    this.agent = new Agent(network, config);
    this.memory = new Memory(config.memorySize);
  }

  async getMove(game: Game2048): Promise<DirectionKey> {
    let inputs = gameToTensor(game);
    let moveIndex = this.model.predict(inputs);

    debugger;
    let move = ALL_MOVES[moveIndex];

    return move;
  }

  async train(originalGame: Game2048) {
    let moves = 0;
    let start = new Date().getDate();
    let clonedGame = clone(originalGame);
    let gameManager = fakeGameFrom(clonedGame);

    let totalReward = 0;

    while (!gameManager.over) {
      let previousGame = clone(gameManager);
      let nextState = null;
      let currentState = gameToTensor(gameManager);
      let epsilon = decayEpsilon(this.config, moves);
      let rankedMoves = await this.agent.act(currentState, epsilon);

      debugger;
      let move = rankedMoves[0] as DirectionKey;

      executeMove(gameManager, move);

      nextState = gameToTensor(gameManager);
      moves++;

      let internalMove = MOVE_KEY_MAP[move];
      let reward = calculateReward(internalMove, previousGame, gameManager);

      if (gameManager.over) {
        nextState = null;
      }

      this.memory.add({ reward, nextState, state: currentState, action: move });

      this.agent.fit(currentState, rankedMoves);
    }

    // await this.replay();

    return {
      moves,
      score: gameManager.score,
      time: new Date().getDate() - start,
    };
  }
}

function decayEpsilon(config: Config, steps: number) {
  let { minEpsilon, maxEpsilon, epsilonDecaySpeed } = config;

  config.epsilon = minEpsilon + (maxEpsilon - minEpsilon) * Math.exp(-epsilonDecaySpeed * steps);

  return config.epsilon;
}

function gameToTensor(game: Game2048) {
  return tf.tensor1d(gameTo1DArray(game));
}

function calculateReward(move: InternalMove, originalGame: Game2048, currentGame: Game2048) {
  let moveData;
  let clonedGame;

  if (!currentGame) {
    clonedGame = clone(originalGame);
    moveData = imitateMove(clonedGame, move);
  } else {
    clonedGame = currentGame;
    moveData = {
      model: currentGame,
      score: currentGame.score,
      wasMoved: !isEqual(currentGame.serialize().grid.cells, originalGame.grid.cells),
    };
  }

  // if (clonedGame.over) {
  //   if (clonedGame.won) {
  //     return 1;
  //   } else {
  //     return -1;
  //   }
  // }

  if (!moveData.wasMoved) {
    // strongly discourage invalid moves
    return -1;
  }

  let grouped = groupByValue(originalGame);
  let newGrouped = groupByValue(moveData.model);

  let highest = Math.max(...Object.keys(grouped));
  let newHighest = Math.max(...Object.keys(newGrouped));

  // highest two were merged, we have a new highest
  if (newHighest > highest) {
    return 1;
  }

  // for each value, determimne if they've been merged
  // highest first
  // let currentValues = Object.keys(newGrouped).sort((a, b) => b - a);

  // let likelyWontMakeItTo = 15; // 2 ^ 30 -- need an upper bound for rewarding

  // for (let value of currentValues) {
  //   // what if it previously didn't exist? but still isn't highest?
  //   if (newGrouped[value] > (grouped[value] || 0)) {
  //     // log2 converts big number to small number
  //     // SEE: inverse of VALUE_MAP
  //     return Math.log2(value) / likelyWontMakeItTo;
  //   }
  // }

  // let bestPossibleMove = outcomesForEachMove(originalGame)[0] || {};
  // let bestPossibleScore = bestPossibleMove.score;

  // if (moveData.score >= bestPossibleScore) {
  //   return 1;
  // }

  if (moveData.score > originalGame.score) {
    return 1 - originalGame.score / moveData.score;

    // Provide a bigger reward the higher the merge value is

    // let additionalPoints = (moveData.score = originalGame.score);

    // let fractionalScore = additionalPoints / Math.pow(2, 13); // highest possible single merge score;

    // return fractionalScore > 1 ? 1 : fractionalScore;
  }

  // next score is equal to current
  // it's possible that we need to do something that doesn't
  // change our score before getting to something good
  return 0; // - originalGame.score / bestPossibleScore;
}
