import tf from '@tensorflow/tfjs';
import random from 'random';

import { QLearn } from './learning/qlearn';

import type { Config } from './learning/types';
import type { DirectionKey, InternalMove } from '../consts';

import { ALL_MOVES } from '../consts';
import { clone, groupByValue, gameToTensor, isEqual } from './utils';
import { imitateMove, executeMove, fakeGameFrom } from '../game';
import { Model } from './model';
import { guidedMove } from '../a-star';

import type { GameManager } from 'ai/rnn/vendor/app.map-worker-edition';

const defaultConfig = {
  minEpsilon: 0.001,
  maxEpsilon: 0.1,
  epsilonDecaySpeed: 0.1,
};

type RewardInfo = {
  reward: number;
  scoreChange: number;
  wasMoved: boolean;
  currentScore: number;
  previousScore: number;
  over: boolean;
  won: boolean;
  nextState: tf.Tensor2D;
};

export class GameTrainer {
  declare config: Required<Config>;
  declare model: Model;
  declare qlearn: QLearn;

  constructor(network: tf.LayersModel, config: Config) {
    this.config = { ...defaultConfig, ...config };
    this.model = new Model(network);
    this.qlearn = new QLearn(config);
  }

  async getMove(game: GameState): Promise<DirectionKey> {
    let inputs = gameToTensor(game);
    let moveInfo = this.model.act(inputs.reshape([16]));
    let validMove = firstValidMoveOf(moveInfo.sorted, game);

    return ALL_MOVES[validMove];
  }

  async train(originalGame: GameState, numberOfGames = 1) {
    let trainingStats = {
      totalMoves: 0,
      totalInvalid: 0,
      totalScore: 0,
      averageScore: 0,
      averageMoves: 0,
      averageInvalid: 0,
      bestScore: 0,
    };

    for (let i = 0; i < numberOfGames; i++) {
      let gameManager = fakeGameFrom(clone(originalGame));

      let result = await this.qlearn.playOnce<GameManager, tf.Tensor1D, InternalMove, RewardInfo>({
        game: gameManager,
        getState: (game) => gameToTensor(game).reshape([16]),
        isGameOver: (game) => game.over,
        isValidAction: (rewardInformation) => rewardInformation.wasMoved,
        getReward: (game, action) => moveAndCalculateReward(action, game),
        getRankedActions: (game, state, useExternalAction) => {
          let moveInfo;

          if (useExternalAction) {
            // moveInfo = guidedMove(4, game);
            moveInfo = randomMoves(this.config.numActions);
          } else {
            moveInfo = this.model.act(state);
          }

          return moveInfo.sorted;
        },
      });

      trainingStats.totalScore += gameManager.score;
      trainingStats.bestScore = Math.max(trainingStats.bestScore, gameManager.score);
      trainingStats.totalMoves += result.numSteps;
      trainingStats.totalInvalid += result.numInvalidSteps;
    }

    trainingStats.averageScore = trainingStats.totalScore / numberOfGames;
    trainingStats.averageMoves = trainingStats.totalMoves / numberOfGames;
    trainingStats.averageInvalid = trainingStats.totalInvalid / numberOfGames;

    console.debug(`"Learning"`);

    await this.qlearn.learn({
      reshapeState: (state) => state.reshape([16]),
      predict: (input) => this.model.predict(input),
      fit: (inputs, outputs) => this.model.fit(inputs, outputs),
    });

    return { ...trainingStats, epsilon: this.qlearn.config.epsilon };
  }
}

export function firstValidMoveOf(moveList: InternalMove[], game: GameState) {
  let gameCopy = clone(game?.serialize?.() || game);

  for (let move of moveList) {
    let { wasMoved } = imitateMove(gameCopy, ALL_MOVES[move]);

    if (wasMoved) {
      return move;
    }
  }

  throw new Error('No moves are valid, is the game over?');
}

export function moveAndCalculateReward(move: InternalMove, currentGame: GameState) {
  let previousGame = clone(currentGame);

  executeMove(currentGame, ALL_MOVES[move]);

  let scoreChange = currentGame.score - previousGame.score;
  let moveData = {
    scoreChange,
    wasMoved: !isEqual(previousGame.grid.cells, currentGame.grid.cells),
    currentScore: currentGame.score,
    previousScore: previousGame.score,
    over: currentGame.over,
    won: currentGame.won,
    nextState: gameToTensor(currentGame),
  };

  if (!moveData.wasMoved) {
    // strongly discourage invalid moves
    return { reward: -1.0, ...moveData };
  }

  let grouped = groupByValue(previousGame);
  let newGrouped = groupByValue(currentGame);

  let highest = Math.max(...Object.keys(grouped).map(parseInt));
  let newHighest = Math.max(...Object.keys(newGrouped).map(parseInt));

  // highest two were merged, we have a new highest
  if (newHighest > highest) {
    return { reward: 1, ...moveData };
  }

  if (currentGame.score > previousGame.score) {
    return { reward: 0.1, ...moveData };
  }

  // next score is equal to current
  // it's possible that we need to do something that doesn't
  // change our score before getting to something good
  return { reward: 0.5, ...moveData };
}

function randomMoves(numActions: number) {
  let result: number[] = [];
  let generateMove = () => random.int(0, numActions - 1);

  while (result.length < 4) {
    let move = generateMove();

    if (!result.includes(move)) {
      result.push(move);
    }
  }

  return { sorted: result };
}
