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
    let moveInfo = this.model.act(reshape(inputs));
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

      let result = await this.qlearn.playOnce<GameManager, tf.Tensor2D, InternalMove, RewardInfo>({
        game: gameManager,
        getState: (game) => reshape(gameToTensor(game)),
        isGameOver: (game) => game.over,
        isValidAction: (rewardInformation) => rewardInformation.wasMoved,
        getReward: (game, action) => moveAndCalculateReward(action, game),
        getRankedActions: (_game, state, useExternalAction) => {
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
      reshapeState: (state) => reshape(state),
      predict: (input) => this.model.predict(input),
      fit: (inputs, outputs) => this.model.fit(inputs, outputs),
    });

    return { ...trainingStats, epsilon: this.qlearn.config.epsilon };
  }
}

function reshape(tensor: tf.Tensor) {
  return tensor.reshape([4, 4, 1]);
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

  let distanceOld = totalDistance(previousGame);
  let distanceNew = totalDistance(currentGame);

  //    hopefully smaller / hopefully bigger
  //           -> gets smaller when lots of stuff merged
  //           -> gets smaller when fewer cells than previous
  //           -> gets bigger when moving in the wrong direction
  //           -> is 1 when no change
  //
  //   subtracting from one inverts the above.
  //
  //   NOTE: this *can* be negative.
  //         - should we allow for small negative values?
  //           (round to nearest 0.1?)
  return 1 - distanceNew / distanceOld;
}

type GroupedPositions = { [key: number]: CellPosition[] };

function totalDistance(game: GameState) {
  let grouped: GroupedPositions = game.grid.cells.flat().reduce((grouped, cell) => {
    if (!cell) {
      return grouped;
    }

    grouped[cell.value] = grouped[cell.value] = [];

    grouped[cell.value].push(cell.position);

    return grouped;
  }, {} as GroupedPositions);

  let distance = 0; // Best?

  for (let [value, positions] of Object.entries(grouped)) {
    // get total collective distance between each position set?
  }
}

function randomMoves(numActions: number) {
  let result: InternalMove[] = [];
  let generateMove = () => random.int(0, numActions - 1);

  while (result.length < 4) {
    let move = generateMove() as InternalMove;

    if (!result.includes(move)) {
      result.push(move);
    }
  }

  return { sorted: result };
}
