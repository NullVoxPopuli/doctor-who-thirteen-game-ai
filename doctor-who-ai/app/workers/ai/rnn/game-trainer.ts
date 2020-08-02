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

  async getMove(game: GameState): Promise<any> {
    let inputs = gameToTensor(game);
    let moveInfo = this.model.act(reshape(inputs));
    let validMove = firstValidMoveOf(moveInfo.sorted, game);

    let move = ALL_MOVES[validMove];
    let rewardInfo = moveAndCalculateReward(validMove, fakeGameFrom(game));

    return { move, rewardInfo };
  }

  async train(originalGame: GameState, numberOfGames = 1) {
    let scores = [];
    let trainingStats = {
      totalMoves: 0,
      totalInvalid: 0,
      totalScore: 0,
      averageScore: 0,
      averageMoves: 0,
      medianScore: 0,
      averageInvalid: 0,
      minScore: Infinity,
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
        getRankedActions: (game, state, useExternalAction) => {
          let moveInfo;

          let needsLearning = game.score > trainingStats.medianScore;
          let useRandom = needsLearning && Math.random() < 0.2;

          if (useExternalAction || useRandom) {
            // moveInfo = guidedMove(4, game);
            moveInfo = randomMoves(this.config.numActions);
          } else {
            moveInfo = this.model.act(state);
          }

          return moveInfo.sorted;
        },
      });

      scores.push(gameManager.score);
      trainingStats.totalScore += gameManager.score;
      trainingStats.bestScore = Math.max(trainingStats.bestScore, gameManager.score);
      trainingStats.minScore = Math.min(trainingStats.minScore, gameManager.score);
      trainingStats.medianScore = scores.sort()[Math.round((scores.length - 1) / 2)];
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
  // convo: 4, 4, 1
  return tensor.reshape([16]);
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

export function moveAndCalculateReward(move: InternalMove, game: Game2048) {
  let previousGame = clone(game);
  let nextGame = fakeGameFrom(game.serialize());

  executeMove(game, ALL_MOVES[move]);

  // used for calculating distances. Can't have new tile.
  executeMove(nextGame, ALL_MOVES[move], true);

  let scoreChange = nextGame.score - previousGame.score;
  let moveData = {
    scoreChange,
    wasMoved: !isEqual(previousGame.grid.cells, nextGame.grid.cells),
    currentScore: nextGame.score,
    previousScore: previousGame.score,
    over: nextGame.over,
    won: nextGame.won,
    nextState: gameToTensor(nextGame),
  };

  if (!moveData.wasMoved) {
    // strongly discourage invalid moves
    return { reward: -1.0, ...moveData };
  }

  let numTilesOld = numTiles(previousGame);
  let numTilesNew = numTiles(nextGame);
  let numTileDiff = numTilesNew - numTilesOld;
  let distanceOld = totalDistance(previousGame);
  // NOTE: there are never less than two tiles
  let distanceNew = totalDistance(nextGame);

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
  let reward = 1 - distanceNew / distanceOld;

  // console.log({ reward, distanceNew, distanceOld });
  // debugger;

  return { reward: reward || 0, ...moveData, distanceNew, distanceOld };
}

type GroupedPositions = { [key: number]: CellPosition[] };

function numTiles(game: GameState) {
  return game.grid.cells.flat().filter(Boolean).length;
}

function totalDistance(game: GameState) {
  let grouped: GroupedPositions = game.grid.cells.flat().reduce((grouped, cell) => {
    if (!cell) {
      return grouped;
    }

    grouped[cell.value] = grouped[cell.value] || [];

    grouped[cell.value].push(cell.position || { x: cell.x, y: cell.y });

    return grouped;
  }, {} as GroupedPositions);

  let distance = 0; // Best?

  for (let [value, positions] of Object.entries(grouped)) {
    // need at least two things to calculate position
    if (positions.length < 2) {
      continue;
    }

    // create a list of pairs to find distances between
    let pairs = [];

    for (let i = 0; i < positions.length; i++) {
      for (let j = i + 1; j < positions.length; j++) {
        pairs.push([positions[i], positions[j]]);
      }
    }

    for (let [a, b] of pairs) {
      let localDistance = Math.hypot(b.x - a.x, b.y - a.y);

      // harshly punish things farther away
      distance += Math.pow(localDistance, 2);
    }
  }

  return distance;
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
