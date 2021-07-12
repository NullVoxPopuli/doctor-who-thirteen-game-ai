import * as tf from '@tensorflow/tfjs';
import { AdamOptimizer } from '@tensorflow/tfjs';
import random from 'random';

import { QLearn } from './learning/qlearn';

import type { Config } from './learning/types';
import type { DirectionKey, InternalMove } from '../consts';

import { ALL_MOVES } from '../consts';
import { clone, groupByValue, gameToTensor, isEqual, printGame } from './utils';
import { imitateMove, executeMove, fakeGameFrom } from '../game';
import { Model, act } from './model';

import type { GameManager } from 'ai/rnn/vendor/app.map-worker-edition';

const defaultConfig = {
  // minEpsilon: 0.001,
  // maxEpsilon: 0.1,
  // epsilonDecaySpeed: 0.1,
  epsilon: 0.05,
  minEpsilon: 0.0001,
  maxEpsilon: 0.8,
  epsilonDecaySpeed: 0.05,
  learningDiscount: 0.95,
  learningRate: 0.95,
  numActions: 4,
  numInputs: 16,
  moveMemorySize: 1000,
};

type RewardInfo = {
  reward: number;
  scoreChange: number;
  wasMoved: boolean;
  currentScore: number;
  previousScore: number;
  over: boolean;
  won: boolean;
  nextState: Game2048;
};

let lastMove = -1;

interface Args {
  network: tf.LayersModel;
  createNetwork: () => tf.LayersModel;
}

export class GameTrainer {
  declare config: Required<Config>;
  declare target: Model;
  declare qlearn: QLearn<Game2048>;
  declare gameQueue: GameManager[];

  declare optimizer: AdamOptimizer;

  constructor({ network, createNetwork }: Args) {
    this.config = { ...defaultConfig };
    this.target = new Model(network);

    network.trainable = false;

    this.optimizer = tf.train.adam(0.01);

    this.qlearn = new QLearn<Game2048>(this.config, { createNetwork, network });
    this.gameQueue = [];
  }

  async getMove(game: GameState): Promise<any> {
    let inputs = getStateTensor(game);
    let moveInfo = act(this.target.model, inputs);
    let validMove = firstValidMoveOf(moveInfo.sorted, game);

    let move = ALL_MOVES[validMove];
    let rewardInfo = moveAndCalculateReward(validMove, fakeGameFrom(game));

    return { move, rewardInfo };
  }

  async train(numberOfGames = 1) {
    let scores = [];
    let trainingStats = {
      totalMoves: 0,
      totalInvalid: 0,
      totalScore: 0,
      totalReward: 0,
      averageScore: 0,
      averageReward: 0,
      averageMoves: 0,
      medianScore: 0,
      averageInvalid: 0,
      minScore: Infinity,
      bestScore: 0,
    };

    if (this.gameQueue.length === 0) {
      let originGame = fakeGameFrom({ score: 0, grid: { size: 4 } });

      originGame.addStartTiles();

      this.gameQueue.unshift(originGame);
    }

    let gameManager = this.gameQueue.pop();

    while (gameManager) {
      if ((globalThis as any).printGames) {
        printGame(gameManager);
      }

      let result = await this.qlearn.playOnce<GameManager, InternalMove, RewardInfo>({
        game: gameManager,
        getState: (game) => game.serialize(),
        getStateTensor: getStateTensor,
        isGameOver: (game) => game.over,
        isValidAction: (rewardInformation) => rewardInformation.wasMoved,
        getReward: (game, action) => moveAndCalculateReward(action, game),
        takeAction: (game, action) => executeMove(game, ALL_MOVES[action]),
        getRankedActions: (game, state, useExternalAction, onlineNetwork) => {
          let moveInfo;
          let inputs = getStateTensor(state);

          // if we're approaching new territory, let's introduce some randomness
          // to make sure we fully explore the new space
          let needsLearning = game.score > trainingStats.averageScore;
          let useRandom = needsLearning && Math.random() < 0.5;

          if (useExternalAction || useRandom) {
            moveInfo = randomMoves(this.config.numActions);

            if (lastMove === moveInfo.sorted[0]) {
              moveInfo.sorted.push(moveInfo.sorted.shift() || 0);
            }
          } else {
            moveInfo = act(onlineNetwork, inputs);
            lastMove = moveInfo.sorted[0];
          }

          // if (this.gameQueue.length < 100) {
          //   let [, secondBestAction, third, fourth] = moveInfo.sorted;

          //   this.gameQueue.unshift(forkGameWithAction(game, secondBestAction));
          //   this.gameQueue.unshift(forkGameWithAction(game, third));
          //   this.gameQueue.unshift(forkGameWithAction(game, fourth));
          // }

          return moveInfo.sorted;
        },
        learnPeriodically: {
          getStateTensor,
          gamma: 0.01,
          batchSize: 200,
          optimizer: this.optimizer,
        },
        // afterGame: async (gameInfo) => {
        //   console.debug(
        //     `Finished game ${scores.length} with ${gameInfo.game.score}. ${this.gameQueue.length} remaining`
        //   );
        // },
      });

      scores.push(gameManager.score);
      trainingStats.totalReward += result.totalReward;
      trainingStats.totalScore += gameManager.score;
      trainingStats.totalMoves += result.numSteps;
      trainingStats.totalInvalid += result.numInvalidSteps;
      trainingStats.averageReward = trainingStats.totalReward / numberOfGames;
      trainingStats.averageScore = trainingStats.totalScore / numberOfGames;
      trainingStats.averageMoves = trainingStats.totalMoves / numberOfGames;

      trainingStats.bestScore = Math.max(...scores);
      trainingStats.minScore = Math.min(...scores);
      trainingStats.medianScore = scores.sort()[Math.round((scores.length - 1) / 2)];
      trainingStats.averageInvalid = trainingStats.totalInvalid / numberOfGames;

      if (scores.length % numberOfGames === 0) {
        console.debug('# Games: ', scores.length);
        console.table([trainingStats]);

        // console.time('Learning');

        // await this.qlearn.learn({
        //   getStateTensor,
        //   gamma: 0.9,
        //   batchSize: 200,
        //   optimizer: this.optimizer,
        // });

        // console.timeEnd('Learning');

        break;
      }

      gameManager = this.gameQueue.pop();
    }

    return trainingStats;
  }
}

function forkGameWithAction(game: GameManager, action: InternalMove) {
  let forkedGame = fakeGameFrom(game.serialize());

  executeMove(forkedGame, ALL_MOVES[action]);

  return forkedGame;
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
    nextState: nextGame.serialize(),
  };

  if (!moveData.wasMoved) {
    // strongly discourage invalid moves
    return { reward: -1.0, ...moveData };
  }

  // let reward = distanceBasedReward(previousGame, nextGame);
  // let reward = adjacentBasedReward(previousGame, nextGame);
  // let reward = scoreBasedReward(previousGame, nextGame);
  // let reward = mergeBasedReward(previousGame, nextGame);
  let reward = tileCountBasedReward(previousGame, nextGame);

  return { reward: reward || 0, ...moveData };
}

type GroupedPositions = { [key: number]: CellPosition[] };

function numTiles(game: GameState) {
  return game.grid.cells.flat().filter(Boolean).length;
}

function tileCountBasedReward(previousGame: Game2048, nextGame: Game2048) {
  let previous = numTiles(previousGame);
  let current = numTiles(nextGame);

  if (current < previous) {
    return 1;
  }

  if (current === previous) {
    return 0;
  }

  return -1;
}

function mergeBasedReward(previousGame: Game2048, nextGame: Game2048) {
  let groupedPrevious = gameGroupedByValue(previousGame);
  let groupedNext = gameGroupedByValue(nextGame);

  let values = Object.keys(groupedPrevious)
    .map(parseInt)
    .sort((a, z) => z - a);

  for (let value of values) {
    let previous = groupedPrevious[value];
    let next = groupedNext[value];

    if (!next) {
      return 1;
    }

    if (next.length < previous.length) {
      return 1;
    }
  }

  return 0;
}

function scoreBasedReward(previousGame: Game2048, nextGame: Game2048) {
  let reward = 0;

  if (previousGame.score < nextGame.score) {
    reward = 1;
  }

  return reward;
}

function adjacentBasedReward(previousGame: GameState, nextGame: GameState) {
  function isAdjacentToOneOf(cell: CellPosition, cells: CellPosition[]) {
    for (let other of cells) {
      if (cell.x === other.x) {
        if (cell.y === other.y + 1 || cell.y === other.y - 1) {
          return true;
        }
      } else if (cell.y === other.y) {
        if (cell.x === other.x + 1 || cell.x === other.x - 1) {
          return true;
        }
      }
    }

    return false;
  }

  function adjacentStatsFor(game: GameState) {
    let grouped = gameGroupedByValue(game);

    let totalNonAdjacent = 0;
    let totalAdjacent = 0;

    for (let [_value, positions] of Object.entries(grouped)) {
      let adjacent = 0;
      let nonAdjacent = 0;

      if (positions.length < 2) {
        continue;
      }

      for (let position of positions) {
        if (isAdjacentToOneOf(position, positions)) {
          adjacent += 1;
        } else {
          nonAdjacent += 1;
        }
      }

      totalNonAdjacent += nonAdjacent;
      totalAdjacent += adjacent;
    }

    return { totalNonAdjacent, totalAdjacent };
  }

  let previousStats = adjacentStatsFor(previousGame);
  let nextStats = adjacentStatsFor(nextGame);

  let reward = 0;

  if (previousStats.totalNonAdjacent > nextStats.totalNonAdjacent) {
    reward = 1;
  } else if (previousStats.totalNonAdjacent < nextStats.totalNonAdjacent) {
    reward = -1;
  }

  return reward;
}

function distanceBasedReward(previousGame: GameState, nextGame: GameState) {
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
  let diff = distanceOld - distanceNew;
  let avg = (distanceOld + distanceNew) / 2;
  let reward = diff / avg;

  return reward;
}

function gameGroupedByValue(game: GameState) {
  let grouped: GroupedPositions = game.grid.cells.flat().reduce((grouped, cell) => {
    if (!cell) {
      return grouped;
    }

    grouped[cell.value] = grouped[cell.value] || [];

    grouped[cell.value].push(cell.position || { x: cell.x, y: cell.y });

    return grouped;
  }, {} as GroupedPositions);

  return grouped;
}

function totalDistance(game: GameState) {
  let grouped = gameGroupedByValue(game);

  let distance = 0; // Best?

  for (let [value, positions] of Object.entries(grouped)) {
    let distanceForValue = 0;

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

      distanceForValue += localDistance;
    }

    // try to normalize if there are changes in the number of a specific tile
    distance += distanceForValue / positions.length;
  }

  return distance;
}

function randomMoves(numActions: number) {
  let result: InternalMove[] = [];
  let generateMove = () => random.int(0, numActions - 1);

  while (result.length < numActions) {
    let move = generateMove() as InternalMove;

    if (!result.includes(move)) {
      result.push(move);
    }
  }

  return { sorted: result };
}

function getStateTensor(state: Game2048 | Game2048[]): tf.Tensor {
  if (!Array.isArray(state)) {
    state = [state];
  }

  const numExamples = state.length;

  const buffer = tf.buffer([numExamples, 4, 4, 1]);

  for (let n = 0; n < numExamples; ++n) {
    let currentState = state[n];

    if (!currentState) {
      continue;
    }

    let cells = currentState.grid.cells;

    for (let i = 0; i < cells.length; i++) {
      for (let j = 0; j < cells.length; j++) {
        let cell = cells[i][j];

        let value = cell?.value || 0;
        // convert power of 2 to the power that 2 is to
        let k = value === 0 ? 0 : Math.log2(value);

        buffer.set(k, n, i, j, 0);
      }
    }
  }

  return buffer.toTensor();
}
