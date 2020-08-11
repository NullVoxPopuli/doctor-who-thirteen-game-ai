import tf from '@tensorflow/tfjs';
import random from 'random';

import { QLearn } from './learning/qlearn';

import type { Config } from './learning/types';
import type { DirectionKey, InternalMove } from '../consts';

import { ALL_MOVES } from '../consts';
import { clone, groupByValue, gameToTensor, isEqual, printGame } from './utils';
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

let lastMove = -1;

export class GameTrainer {
  declare config: Required<Config>;
  declare model: Model;
  declare qlearn: QLearn;
  declare gameQueue: GameManager[];

  constructor(network: tf.LayersModel, config: Config) {
    this.config = { ...defaultConfig, ...config };
    this.model = new Model(network);
    this.qlearn = new QLearn(config);
    this.gameQueue = [];
  }

  async getMove(game: GameState): Promise<any> {
    let inputs = gameToTensor(game);
    let moveInfo = this.model.act(reshape(inputs));
    let validMove = firstValidMoveOf(moveInfo.sorted, game);

    let move = ALL_MOVES[validMove];
    let rewardInfo = moveAndCalculateReward(validMove, fakeGameFrom(game));

    return { move, rewardInfo };
  }

  async train(_: GameState, numberOfGames = 1) {
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

    if (this.gameQueue.length === 0) {
      let originGame = fakeGameFrom({ score: 0, grid: { size: 4 } });

      originGame.addStartTiles();

      this.gameQueue.unshift(originGame);
    }

    let gameManager = this.gameQueue.pop();

    while (gameManager) {
      printGame(gameManager);

      let result = await this.qlearn.playOnce<GameManager, tf.Tensor2D, InternalMove, RewardInfo>({
        game: gameManager,
        getState: (game) => reshape(gameToTensor(game)),
        isGameOver: (game) => game.over,
        isValidAction: (rewardInformation) => rewardInformation.wasMoved,
        getReward: (game, action) => moveAndCalculateReward(action, game),
        takeAction: (game, action) => executeMove(game, ALL_MOVES[action]),
        getRankedActions: (game, state, useExternalAction) => {
          let moveInfo;

          // if we're approaching new territory, let's introduce some randomness
          // to make sure we fully explore the new space
          let needsLearning = game.score > trainingStats.averageScore;
          let useRandom = needsLearning && Math.random() < 0.5;

          if (useExternalAction || useRandom) {
            // moveInfo = guidedMove(4, game);
            moveInfo = randomMoves(this.config.numActions);

            if (lastMove === moveInfo.sorted[0]) {
              moveInfo.sorted.push(moveInfo.sorted.shift() || 0);
            }
          } else {
            moveInfo = this.model.act(state);
            lastMove = moveInfo.sorted[0];
          }

          if (this.gameQueue.length < 1000) {
            let [, secondBestAction, third, fourth] = moveInfo.sorted;

            this.gameQueue.unshift(forkGameWithAction(game, secondBestAction));
            this.gameQueue.unshift(forkGameWithAction(game, third));
            this.gameQueue.unshift(forkGameWithAction(game, fourth));
          }

          return moveInfo.sorted;
        },
        afterGame: async (gameInfo) => {
          console.log(
            `Finished game ${scores.length} with ${gameInfo.game.score}. ${this.gameQueue.length} remaining`
          );

          await this.qlearn.learnFromGame(gameInfo, {
            reshapeState: (state) => reshape(state),
            predict: (input) => this.model.predict(input),
            fit: (inputs, outputs) => this.model.fit(inputs, outputs),
          });
        },
      });

      scores.push(gameManager.score);
      trainingStats.totalScore += gameManager.score;
      trainingStats.bestScore = Math.max(trainingStats.bestScore, gameManager.score);
      trainingStats.minScore = Math.min(trainingStats.minScore, gameManager.score);
      trainingStats.medianScore = scores.sort()[Math.round((scores.length - 1) / 2)];
      trainingStats.totalMoves += result.numSteps;
      trainingStats.totalInvalid += result.numInvalidSteps;
      trainingStats.averageScore = trainingStats.totalScore / numberOfGames;
      trainingStats.averageMoves = trainingStats.totalMoves / numberOfGames;

      // if (scores.length % this.config.gameMemorySize === 0) {
      //   console.time('Learning');
      //   await this.qlearn.learn({
      //     reshapeState: (state) => reshape(state),
      //     predict: (input) => this.model.predict(input),
      //     fit: (inputs, outputs) => this.model.fit(inputs, outputs),
      //   });
      //   console.timeEnd('Learning');
      // }

      if (scores.length % numberOfGames === 0) {
        break;
      }

      gameManager = this.gameQueue.pop();
    }

    trainingStats.averageInvalid = trainingStats.totalInvalid / numberOfGames;

    console.log('# Games: ', scores.length);

    return { ...trainingStats, epsilon: this.qlearn.config.epsilon };
  }
}

function forkGameWithAction(game: GameManager, action: InternalMove) {
  let forkedGame = fakeGameFrom(game.serialize());

  executeMove(forkedGame, ALL_MOVES[action]);

  return forkedGame;
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
    nextState: gameToTensor(nextGame),
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
