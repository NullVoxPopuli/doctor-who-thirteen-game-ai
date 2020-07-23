import tf from '@tensorflow/tfjs';

import { Agent } from './learning/agent';
import { Memory } from './learning/memory';

import type { Config } from './learning/types';
import type { DirectionKey, InternalMove } from './consts';

import { ALL_MOVES, MOVE_KEY_MAP } from './consts';
import { clone, groupByValue, gameToTensor, isEqual } from './utils';
import { imitateMove, executeMove, fakeGameFrom } from './game';
import { Orchestrator } from './tfjs-mountaincar/orchestrator';

const defaultConfig = {
  minEpsilon: 0.001,
  maxEpsilon: 0.1,
  epsilonDecaySpeed: 0.1,
};

export type MoveMemory = [tf.Tensor2D, number, number, tf.Tensor2D];

export type GameMemory = {
  totalReward: number;
  moveMemory: Memory<MoveMemory>;
};

export class GameTrainer {
  declare memory: Memory<GameMemory>;
  declare config: Required<Config>;
  declare agent: Agent;
  declare model: tf.LayersModel;
  declare epsilon: number;
  declare orchestrator: Orchestrator;

  trainingStats = {
    totalGames: 0,
    totalMoves: 0,
    averageMovesPerGame: 0,
  };

  constructor(network: tf.LayersModel, config: Config) {
    this.config = { ...defaultConfig, ...config };
    this.model = network;
    this.agent = new Agent(network, config);
    this.memory = new Memory(config.gameMemorySize);
    this.epsilon = config.epsilon;
    this.orchestrator = new Orchestrator(
      this.agent,
      this.memory,
      this.config.epsilonDecaySpeed,
      400
    );
  }

  async getMove(game: Game2048): Promise<DirectionKey> {
    let inputs = gameToTensor(game);
    let moveInfo = this.agent.act(inputs.reshape([16]));
    let validMove = firstValidMoveOf(moveInfo.sorted, game);

    return ALL_MOVES[validMove];
  }

  async train(originalGame: Game2048) {}
}

export function firstValidMoveOf(moveList: InternalMove[], game: Game2048) {
  let gameCopy = clone(game?.serialize?.() || game);

  for (let move of moveList) {
    let { wasMoved } = imitateMove(gameCopy, ALL_MOVES[move]);

    if (wasMoved) {
      return move;
    }
  }

  throw new Error('No moves are valid, is the game over?');
}

function decayEpsilon(config: Required<Config>, steps: number) {
  let { minEpsilon, maxEpsilon, epsilonDecaySpeed } = config;

  config.epsilon = minEpsilon + (maxEpsilon - minEpsilon) * Math.exp(-epsilonDecaySpeed * steps);

  return config.epsilon;
}

export function moveAndCalculateReward(move: InternalMove, currentGame: Game2048) {
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
    state: gameToTensor(currentGame),
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
  return { reward: 0, ...moveData };
}
