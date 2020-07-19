import tf from '@tensorflow/tfjs';

import { Agent } from './learning/agent';
import { Memory } from './learning/memory';

import type { Config } from './learning/types';
import type { DirectionKey, InternalMove } from './consts';

import { ALL_MOVES, MOVE_KEY_MAP } from './consts';
import { clone, groupByValue, gameTo1DArray, isEqual } from './utils';
import { imitateMove, executeMove, fakeGameFrom } from './game';

const defaultConfig = {
  minEpsilon: 0.001,
  maxEpsilon: 0.1,
  epsilonDecaySpeed: 0.1,
};

type MoveMemory = {
  state: tf.Tensor1D;
  action: DirectionKey;
  reward: number;
  nextState: tf.Tensor1D | null;
};

type GameMemory = {
  totalReward: number;
  moveMemory: Memory<MoveMemory>;
};

export class GameTrainer {
  declare memory: Memory<GameMemory>;
  declare config: Required<Config>;
  declare agent: Agent;
  declare model: tf.LayersModel;
  declare epsilon: number;

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

    let memoryForGame = new Memory<MoveMemory>(this.config.moveMemorySize);

    while (!gameManager.over) {
      let inputs = gameToTensor(gameManager);
      let moveIndex = await agent.step(inputs);
      let move = ALL_MOVES[moveIndex];

      let { reward, wasMoved } = moveAndCalculateReward(move, gameManager);

      executeMove(gameManager, move);

      let internalMove = MOVE_KEY_MAP[move];

      if (!wasMoved) {
        totalNonMoves += 1;
      }

      let previousGame = clone(gameManager);
      let nextState = null;
      let currentState = gameToTensor(gameManager);
      let rankedMoves = await this.agent.act(currentState, this.epsilon);

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

      memoryForGame.add({ reward, nextState, state: currentState, action: move });
    }

    this.epsilon = decayEpsilon(this.config, moves);

    this.memory.add({
      totalReward,
      moveMemory: memoryForGame,
    });

    await this.replay();

    return {
      moves,
      score: gameManager.score,
      time: new Date().getDate() - start,
    };
  }

  /**
   * Trains the network on the last 20% (arbitrary)
   * best games
   *
   */
  replay() {
    // Sample from memory
    let games = this.memory.recallTopBy((item) => item.totalReward);

    for (let game of games) {
      let batch = game.moveMemory.recallRandomly(100);
      const states = batch.map(([state, , ,]) => state);
      const nextStates = batch.map(([, , , nextState]) =>
        nextState ? nextState : tf.zeros([this.config.numActions])
      );
      // Predict the values of each action at each state
      const qsa = states.map((state) => this.model.predict(state));
      // Predict the values of each action at each next state
      const qsad = nextStates.map((nextState) => this.model.predict(nextState));

      let x = new Array();
      let y = new Array();

      // Update the states rewards with the discounted next states rewards
      batch.forEach(([state, action, reward, nextState], index) => {
        const currentQ = qsa[index];

        currentQ[action] = nextState ? reward + 0.01 * qsad[index].max() : reward;
        x.push(state.dataSync());
        y.push(currentQ.dataSync());
      });

      // Clean unused tensors
      qsa.forEach((state) => state.dispose());
      qsad.forEach((state) => state.dispose());

      // Reshape the batches to be fed to the network
      x = tf.tensor2d(x, [x.length, 16]);
      y = tf.tensor2d(y, [y.length, this.config.numActions]);

      // Learn the Q(s, a) values given associated discounted rewards
      await this.model.fit(x, y);

      x.dispose();
      y.dispose();
    }
  }
}

function decayEpsilon(config: Required<Config>, steps: number) {
  let { minEpsilon, maxEpsilon, epsilonDecaySpeed } = config;

  config.epsilon = minEpsilon + (maxEpsilon - minEpsilon) * Math.exp(-epsilonDecaySpeed * steps);

  return config.epsilon;
}

function gameToTensor(game: Game2048) {
  let result = [];
  let cells = game.grid.cells;

  for (let i = 0; i < cells.length; i++) {
    // result[i] = [];

    for (let j = 0; j < cells.length; j++) {
      let cell = cells[i][j];

      let value = cell?.value || 0;
      let k = Math.log(value) || 0;

      // result[i][j][k] = 1;
      // result[i][j] = k;
      result.push(k);
    }
  }

  return tf.tensor1d(result);
}

function moveAndCalculateReward(move: DirectionKey, currentGame: Game2048) {
  let previousGame = clone(currentGame);

  executeMove(currentGame, move);

  let scoreChange = currentGame.score - previousGame.score;
  let moveData = {
    scoreChange,
    wasMoved: scoreChange !== 0,
    currentScore: currentGame.score,
    previousScore: previousGame.score,
  };

  if (!moveData.wasMoved) {
    // strongly discourage invalid moves
    return { reward: -1, ...moveData };
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
    return { reward: 0.5, ...moveData };
  }

  // next score is equal to current
  // it's possible that we need to do something that doesn't
  // change our score before getting to something good
  return { reward: 0, ...moveData };
}
