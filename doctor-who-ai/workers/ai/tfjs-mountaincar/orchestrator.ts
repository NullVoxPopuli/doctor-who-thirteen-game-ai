import * as tf from '@tensorflow/tfjs';

import { Memory } from '../learning/memory';
import { gameToTensor, clone } from '../utils';
import { fakeGameFrom } from '../game';
import { moveAndCalculateReward, firstValidMoveOf } from '../game-trainer';
import { MOVE_NAMES } from '../consts';

import type { Agent } from '../learning/agent';
import type { GameMemory, MoveMemory } from '../game-trainer';

const MIN_EPSILON = 0.00001;
const MAX_EPSILON = 0.3;
const LAMBDA = 0.00001;
// const LAMBDA = 0.00001;

const NUM_ACTIONS = 4;
const NUM_STATES = 16;

let totalMoves = 0;
let totalInvalidMoves = 0;
let totalReward = 0;
let totalTrainedGames = 0;

export class Orchestrator {
  declare memory: Memory<GameMemory>;
  declare model: Agent;
  declare eps: number;
  declare steps: number;
  declare maxStepsPerGame: number;
  declare discountRate: number;

  constructor(model: Agent, memory: Memory<any>, discountRate: number, maxStepsPerGame: number) {
    this.model = model;
    this.memory = memory;

    // The exploration parameter
    this.eps = MAX_EPSILON;

    // Keep tracking of the elapsed steps
    this.steps = 0;
    this.maxStepsPerGame = maxStepsPerGame;

    this.discountRate = discountRate;
  }

  async run(originalGame: Game2048, gameNumber: number) {
    let step = 0;
    let invalidMoves = 0;

    let clonedGame = clone(originalGame);
    let gameManager = fakeGameFrom(clonedGame);
    let nextState: tf.Tensor2D;

    let moveMemory = new Memory<MoveMemory>(10000);

    while (!gameManager.over) {
      // Interaction with the environment
      let inputs = gameToTensor(gameManager);
      // while learning, we want to throw invalid moves at the neural network
      let moveInfo = this.model.act(inputs.reshape([16]), this.eps);
      let move = moveInfo.sorted[0];
      let { reward, state, over, wasMoved } = moveAndCalculateReward(move, gameManager);

      // account for the network getting stuck choosing an incorrect direction
      while (!wasMoved) {
        moveInfo = this.model.act(inputs.reshape([16]), Infinity);
        move = moveInfo.sorted[0];

        ({ reward, state, over, wasMoved } = moveAndCalculateReward(move, gameManager));
      }

      nextState = state;

      if (over) {
        nextState = undefined;
      }

      moveMemory.add([inputs, move, reward, nextState]);

      step += 1;
      totalMoves++;
      totalReward += reward;
      this.steps += 1;

      if (!wasMoved) {
        invalidMoves++;
        totalInvalidMoves++;
      }
    }

    this.memory.add({ totalReward, moveMemory });

    totalTrainedGames++;

    if (totalTrainedGames % 100 === 0) {
      console.log('Replaying...');
      await this.replay();
    }

    // TODO: change epsilon based on the percent of invalid moves
    // Exponentially decay the exploration parameter
    this.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps);

    // reset epsilon every 10k games
    if (totalTrainedGames % 10000 === 0) {
      this.eps = MAX_EPSILON;
    }

    return {
      score: gameManager.score,
      totalMoves,
      moves: step,
      eps: this.eps,
    };
  }

  async replay() {
    let games = this.memory.recallTopBy((item) => item.totalReward, 0.1);

    // smallest first, so we do the biggest rewards later, just in case that matters?
    games = games.reverse();

    for (let game of games) {
      let rewardMultiplier = (games.indexOf(game) + 1) / games.length;

      // Sample from memory
      const batch = game.moveMemory.recallRandomly(1000);
      const states = batch.map(([state, , ,]) => state);
      const nextStates = batch.map(([, , , nextState]) =>
        nextState ? nextState : tf.zeros([NUM_STATES])
      );
      // Predict the values of each action at each state
      const qsa = states.map((state) => this.model.predict(state.reshape([16])));
      // Predict the values of each action at each next state
      const qsad = nextStates.map((nextState) => this.model.predict(nextState.reshape([16])));

      let x = new Array().fill(0);
      let y = new Array().fill(0);

      // Update the states rewards with the discounted next states rewards
      batch.forEach(([state, action, reward, nextState], index) => {
        const currentQ = qsa[index];

        if (nextState) {
          currentQ[action] = reward * rewardMultiplier + this.discountRate * qsad[index].max();
        } else {
          currentQ[action] = reward * rewardMultiplier;
        }

        x.push(state.dataSync());
        y.push(currentQ.dataSync());
      });

      // Clean unused tensors
      qsa.forEach((state) => state.dispose());
      qsad.forEach((state) => state.dispose());

      // Reshape the batches to be fed to the network
      let inputs = tf.tensor2d(x, [x.length, NUM_STATES]);
      let outputs = tf.tensor2d(y, [y.length, NUM_ACTIONS]);

      // Learn the Q(s, a) values given associated discounted rewards
      await this.model.fit(inputs, outputs);

      inputs.dispose();
      outputs.dispose();
    }
  }
}
