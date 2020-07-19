import { Memory } from '../learning/memory';
import { gameToTensor, clone } from '../utils';
import { imitateMove, executeMove, fakeGameFrom } from '../game';
import { moveAndCalculateReward } from '../game-trainer';

import * as tf from '@tensorflow/tfjs';

import type { Agent } from '../learning/agent';

const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.2;
const LAMBDA = 0.01;

const NUM_ACTIONS = 4;
const NUM_STATES = 16;

export class Orchestrator {
  declare memory: Memory<[tf.Tensor2D, number, number, tf.Tensor2D]>;
  declare model: Agent;
  declare eps: number;
  declare steps: number;
  declare maxStepsPerGame: number;
  declare discountRate: number;

  /**
   * @param {MountainCar} mountainCar
   * @param {Model} model
   * @param {Memory} memory
   * @param {number} discountRate
   * @param {number} maxStepsPerGame
   */
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

  /**
   * @param {number} position
   * @returns {number} Reward corresponding to the position
   */

  async run(originalGame: Game2048) {
    let step = 0;

    let clonedGame = clone(originalGame);
    let gameManager = fakeGameFrom(clonedGame);

    while (step < this.maxStepsPerGame) {
      // Interaction with the environment
      let inputs = gameToTensor(gameManager);
      let move = this.model.act(inputs.reshape([16]), this.eps);
      let { reward, state, over } = moveAndCalculateReward(move, gameManager);
      let nextState = state;

      if (over) {
        nextState = null;
      }

      this.memory.add([inputs, move, reward, nextState]);

      this.steps += 1;
      // Exponentially decay the exploration parameter
      this.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps);

      step += 1;
    }

    await this.replay();
  }

  async replay() {
    // Sample from memory
    const batch = this.memory.recallTopBy(([, , reward]) => reward);
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

      currentQ[action] = nextState ? reward + this.discountRate * qsad[index].max() : reward;
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
