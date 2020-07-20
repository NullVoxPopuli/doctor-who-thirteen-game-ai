import * as tf from '@tensorflow/tfjs';

import { Memory } from '../learning/memory';
import { gameToTensor, clone } from '../utils';
import { fakeGameFrom } from '../game';
import { moveAndCalculateReward } from '../game-trainer';
import { MOVE_NAMES } from '../consts';

import type { Agent } from '../learning/agent';

const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.9;
const LAMBDA = 0.001;

const NUM_ACTIONS = 4;
const NUM_STATES = 16;

export class Orchestrator {
  declare memory: Memory<[tf.Tensor2D, number, number, tf.Tensor2D]>;
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

    while (!gameManager.over && step < 1000) {
      // Interaction with the environment
      let inputs = gameToTensor(gameManager);
      let move = this.model.act(inputs.reshape([16]), this.eps);
      let { reward, state, over, wasMoved } = moveAndCalculateReward(move, gameManager);

      nextState = state;

      if (over) {
        nextState = undefined;
      }

      this.memory.add([inputs, move, reward, nextState]);

      step += 1;
      this.steps += 1;

      if (!wasMoved) {
        invalidMoves++;
      }

      if (step % 100 === 0) {
        console.log('Replaying...');
        await this.replay();
      }

      console.group(
        `${gameNumber} | ${MOVE_NAMES[move]} : ${wasMoved} -- ` +
          `Score: ${gameManager.score} @ ${step} moves. ` +
          `% valid ${Math.round(((step - invalidMoves) / step) * 100)} \n` +
          `#invalid ${invalidMoves} -- ` +
          `eps: ${this.eps} -- reward: ${reward}`
      );
      inputs.print();
      state.print();
      console.groupEnd();
    }

    // Exponentially decay the exploration parameter
    this.eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps);

    return {
      score: gameManager.score,
      moves: step,
      eps: this.eps,
    };
  }

  async replay() {
    // Sample from memory
    const batch = this.memory.recallRandomly(50); // half
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
