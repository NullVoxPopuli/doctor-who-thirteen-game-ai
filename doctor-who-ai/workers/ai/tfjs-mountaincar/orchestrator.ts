import * as tf from '@tensorflow/tfjs';

import { Memory } from '../learning/memory';
import { gameToTensor, clone } from '../utils';
import { fakeGameFrom } from '../game';
import { moveAndCalculateReward, firstValidMoveOf } from '../game-trainer';
import { MOVE_NAMES } from '../consts';

import type { Agent } from '../learning/agent';

const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.9;
const LAMBDA = 0.000001;
// const LAMBDA = 0.00001;

const NUM_ACTIONS = 4;
const NUM_STATES = 16;

let totalMoves = 0;
let totalInvalidMoves = 0;
let totalReward = 0;
let totalTrainedGames = 0;

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

      this.memory.add([inputs, move, reward, nextState]);

      step += 1;
      totalMoves++;
      totalReward += reward;
      this.steps += 1;

      if (!wasMoved) {
        invalidMoves++;
        totalInvalidMoves++;
      }

      if (totalMoves % 1000 === 0) {
        console.log('Replaying...');
        await this.replay();
      }

      // console.group(
      //   `Move: ${totalMoves} = ${MOVE_NAMES[move]} ${wasMoved ? 'succeeded  ' : 'was invalid'} ` +
      //     `${Math.round(
      //       ((totalMoves - totalInvalidMoves) / totalMoves) * 100
      //     )}% total valid moves.` +
      //     '\n' +
      //     `Epsilon: ${this.eps} | Reward: ${reward} | Average: ${
      //       Math.round((totalReward / totalMoves) * 100) / 100
      //     } \n` +
      //     `Score: ${gameManager.score} @ ${step} moves. ` +
      //     `Move % valid: ${Math.round(((step - invalidMoves) / step) * 100)}`
      // );
      // inputs.print();
      // state.print();
      // console.groupEnd();
    }

    totalTrainedGames++;

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
    // Sample from memory
    const batch = this.memory.recallRandomly(2000); // half of memory, max 2 games
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
