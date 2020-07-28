import * as tf from '@tensorflow/tfjs';

import { Memory } from './memory';

import type { Config } from './types';

interface RequiredRewardInfo {
  nextState: tf.Tensor;
  reward: number;
}

type LessonConfig<Game, State, Action, RewardInfo extends RequiredRewardInfo> = {
  game: Game;
  getState: (game: Game) => State;
  isGameOver: (game: Game) => boolean;
  getRankedActions: (game: Game, state: State, useExternalAction: boolean) => Action[];
  getReward: (game: Game, action: Action) => RewardInfo;
  isValidAction: (rewardInfo: RewardInfo) => boolean;
};

type LearningConfig<Prediction> = {
  reshapeState: (state: tf.Tensor) => tf.Tensor;
  predict: (input: tf.Tensor) => Prediction;
  fit: (inputs: tf.Tensor, outputs: tf.Tensor) => Promise<tf.History>;
};

type Maybe<T> = T | undefined;
type ActionMemory = [tf.Tensor, number, number, Maybe<tf.Tensor>];

type GameMemory = {
  totalReward: number;
  moveMemory: Memory<ActionMemory>;
};

export class QLearn {
  declare gameMemory: Memory<GameMemory>;
  declare config: Required<Config>;

  playCount = 0;

  constructor(config: Config) {
    this.config = {
      maxEpsilon: 0.9,
      minEpsilon: 0.001,
      epsilonDecaySpeed: 0.00001,
      ...config,
    };
    this.gameMemory = new Memory<GameMemory>(config.gameMemorySize);
  }

  async playOnce<
    Game,
    State extends tf.Tensor,
    Action extends number,
    RewardInfo extends RequiredRewardInfo
  >(lessonConfig: LessonConfig<Game, State, Action, RewardInfo>) {
    let { game, isGameOver, getState, getRankedActions, getReward, isValidAction } = lessonConfig;

    let moveMemory = new Memory<ActionMemory>(this.config.moveMemorySize);

    let totalReward = 0;
    let numSteps = 0;
    let numInvalidSteps = 0;

    while (!isGameOver(game)) {
      let inputs = getState(game);

      let useExternalAction = Math.random() < this.config.epsilon;
      let rankedActions = getRankedActions(game, inputs, useExternalAction);

      let action = rankedActions[0];

      let rewardInfo = getReward(game, action);

      if (!isValidAction(rewardInfo)) {
        numInvalidSteps++;

        // // skip the first, we already tried it
        for (let i = 1; i < rankedActions.length; i++) {
          action = rankedActions[i];

          let rewardInfo2 = getReward(game, action);

          if (isValidAction(rewardInfo2)) {
            break;
          }

          numInvalidSteps++;
        }
      }

      numSteps++;
      this.playCount++;

      let nextState = isGameOver(game) ? undefined : rewardInfo.nextState;

      moveMemory.add([inputs, action, rewardInfo.reward, nextState]);
    }

    decayEpsilon(this.config, this.playCount);

    this.gameMemory.add({ totalReward, moveMemory });

    return {
      numSteps,
      numInvalidSteps,
    };
  }

  async learn<Prediction extends tf.Tensor<tf.Rank>[] | tf.Tensor<tf.Rank>>(
    learningConfig: LearningConfig<Prediction>
  ) {
    let { reshapeState, predict, fit } = learningConfig;

    // let games = this.gameMemory.recallRandomly()
    let games = this.gameMemory.recallTopBy((item) => item.totalReward, 0.5);

    // smallest first, so we do the biggest rewards later, just in case that matters?
    // games = games.reverse();

    // try only learning from the highest reward game
    // games = [games[0]];

    for (let game of games) {
      let rewardMultiplier = (games.indexOf(game) + 1) / games.length;

      // Sample from memory
      const batch = game.moveMemory.recallRandomly(600);
      const states = batch.map(([state, , ,]) => state);
      const nextStates = batch.map(([, , , nextState]) =>
        nextState ? nextState : tf.zeros([this.config.numInputs])
      );

      // Predict the values of each action at each state
      const qsa = states.map((state) => predict(reshapeState(state)));

      // Predict the values of each action at each next state
      const qsad = nextStates.map((nextState) => predict(reshapeState(nextState)));

      let x = [].fill(0);
      let y = [].fill(0);

      // Update the states rewards with the discounted next states rewards
      batch.forEach(([state, action, reward, nextState], index) => {
        const currentQ = qsa[index];

        if (nextState) {
          currentQ[action] =
            reward * rewardMultiplier + this.config.epsilonDecaySpeed * qsad[index].max();
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
      let inputs = tf.tensor(x, [x.length, 4, 4, 1]);
      let outputs = tf.tensor2d(y, [y.length, this.config.numActions]);

      // Learn the Q(s, a) values given associated discounted rewards
      await fit(inputs, outputs);

      inputs.dispose();
      outputs.dispose();
    }
  }
}

function decayEpsilon(config: Required<Config>, steps: number) {
  let { minEpsilon, maxEpsilon, epsilonDecaySpeed } = config;

  config.epsilon = minEpsilon + (maxEpsilon - minEpsilon) * Math.exp(-epsilonDecaySpeed * steps);

  return config.epsilon;
}
