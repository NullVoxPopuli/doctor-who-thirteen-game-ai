import * as tf from '@tensorflow/tfjs';

import { Memory } from './memory';

import { decayEpsilon } from './utils';

import type { Config } from './types';
import { copyWeights } from '../model/tf-utils';

interface RequiredRewardInfo<State> {
  nextState: State;
  reward: number;
}

type Maybe<T> = T | undefined;

type LessonConfig<Game, State, Action, RewardInfo extends RequiredRewardInfo<State>> = {
  game: Game;
  getState: (game: Game) => State;
  getStateTensor: (state: State) => tf.Tensor;
  isGameOver: (game: Game) => boolean;
  getRankedActions: (
    game: Game,
    state: State,
    useExternalAction: boolean,
    onlineNetwork: tf.LayersModel
  ) => Action[];
  takeAction: (game: Game, action: Action) => void;
  getReward: (game: Game, action: Action) => RewardInfo;
  isValidAction: (rewardInfo: RewardInfo) => boolean;
  afterGame?: (gameInfo: { game: Game }) => Promise<void>;
  learnPeriodically?: LearningConfig<State>;
};

type LearningConfig<State> = {
  getStateTensor: (state: State) => tf.Tensor;
  gamma: number;
  batchSize: number;
  optimizer: tf.Optimizer;
};

interface NetworkArgs {
  network: tf.LayersModel;
  createNetwork: () => tf.LayersModel;
}

export class QLearn<State> {
  declare actionMemory: Memory<[State, number, number, Maybe<State>, boolean]>;
  declare config: Required<Config>;

  declare onlineNetwork: tf.LayersModel;
  declare targetNetwork: tf.LayersModel;

  playCount = 0;

  constructor(config: Config, { network, createNetwork }: NetworkArgs) {
    this.config = {
      maxEpsilon: 0.9,
      minEpsilon: 0.001,
      epsilonDecaySpeed: 0.00001,
      ...config,
    };
    this.actionMemory = new Memory(this.config.moveMemorySize);

    this.targetNetwork = network;
    this.targetNetwork.trainable = false;
    this.onlineNetwork = createNetwork();

    // this.onlineNetwork.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    // Copy weights to our online network
    copyWeights(this.onlineNetwork, this.targetNetwork);
  }

  decayEpsilon(iterations: number) {
    this.config.epsilon = decayEpsilon(this.config, iterations);
  }

  async playOnce<Game, Action extends number, RewardInfo extends RequiredRewardInfo<State>>(
    lessonConfig: LessonConfig<Game, State, Action, RewardInfo>
  ) {
    let {
      game,
      isGameOver,
      getState,
      getRankedActions,
      takeAction,
      getReward,
      isValidAction,
    } = lessonConfig;

    let totalReward = 0;
    let numSteps = 0;
    let numInvalidSteps = 0;

    while (!isGameOver(game)) {
      let state = getState(game);

      let useExternalAction = Math.random() < this.config.epsilon;
      let rankedActions = getRankedActions(game, state, useExternalAction, this.onlineNetwork);

      let action: Action | undefined;

      // explore the move space a little more than just playind would allow
      for (let possibleAction of rankedActions) {
        let rewardInfo = getReward(game, possibleAction);
        let isDone = isGameOver(game);
        let nextState = isDone ? undefined : rewardInfo.nextState;

        this.actionMemory.add([state, possibleAction, rewardInfo.reward, nextState, isDone]);

        // if (!action && isValidAction(rewardInfo)) {
        //   action = possibleAction;
        //   totalReward += rewardInfo.reward;
        //   break;
        // } else if (!action) {
        //   numInvalidSteps++;
        // }

        if (!isValidAction(rewardInfo)) {
          numInvalidSteps++;
        }

        if (!action) {
          action = possibleAction;
          totalReward += rewardInfo.reward;
          break;
        }

        // harshly punish repeated failure
        break;
      }

      takeAction(game, action || rankedActions[0]);

      if (lessonConfig.learnPeriodically && numSteps % 1000 === 0) {
        console.time('Learning');
        await this.learn(lessonConfig.learnPeriodically);
        console.timeEnd('Learning');
      }

      numSteps++;
      this.playCount++;
    }

    if (lessonConfig.afterGame) {
      await lessonConfig.afterGame({ game });
    }

    return {
      totalReward,
      numSteps,
      numInvalidSteps,
    };
  }

  async learn(learningConfig: LearningConfig<State>) {
    let { getStateTensor, gamma, batchSize, optimizer } = learningConfig;

    // Get a batch of examples from the replay buffer
    let batch = this.actionMemory.recallRandomly(batchSize);

    let lossFunction = () =>
      tf.tidy(() => {
        let states = getStateTensor(batch.map(([state]) => state));
        let actions = tf.tensor1d(
          batch.map(([, action]) => action),
          'int32'
        );

        let qs = this.onlineNetwork
          .apply(states, { training: true })
          .mul(tf.oneHot(actions, this.config.numActions))
          .sum(-1);

        let rewards = tf.tensor1d(batch.map(([, , reward]) => reward));
        let nextStates = getStateTensor(batch.map(([, , , nextState]) => nextState));

        // todo: targetnetwork
        let nextMaxQ = this.targetNetwork.predict(nextStates).max(-1);

        let doneMask = tf
          .scalar(1)
          .sub(tf.tensor1d(batch.map(([, , , , done]) => done)).asType('float32'));

        let targetQs = rewards.add(nextMaxQ.mul(doneMask).mul(this.config.learningDiscount));

        return tf.losses.meanSquaredError(targetQs, qs);
      });

    // Calculate the gradients of the loss function with respect to the weights
    // of the online DQN
    let gradients = tf.variableGrads(lossFunction);

    // Use the gradients to update the online DQN's weights.
    optimizer.applyGradients(gradients.grads);

    tf.dispose(gradients);
  }
}
