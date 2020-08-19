import * as tf from '@tensorflow/tfjs';

import { Memory } from './memory';

import { decayEpsilon } from './utils';

import type { Config } from './types';
import { TypedArray } from '@tensorflow/tfjs';
import { highestValue } from '../../game';

interface RequiredRewardInfo {
  nextState: tf.Tensor;
  reward: number;
}

type Maybe<T> = T | undefined;
type ActionMemory = [tf.Tensor, number, number, Maybe<tf.Tensor>, boolean];

type LessonConfig<Game, State, Action, RewardInfo extends RequiredRewardInfo> = {
  game: Game;
  getState: (game: Game) => State;
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
  afterGame: (gameInfo: { game: Game }) => Promise<void>;
};

type LearningConfig = {
  reshapeState: (state: tf.Tensor) => tf.Tensor;
  gamma: number;
  batchSize: number;
  optimizer: tf.Optimizer;
};

interface NetworkArgs {
  network: tf.LayersModel;
  createNetwork: () => tf.LayersModel;
}

export class QLearn {
  declare actionMemory: Memory<ActionMemory>;
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
    this.actionMemory = new Memory<ActionMemory>(this.config.moveMemorySize);

    this.targetNetwork = network;
    this.onlineNetwork = createNetwork();
  }

  decayEpsilon(iterations: number) {
    this.config.epsilon = decayEpsilon(this.config, iterations);
  }

  async playOnce<
    Game,
    State extends tf.Tensor,
    Action extends number,
    RewardInfo extends RequiredRewardInfo
  >(lessonConfig: LessonConfig<Game, State, Action, RewardInfo>) {
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
      let inputs = getState(game);

      let useExternalAction = Math.random() < this.config.epsilon;
      let rankedActions = getRankedActions(game, inputs, useExternalAction, this.onlineNetwork);

      let action: Action | undefined;

      // explore the move space a little more than just playind would allow
      for (let possibleAction of rankedActions) {
        let rewardInfo = getReward(game, possibleAction);
        let isDone = isGameOver(game);
        let nextState = isDone ? undefined : rewardInfo.nextState;

        this.actionMemory.add([inputs, possibleAction, rewardInfo.reward, nextState, isDone]);

        if (!action && isValidAction(rewardInfo)) {
          action = possibleAction;
          totalReward += rewardInfo.reward;
          break;
        } else if (!action) {
          numInvalidSteps++;
        }
      }

      takeAction(game, action || rankedActions[0]);

      numSteps++;
      this.playCount++;
    }

    await lessonConfig.afterGame({ game });

    return {
      numSteps,
      numInvalidSteps,
    };
  }

  async learn<Prediction extends tf.Tensor<tf.Rank>[] | tf.Tensor<tf.Rank>>(
    learningConfig: LearningConfig<Prediction>
  ) {
    let { reshapeState, gamma, batchSize, optimizer } = learningConfig;

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
        let nextStates = getStateTensor(
          batch.map(([, , , nextState]) => nextState || tf.zeros([this.config.numInputs]))
        );

        let nextMaxQ = this.targetNetwork.predict(nextStates).max(-1);

        let doneMask = tf
          .scalar(1)
          .sub(tf.tensor1d(batch.map(([, , , , done]) => done)).asType('float32'));

        let targetQs = rewards.add(nextMaxQ.mul(doneMask).mul(gamma));

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

function getStateTensor(state: tf.Tensor | tf.Tensor[]) {
  if (!Array.isArray(state)) {
    state = [state];
  }

  const numExamples = state.length;

  // TODO(cais): Maintain only a single buffer for efficiency.
  const buffer = tf.buffer([numExamples, 4, 4, 1]);

  for (let n = 0; n < numExamples; ++n) {
    if (state[n] == null) {
      continue;
    }

    buffer.set(n, ...state[n].dataSync());
  }

  return buffer.toTensor();
}
