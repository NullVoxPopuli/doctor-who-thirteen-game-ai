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
type ActionMemory = [tf.Tensor, number, number, Maybe<tf.Tensor>];

interface GameMemory {
  score: number;
  moves: number;
  highestValue: number;
  totalReward: number;
  moveMemory: Memory<ActionMemory>;
}

type LessonConfig<Game, State, Action, RewardInfo extends RequiredRewardInfo> = {
  game: Game;
  getState: (game: Game) => State;
  isGameOver: (game: Game) => boolean;
  getRankedActions: (game: Game, state: State, useExternalAction: boolean) => Action[];
  takeAction: (game: Game, action: Action) => void;
  getReward: (game: Game, action: Action) => RewardInfo;
  isValidAction: (rewardInfo: RewardInfo) => boolean;
  afterGame: (gameInfo: {
    game: Game;
    totalReward: number;
    moveMemory: Memory<ActionMemory>;
  }) => Promise<void>;
};

type LearningConfig<Prediction> = {
  reshapeState: (state: tf.Tensor) => tf.Tensor;
  predict: (input: tf.Tensor) => Prediction;
  fit: (inputs: tf.Tensor, outputs: tf.Tensor) => Promise<tf.History>;
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
    this.gameMemory = new Memory<GameMemory>(
      config.gameMemorySize,
      0.025,
      (via) => via.totalReward
    );
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

    let moveMemory = new Memory<ActionMemory>(this.config.moveMemorySize);

    let totalReward = 0;
    let numSteps = 0;
    let numInvalidSteps = 0;

    while (!isGameOver(game)) {
      let inputs = getState(game);

      let useExternalAction = Math.random() < this.config.epsilon;
      let rankedActions = getRankedActions(game, inputs, useExternalAction);

      let action: Action | undefined;

      // explore the move space a little more than just playind would allow
      for (let possibleAction of rankedActions) {
        let rewardInfo = getReward(game, possibleAction);
        let nextState = isGameOver(game) ? undefined : rewardInfo.nextState;

        moveMemory.add([inputs, possibleAction, rewardInfo.reward, nextState]);

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

    await lessonConfig.afterGame({ game, totalReward, moveMemory });
    // this.gameMemory.add({
    //   totalReward,
    //   moveMemory,
    //   score: game.score,
    //   moves: numSteps,
    //   highestValue: highestValue(game),
    // });

    return {
      numSteps,
      numInvalidSteps,
    };
  }

  async learn<Prediction extends tf.Tensor<tf.Rank>[] | tf.Tensor<tf.Rank>>(
    learningConfig: LearningConfig<Prediction>
  ) {
    // let games: GameMemory[] = this.gameMemory.recallRandomly(300);
    let games = this.gameMemory
      .recallTopBy((item) => item.score, 0.5)
      .sort((a, z) => z.score - a.score);

    let bestReward = Math.max(...games.filter((game) => game.score));

    for (let game of games) {
      let weight = game.score / bestReward;

      await this.learnFromGame(game, learningConfig, weight);
    }
  }

  async learnFromGame<Prediction extends tf.Tensor<tf.Rank>[] | tf.Tensor<tf.Rank>>(
    game: GameMemory,
    learningConfig: LearningConfig<Prediction>,
    gameWeight?: number
  ) {
    let { reshapeState, predict, fit } = learningConfig;

    let { learningRate, learningDiscount } = this.config;

    // Sample from memory
    const batch = game.moveMemory.recallRandomly(200);

    let x: TypedArray[] = [];
    let y: number[] = [];

    // Update the states rewards with the discounted next states rewards
    batch.forEach(([state, action, reward, nextState], index) => {
      let predictionForState = predict(reshapeState(state));
      let predictionForNextState = predict(
        reshapeState(nextState || tf.zeros([this.config.numInputs]))
      );

      let inputs = state.dataSync();
      let qsa = predictionForState.dataSync();
      let nextQsaMax = predictionForNextState.max();

      if (nextState) {
        // Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        qsa[action] =
          qsa[action] + learningRate * (reward + learningDiscount * nextQsaMax - qsa[action]);
      } else {
        qsa[action] = reward;
      }

      if (gameWeight) {
        qsa[action] *= gameWeight;
      }

      for (let i = 0; i < this.config.numActions; i++) {
        if (i === action) {
          continue;
        }

        qsa[i] = 0;
      }

      x.push(inputs);
      y.push(qsa);

      predictionForState.dispose();
      predictionForNextState.dispose();
    });

    // Reshape the batches to be fed to the network
    let inputs = tf.tensor(x, [x.length, 4, 4, 1]);
    let outputs = tf.tensor(y, [y.length, this.config.numActions]);

    // Learn the Q(s, a) values given associated discounted rewards
    await fit(inputs, outputs);

    inputs.dispose();
    outputs.dispose();
  }
}
