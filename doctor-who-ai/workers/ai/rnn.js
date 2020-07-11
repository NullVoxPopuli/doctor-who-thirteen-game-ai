/* global RL */
// ^ externally loaded

import { ALL_MOVES } from './consts';
import { clone, groupByValue, loadDependencies, gameTo1DArray } from './utils';
import { imitateMove } from './game';

let areRNNDependenciesLoaded = false;
let rnn;

export async function run(game, trainingData) {
  if (!areRNNDependenciesLoaded) {
    await loadDependencies([
      // RL
      'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/vendor/rl.js',
      // GameManager
      'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/vendor/game.js',
    ]);

    areRNNDependenciesLoaded = true;
  }

  await getMove(game, trainingData);
}

async function getMove(game, trainingData) {
  Object.freeze(game.grid);

  if (!rnn) {
    rnn = createRnn();

    if (trainingData) {
      rnn.fromJSON(trainingData);
    }
  }

  let inputs = gameTo1DArray(game);

  // normalized to 0-getMaxNumAction() - 1
  let moveIndex = await rnn.act(inputs);
  let move = ALL_MOVES[moveIndex];
  let reward = calculateReward(move, game);

  rnn.learn(reward);

  return { move, trainingData: rnn.toJSON() };
}

const calculateReward = (move, originalGame) => {
  let clonedGame = clone(originalGame);
  let moveData = imitateMove(clonedGame, move);

  if (clonedGame.over) {
    if (clonedGame.won) {
      return 1;
    } else {
      return -1;
    }
  }

  if (!moveData.wasMoved) {
    // strongly discourage invalid moves
    return -1;
  }

  let grouped = groupByValue(originalGame);
  let newGrouped = groupByValue(moveData.model);

  let highest = Math.max(...Object.keys(grouped));
  let newHighest = Math.max(...Object.keys(newGrouped));

  // highest two were merged, we have a new highest
  if (newHighest > highest) {
    return 1;
  }

  // for each value, determimne if they've been merged
  // highest first
  let currentValues = Object.keys(newGrouped).sort((a, b) => b - a);

  let likelyWontMakeItTo = 30; // 2 ^ 30 -- need an upper bound for rewarding

  for (let value of currentValues) {
    // what if it previously didn't exist? but still isn't highest?
    if (newGrouped[value] > (grouped[value] || 0)) {
      // log2 converts big number to small number
      // SEE: inverse of VALUE_MAP
      return Math.log2(value) / likelyWontMakeItTo;
    }
  }

  // let bestPossibleMove = outcomesForEachMove(originalGame)[0] || {};
  // let bestPossibleScore = bestPossibleMove.score;

  // if (moveData.score >= bestPossibleScore) {
  //   return 1;
  // }

  if (moveData.score > originalGame.score) {
    return 1 - originalGame.score / moveData.score;

    // Provide a bigger reward the higher the merge value is

    // let additionalPoints = (moveData.score = originalGame.score);

    // let fractionalScore = additionalPoints / Math.pow(2, 13); // highest possible single merge score;

    // return fractionalScore > 1 ? 1 : fractionalScore;
  }

  // next score is equal to current
  // it's possible that we need to do something that doesn't
  // change our score before getting to something good
  return 0; // - originalGame.score / bestPossibleScore;
};

function createRnn() {
  // followed:
  //   https://codepen.io/Samid737/pen/opmvaR
  //   https://github.com/karpathy/reinforcejs

  /*
   *
   * spec.gamma is the discount rate. When it is zero, the agent will be maximally
   *            greedy and won't plan ahead at all. It will grab all the reward it
   *            can get right away. For example, children that fail the marshmallow
   *            experiment have a very low gamma. This parameter goes up to 1, but
   *            cannot be greater than or equal to 1 (this would make the discounted
   *            reward infinite).
   * spec.epsilon controls the epsilon-greedy policy. High epsilon (up to 1) will
   *              cause the agent to take more random actions. It is a good idea to
   *              start with a high epsilon (e.g. 0.2 or even a bit higher) and decay
   *              it over time to be lower (e.g. 0.05).
   * spec.num_hidden_units: currently the DQN agent is hardcoded to use a neural net
   *                        with one hidden layer, the size of which is controlled with
   *                        this parameter. For each problems you may get away with
   *                        smaller networks.
   * spec.alpha controls the learning rate. Everyone sets this by trial and error and
   *            that's pretty much the best thing we have.
   * spec.experience_add_every: REINFORCEjs won't add a new experience to replay every
   *                            single frame to try to conserve resources and get more
   *                            variaty. You can turn this off by setting this parameter
   *                            to 1. Default = 5
   * spec.experience_size: size of memory. More difficult problems may need bigger memory
   * spec.learning_steps_per_iteration: the more the better, but slower. Default = 20
   * spec.tderror_clamp: for robustness, clamp the TD Errror gradient at this value.
   *
   *
   */
  let spec = {
    update: 'qlearn', // qlearn | sarsa algorithm
    gamma: 0.9, // discount factor, [0, 1)
    epsilon: 0.2, // initial epsilon for epsilon-greedy policy, [0, 1)
    alpha: 0.005, // value function learning rate
    experience_add_every: 1, // number of time steps before we add another experience to replay memory
    experience_size: 100000, // size of experience replay memory
    learning_steps_per_iteration: 10,
    tderror_clamp: 1.0, // for robustness
    num_hidden_units: Math.pow(2, 13), // number of neurons in hidden layer
  };

  let env = {
    getNumStates: () => 4,
    getMaxNumActions: () => 4,
  };

  return new RL.DQNAgent(env, spec);
}
