import ReImprove from 'reimprovejs';
import tf from '@tensorflow/tfjs';

import { ALL_MOVES, MOVE_KEY_MAP } from './consts';
import {
  clone,
  groupByValue,
  loadDependencies,
  gameTo1DArray,
  isEqual,
} from './utils';
import { imitateMove, executeMove, fakeGameFrom } from './game';

let _reImprove = {};
let iterations = 0;

const dataLocation = 'downloads://re-improve.model';

export async function runReImprove(game, trainingData) {
  Object.freeze(game.grid);

  if (tf.getBackend() !== 'webgl') {
    await tf.setBackend('webgl');
  }

  if (!_reImprove.agent) {
    Object.assign(_reImprove, await createNetwork());
  }

  let move = await getMove(game);
  let reward = calculateReward(move, game);

  _reImprove.academy.addRewardToAgent(_reImprove.agent, reward);

  return move;
}

export async function train100Games(game) {
  Object.freeze(game.grid);

  if (tf.getBackend() !== 'webgl') {
    await tf.setBackend('webgl');
  }

  if (!_reImprove.agent) {
    Object.assign(_reImprove, await createNetwork());
  }

  for (let i = 0; i < 100; i++) {
    await trainABit(game);
  }

  await _reImprove.model.model.save(dataLocation);
}

const modelFitConfig = {
  // Exactly the same idea here by using tfjs's model's
  epochs: 1, // fit config.
  stepsPerEpoch: 16,
};

const numActions = 3; // (including 0?)                 // The number of actions your agent can choose to do
// const inputSize = 16; // Inputs size (10x10 image for instance)
const temporalWindow = 1; // The window of data which will be sent yo your agent
// For instance the x previous inputs, and what actions the agent took

// const totalInputSize =
//   inputSize * temporalWindow + numActions * temporalWindow + inputSize;

function createNewModel() {
  // const network = new ReImprove.NeuralNetwork();

  // network.InputShape = [totalInputSize];

  let tfModel = tf.sequential();

  let hiddenLayers = [
    Math.pow(2, 8),
    Math.pow(2, 11),
    Math.pow(2, 10),
    Math.pow(2, 9),
    Math.pow(2, 8),
    Math.pow(2, 6),
    Math.pow(2, 5),
  ];

  for (let i = 0; i < hiddenLayers.length; i++) {
    let hiddenLayer = { name: 'hidden', units: hiddenLayers[i], activation: 'relu' };

    if (i === 0) {
      hiddenLayer.inputShape = [16];
    }

    tfModel.add(tf.layers.dense(hiddenLayer));
  }

  tfModel.add(tf.layers.dense({ name: 'output', units: numActions, activation: 'softmax' }));

  tfModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  let model = new ReImprove.Model();

  model.model = tfModel;

  return model;
}

async function getTfModel() {
  try {
    return await tf.loadLayersModel('/re-improve.model');
  } catch (e) {
    console.error(e);

    return createNewModel();
  }
}

async function createNetwork() {
  let model = new ReImprove.Model();

  model.model = await getTfModel();
  model.FitConfig = modelFitConfig;

  // Every single field here is optionnal, and has a default value. Be careful, it may not
  // fit your needs ...

  const teacherConfig = {
    lessonsQuantity: 10000,
    lessonLength: 20,
    lessonsWithRandom: 20, // We do not care about full random sessions
    epsilon: 0.5, // Maybe a higher random rate at the beginning ?
    epsilonDecay: 0.995,
    epsilonMin: 0.05,
    gamma: 0.9,
  };

  const agentConfig = {
    model: model, // Our model corresponding to the agent
    agentConfig: {
      memorySize: 1000, // The size of the agent's memory (Q-Learning)
      batchSize: 16, // How many tensors will be given to the network when fit
      temporalWindow: temporalWindow, // The temporal window giving previous inputs & actions
    },
  };

  const academy = new ReImprove.Academy(); // First we need an academy to host everything
  const teacher = academy.addTeacher(teacherConfig);
  const agent = academy.addAgent(agentConfig);

  academy.assignTeacherToAgent(agent, teacher);

  return { model, academy, agent, teacher };
}

async function getMove(game) {
  let inputs = gameTo1DArray(game);

  let result = await _reImprove.academy.step([
    // Let the magic operate ...
    { teacherName: _reImprove.teacher, agentsInput: inputs },
  ]);

  let moveIndex = result.get(_reImprove.agent);
  let move = ALL_MOVES[moveIndex];

  return move;
}

async function trainABit(originalGame) {
  console.debug('Running simulated game to completion...');
  let moves = 0;
  let start = new Date();
  // copy the game
  // run to completion
  let clonedGame = clone(originalGame);
  let gameManager = fakeGameFrom(clonedGame);

  while (!gameManager.over) {
    moves++;

    // if (moves % 100 === 0) {
    //   console.debug(`at ${moves} moves...`);
    // }

    let previousGame = clone(gameManager);
    let move = await getMove(gameManager);

    executeMove(gameManager, move);

    let internalMove = MOVE_KEY_MAP[move];
    let reward = calculateReward(internalMove, previousGame, gameManager);

    _reImprove.academy.addRewardToAgent(_reImprove.agent, reward);
  }

  iterations++;
  console.debug('Simulation Finished', {
    moves,
    numTrainedGames: iterations,
    score: gameManager.score,
    time: new Date() - start,
  });
}

const calculateReward = (move, originalGame, currentGame) => {
  let moveData;
  let clonedGame;

  if (!currentGame) {
    clonedGame = clone(originalGame);
    moveData = imitateMove(clonedGame, move);
  } else {
    clonedGame = currentGame;
    moveData = {
      model: currentGame,
      score: currentGame.score,
      wasMoved: !isEqual(
        currentGame.serialize().grid.cells,
        originalGame.grid.cells
      ),
    };
  }

  // if (clonedGame.over) {
  //   if (clonedGame.won) {
  //     return 1;
  //   } else {
  //     return -1;
  //   }
  // }

  // if (!moveData.wasMoved) {
  //   // strongly discourage invalid moves
  //   return -1;
  // }

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
  // let currentValues = Object.keys(newGrouped).sort((a, b) => b - a);

  // let likelyWontMakeItTo = 15; // 2 ^ 30 -- need an upper bound for rewarding

  // for (let value of currentValues) {
  //   // what if it previously didn't exist? but still isn't highest?
  //   if (newGrouped[value] > (grouped[value] || 0)) {
  //     // log2 converts big number to small number
  //     // SEE: inverse of VALUE_MAP
  //     return Math.log2(value) / likelyWontMakeItTo;
  //   }
  // }

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
