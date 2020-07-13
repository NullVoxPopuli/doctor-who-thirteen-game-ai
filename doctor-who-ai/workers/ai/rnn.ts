import tf from '@tensorflow/tfjs';
import sarsa from 'sarsa';

import { useGPU, getNetwork, save } from './tf-utils';
import { ALL_MOVES, MOVE_KEY_MAP } from './consts';
import {
  clone,
  groupByValue,
  gameTo1DArray,
  isEqual,
} from './utils';
import { imitateMove, executeMove, fakeGameFrom } from './game';

let network!: tf.LayersModel;
let learner!: any;
let iterations = 0;

export async function run(game: Game2048) {
  Object.freeze(game.grid);

  await useGPU();
  await ensureNetwork();

  let move = await getMove(game);

  return move;
}

async function ensureNetwork() {
  if (!network) {
    network = await getNetwork();
    learner = sarsa();
  }
}

export async function train100Games(game: Game2048) {
  Object.freeze(game.grid);

  await useGPU();
  await ensureNetwork();

  for (let i = 0; i < 100; i++) {
    await trainABit(game);
  }

  await save(network);
}

async function getMove(game: Game2048) {
  let inputs = tf.tensor1d(gameTo1DArray(game));

  let result = network.predict(inputs);

  inputs.dispose();

  console.log(result);

  let moveIndex = result.toInt();
  let move = ALL_MOVES[moveIndex];

  return move;
}

async function trainABit(originalGame: Game2048) {
  console.debug('Running simulated game to completion...');
  let moves = 0;
  let start = (new Date()).getDate();
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
    network.train()

    let internalMove = MOVE_KEY_MAP[move];
    let reward = calculateReward(internalMove, previousGame, gameManager);

    sarsa.chooseAction()
    // _reImprove.academy.addRewardToAgent(_reImprove.agent, reward);
  }

  iterations++;
  console.debug('Simulation Finished', {
    moves,
    numTrainedGames: iterations,
    score: gameManager.score,
    time: (new Date()).getDate() - start,
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
