// import tf from '@tensorflow/tfjs';
// import sarsa from 'sarsa';

import { useGPU, getNetwork, getAgent, save } from './tf-utils';
import { ALL_MOVES, MOVE_KEY_MAP } from './consts';

import type { DirectionKey, InternalMove }  from './consts';

import {
  clone,
  groupByValue,
  gameTo1DArray,
  isEqual,
} from './utils';

import { imitateMove, executeMove, fakeGameFrom } from './game';

// let network!: tf.LayersModel;
let network: any;
let agent: any;
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
    agent = await getAgent(network);
  }
}

export async function train100Games(game: Game2048) {
  Object.freeze(game.grid);

  await useGPU();
  await ensureNetwork();

  let games = 0;

  return await new Promise(resolve => {
    let trainOnce = () => trainABit(game);
    let callback = async () => {
      games++;
      await trainOnce();

      if (games < 10000) {
        requestAnimationFrame(callback);
      } else {
        await save(network);

        // let the call-site continue
        resolve();
      }
    }

    window.requestAnimationFrame(callback);
  });
}

async function getMove(game: Game2048): Promise<DirectionKey> {
  let inputs = gameTo1DArray(game);
  let moveIndex = await agent.step(inputs);
  let move = ALL_MOVES[moveIndex];

  return move;
}

async function trainABit(originalGame: Game2048) {
  console.debug('Running simulated game to completion...');
  let moves = 0;
  let start = (new Date()).getDate();
  let clonedGame = clone(originalGame);
  let gameManager = fakeGameFrom(clonedGame);

  while (!gameManager.over) {
    moves++;

    let previousGame = clone(gameManager);
    let move = await getMove(gameManager);

    executeMove(gameManager, move);

    let internalMove = MOVE_KEY_MAP[move];
    let reward = calculateReward(internalMove, previousGame, gameManager);

    agent.reward(reward);
  }

  iterations++;
  console.debug('Simulation Finished', {
    moves,
    numTrainedGames: iterations,
    score: gameManager.score,
    time: (new Date()).getDate() - start,
  });
}

const calculateReward = (move: InternalMove, originalGame: Game2048, currentGame: Game2048) => {
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
