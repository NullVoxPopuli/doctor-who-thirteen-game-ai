import tf from '@tensorflow/tfjs';
// import sarsa from 'sarsa';

import { useGPU, getNetwork, getAgent, save } from './tf-utils';
import { ALL_MOVES, MOVE_KEY_MAP } from './consts';

import type { DirectionKey, InternalMove } from './consts';

import { clone, groupByValue, gameTo1DArray, isEqual } from './utils';

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
  console.time('Training');
  Object.freeze(game.grid);

  await useGPU();
  await ensureNetwork();

  let games = 0;
  let batches = 5;
  let gamesPerBatch = 40;
  let total = batches * gamesPerBatch;
  // work has to be batched, cause the browser tab
  // keeps crashing
  // can this be moved to a web worker?
  let trainOnce = () => trainABit(game);

  let trainBatch = async () => {
    for (let i = 0; i < gamesPerBatch; i++) {
      games++;
      let trainingResult = await trainOnce();

      console.debug(`${total - games} left until displayed game. Last: `, trainingResult);
    }
  };

  return new Promise((resolve) => {
    let callback = async () => {
      await trainBatch();

      if (games < total) {
        setTimeout(() => {
          requestIdleCallback(callback);
          // 1s break to trick the browser in to thnking
          // the page is responsive
        }, 1000);
      } else {
        // await save(network);

        // let the call-site continue
        console.timeEnd('Training');
        resolve();
      }
    };

    requestIdleCallback(callback);
  });
}

async function getMove(game: Game2048): Promise<DirectionKey> {
  let inputs = gameTo1DArray(game);
  let moveIndex = await agent.step(inputs);
  let move = ALL_MOVES[moveIndex];

  return move;
}

async function trainABit(originalGame: Game2048) {
  let moves = 0;
  // let start = new Date().getDate();
  let clonedGame = clone(originalGame);
  let gameManager = fakeGameFrom(clonedGame);

  // let totalReward = 0;
  let totalNonMoves = 0;

  while (!gameManager.over) {
    moves++;

    let inputs = gameTo1DArray(gameManager);
    let moveIndex = await agent.step(inputs);
    let move = ALL_MOVES[moveIndex];

    let { reward, wasMoved } = moveAndCalculateReward(move, gameManager);

    executeMove(gameManager, move);

    tf.tidy(() => {
      agent.reward(reward);
    });

    if (!wasMoved) {
      totalNonMoves += 1;
    }

    // totalReward += reward;
  }

  iterations++;

  return {
    totalGames: iterations,
    score: gameManager.score,
    moves,
    totalNonMoves,
    percentValidMoves: Math.round(((moves - totalNonMoves) / moves) * 100),
  };
}

const moveAndCalculateReward = (move: DirectionKey, currentGame: Game2048) => {
  let moveData;
  let previousGame = clone(currentGame);

  executeMove(currentGame, move);

  moveData = {
    currentScore: currentGame.score,
    previousScore: previousGame.score,
    scoreChange: currentGame.score - previousGame.score,
    wasMoved: false,
  };
  moveData.wasMoved = moveData.scoreChange !== 0;

  if (!moveData.wasMoved) {
    return { reward: -0.05, ...moveData };
  }

  let grouped = groupByValue(previousGame);
  let newGrouped = groupByValue(currentGame);

  let highest = Math.max(...Object.keys(grouped));
  let newHighest = Math.max(...Object.keys(newGrouped));

  // highest two were merged, we have a new highest
  if (newHighest > highest) {
    return { reward: 1, ...moveData };
  }

  if (currentGame.score > previousGame.score) {
    return { reward: 0.5, ...moveData };
  }

  // next score is equal to current
  // it's possible that we need to do something that doesn't
  // change our score before getting to something good
  return { reward: 0, ...moveData };
};
