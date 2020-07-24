import tf from '@tensorflow/tfjs';

import { useGPU, getNetwork, save } from './tf-utils';

import { Agent } from './learning/agent';
import { GameTrainer } from './game-trainer';
import { Orchestrator } from './tfjs-mountaincar/orchestrator';

let network!: tf.LayersModel;
let agent: GameTrainer;
let highestScore = 0;
let totalGames = 0;

export async function run(game: Game2048) {
  Object.freeze(game.grid);

  await useGPU();
  await ensureNetwork();

  let move = await agent.getMove(game);

  return move;
}

async function ensureNetwork() {
  if (!network) {
    network = await getNetwork();
    agent = new GameTrainer(network, {
      epsilon: 0.05,
      numActions: 4,
      gameMemorySize: 100,
      moveMemorySize: 10000,
    });
  }
}

export async function train100Games(game: Game2048) {
  console.time('Training');
  Object.freeze(game.grid);

  await useGPU();
  await ensureNetwork();

  let totalScore = 0;
  let games = 0;
  let batches = 50;
  let gamesPerBatch = 10;
  let total = batches * gamesPerBatch;
  // work has to be batched, cause the browser tab
  // keeps crashing
  // can this be moved to a web worker?
  let trainOnce = () => agent.orchestrator.run(game, games);

  let trainBatch = async () => {
    for (let i = 0; i < gamesPerBatch; i++) {
      games++;
      totalGames++;
      let trainingResult = await trainOnce();

      totalScore += trainingResult.score;
      highestScore = Math.max(highestScore, trainingResult.score);

      console.debug(`${total - games} left until displayed game.`);
      console.table([
        {
          totalGames,
          highestScore,
          averageScore: Math.round((totalScore / games) * 100) / 100,
          ...trainingResult,
        },
      ]);
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
        await save(network);

        // let the call-site continue
        console.timeEnd('Training');
        resolve();
      }
    };

    requestIdleCallback(callback);
  });
}
