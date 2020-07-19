import tf from '@tensorflow/tfjs';

import { useGPU, getNetwork, save } from './tf-utils';

import { Agent } from './learning/agent';
import { GameTrainer } from './game-trainer';
import { Orchestrator } from './tfjs-mountaincar/orchestrator';

let network!: tf.LayersModel;
let agent: GameTrainer;

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
      moveMemorySize: 8000,
    });
  }
}

export async function train100Games(game: Game2048) {
  console.time('Training');
  Object.freeze(game.grid);

  await useGPU();
  await ensureNetwork();

  let games = 0;
  let batches = 10;
  let gamesPerBatch = 20;
  let total = batches * gamesPerBatch;
  // work has to be batched, cause the browser tab
  // keeps crashing
  // can this be moved to a web worker?
  let trainOnce = () => agent.orchestrator.run(game);

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
