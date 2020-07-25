import * as tf from '@tensorflow/tfjs';

import { useGPU, getNetwork, save } from './model/tf-utils';
import { GameTrainer } from './game-trainer';

let network!: tf.LayersModel;
let agent: GameTrainer;
let totalGames = 0;

export async function getMove(game: GameState) {
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
      numInputs: 16,
      gameMemorySize: 100,
      moveMemorySize: 10000,
    });
  }
}

export async function trainBatch(game: GameState) {
  console.time('Training');
  Object.freeze(game.grid);

  await useGPU();
  await ensureNetwork();

  let games = 0;
  let batches = 20;
  let gamesPerBatch = 10;
  let total = batches * gamesPerBatch;

  for (let i = 0; i < batches; i++) {
    console.debug(`Starting Batch ${i}`);
    let stats = await agent.train(game, gamesPerBatch);

    games += gamesPerBatch;
    totalGames += gamesPerBatch;

    console.debug(`${total - games} left until displayed game. Total: ${totalGames}`);
    console.table([stats]);
  }

  await save(network);

  console.timeEnd('Training');
}
