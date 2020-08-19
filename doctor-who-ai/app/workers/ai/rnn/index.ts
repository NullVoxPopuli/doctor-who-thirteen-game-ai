import * as tf from '@tensorflow/tfjs';

import { useGPU, getNetwork, createNetwork, save } from './model/tf-utils';
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
    agent = new GameTrainer({
      network,
      createNetwork,
    });
  }
}

export async function trainBatch() {
  console.time('Training');

  await useGPU();
  await ensureNetwork();

  let games = 0;
  let batches = 300;
  let gamesPerBatch = 10;
  let total = batches * gamesPerBatch;

  for (let i = 0; i < batches; i++) {
    console.time(`Batch ${i}`);
    let stats = await agent.train(undefined, gamesPerBatch);

    console.timeEnd(`Batch ${i}`);

    agent.qlearn.decayEpsilon(i);

    games += gamesPerBatch;
    totalGames += gamesPerBatch;

    console.debug(`${total - games} left until displayed game. Total: ${totalGames}`);
    console.table([stats]);

    await save(network);
  }

  console.timeEnd('Training');
}
