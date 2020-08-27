import * as tf from '@tensorflow/tfjs';

import { useGPU, getNetwork, createNetwork, save, copyWeights } from './model/tf-utils';
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

  // Loads the saved network as the "Target Network"
  await ensureNetwork();

  let games = 0;
  let batches = 300;
  let gamesPerBatch = 10;
  let total = batches * gamesPerBatch;

  let bestRewardForBatch = 0;
  let batchesSinceLastSave = 0;

  for (let i = 0; i < batches; i++) {
    console.time(`Batch ${i}`);
    let stats = await agent.train(gamesPerBatch);

    console.timeEnd(`Batch ${i}`);
    batchesSinceLastSave++;

    agent.qlearn.decayEpsilon(i);

    games += gamesPerBatch;
    totalGames += gamesPerBatch;

    console.debug(
      `Average Reward of Batch: ${stats.averageReward} vs best average reward so far: ${bestRewardForBatch}`
    );

    // Saves the target network
    if (stats.averageReward > bestRewardForBatch) {
      bestRewardForBatch = stats.averageReward;
      console.debug(`Average improved, saving network`);

      await save(agent.qlearn.onlineNetwork);
      batchesSinceLastSave = 0;
    }

    if (batchesSinceLastSave > 10) {
      console.debug(`It's been a while -- saving network`);

      await save(agent.qlearn.onlineNetwork);
      batchesSinceLastSave = 0;
    }

    copyWeights(agent.qlearn.targetNetwork, agent.qlearn.onlineNetwork);
  }

  console.timeEnd('Training');
}
