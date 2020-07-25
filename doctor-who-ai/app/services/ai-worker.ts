import Service, { inject as service } from '@ember/service';
import { action } from '@ember/object';
import { tracked } from '@glimmer/tracking';

import env from 'doctor-who-ai/config/environment';

import { PWBHost } from 'promise-worker-bi';

// import { worker } from '@skyrocketjs/ember';

// import { BOT } from './bot';

import Game from './game';

export type Algorithm = 'rnn' | 'random';

export default class AIWorker extends Service {
  @service game!: Game;

  worker?: PWBHost;
  // @worker('ai') worker;

  @tracked isReady = false;

  @action
  async train(seedGame: Game2048) {
    if (!this.worker) {
      this.worker = await createWorker();
    }

    return await this.worker.postMessage({ type: 'rnn', action: 'train-batch', data: seedGame });
  }

  @action
  async requestMove(state: Game2048, algorithm: Algorithm) {
    if (!this.worker) {
      this.worker = await createWorker();
    }

    let options = { type: algorithm, action: 'get-move', data: state };

    let { move } = await this.worker.postMessage(options);

    return { move };
  }

  willDestroy() {
    if (this.worker) {
      this.worker._worker.terminate();
    }
  }
}

const workerUrl = env.isDevelopment
  ? '/workers/ai.js'
  : 'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/worker.js';

// fetching the URL instead of directly loading in a script
// tag allows us to get around CORS issues
async function createWorker() {
  let response = await fetch(workerUrl);
  let script = await response.text();
  let blob = new Blob([script], { type: 'text/javascript' });

  let worker = new Worker(URL.createObjectURL(blob));
  let promiseWorker = new PWBHost(worker);

  return promiseWorker;
}
