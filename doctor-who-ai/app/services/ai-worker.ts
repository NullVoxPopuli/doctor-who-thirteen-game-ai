import Service, { inject as service } from '@ember/service';
import { action } from '@ember/object';
import { tracked } from '@glimmer/tracking';

// import { worker } from '@skyrocketjs/ember';

// import { BOT } from './bot';

import { run as getMove, train100Games } from './ai/rnn';

export default class AIWorker extends Service {
  @service game;
  // @worker('ai') worker;

  @tracked isReady = false;

  @action
  async train(seedGame: Game2048) {
    await train100Games(seedGame);
  }

  @action
  async requestMove(state: Game2048, algorithm) {
    // if (!this.worker) {
    //   console.debug('Worker not loaded yet');

    //   return;
    // }

    // let options = { type: 'run', game: state, algorithm };

    // if (algorithm === BOT.RNN) {
    //   options.trainingData = this.trainingData;
    // }

    // return await this.worker.postMessage(options);
    const move = await getMove(state);

    return { move };
  }

  @action
  onMessage(e) {
    // let { data } = e;
    // switch (data.type) {
    //   case 'ack':
    //   case 'ready':
    //     return (this.isReady = true);
    //   default:
    //     console.error(data);
    //     throw new Error('Unrecognized Message');
    // }
  }

  willDestroy() {
    // this.worker._worker.terminate();
  }
}

// const workerUrl =
//   'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/worker.js';

// // fetching the URL instead of directly loading in a script
// // tag allows us to get around CORS issues
// async function createWorker() {
//   let response = await fetch(workerUrl);
//   let script = await response.text();
//   let blob = new Blob([script], { type: 'text/javascript' });

//   let worker = new Worker(URL.createObjectURL(blob));
//   let promiseWorker = new PWBHost(worker);

//   return promiseWorker;
// }
