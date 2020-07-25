// import Worker, { method } from '@skyrocketjs/worker';

import * as rnn from './rnn';
import { run as random } from './random';

// export default class AIWorker extends Worker {
//   @method
//   runRNN({ game, trainingData }) {
//     return rnn(game, trainingData);
//   }

//   @method
//   runRandom() {
//     return random();
//   }
// }

// --------------------------------------------------

import { PWBWorker } from 'promise-worker-bi';

let promiseWorker = new PWBWorker();

type RnnMessage =
  | { type: 'rnn'; action: 'train-batch'; data: Game2048 }
  | { type: 'rnn'; action: 'get-move'; data: Game2048 }
  | { type: 'rnn'; action: 'save'; data: never };

type RandomMessage = { type: 'random'; action: 'get-move'; data: never };

type Message = RnnMessage | RandomMessage;

promiseWorker.register(({ type, action, data }: Message) => {
  switch (type) {
    case 'rnn':
      return handleRnn(action, data);
    case 'random':
      return handleRandom(action);
    default:
      throw new Error(`Unknown message: ${type}`);
  }
});

function handleRnn(action: RnnMessage['action'], data: Game2048) {
  switch (action) {
    case 'get-move':
      return rnn.getMove(data);
    case 'save':
      // return rnn.save();
      throw new Error('Not implemented');
    case 'train-batch':
      return rnn.trainBatch(data);
  }
}

function handleRandom(_action: RandomMessage['action']) {
  return random();
}
