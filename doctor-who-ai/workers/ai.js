import Worker, { method } from '@skyrocketjs/worker';

import { run as rnn } from './ai/rnn';
import { run as random } from './ai/random';

export default class AIWorker extends Worker {
  @method
  runRNN({ game, trainingData }) {
    return rnn(game, trainingData);
  }

  @method
  runRandom() {
    return random();
  }
}
