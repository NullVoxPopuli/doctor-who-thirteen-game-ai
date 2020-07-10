import { run as runRNN } from './rnn';
import { run as runRandom } from './random';

export const BOT = {
  RNN: 'rnn',
  RANDOM: 'random',
}

export function handleMessage(message) {
  switch (message.type) {
    case 'run':
      return run(message);
    default:
      console.error(message);
      throw new Error('Unrecognized Message');
  }
}

function run({ game, algorithm, trainingData }) {
  switch (algorithm) {
    case 'RNN':
      return runRNN(game, trainingData);
    case 'random':
      return runRandom();
    default:
      console.error(...arguments);
      throw new Error('Unrecognized Algorithm', algorithm);
  }
}
