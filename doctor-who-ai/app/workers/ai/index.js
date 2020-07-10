import { PWBWorker } from 'promise-worker-bi';
import { handleMessage } from './messages';

let promiseWorker = new PWBWorker();

promiseWorker.register(function (message) {
  return handleMessage(message);
});
