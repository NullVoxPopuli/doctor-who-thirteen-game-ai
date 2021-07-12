import Service, { inject as service } from '@ember/service';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';

import { timeout } from 'ember-concurrency';
import { restartableTask } from 'ember-concurrency-decorators';
import { taskFor } from 'ember-concurrency-ts';

import type AIWorker from './ai-worker';
import type { Algorithm } from './ai-worker';
import type Game from './game';
import type GameHistory from './history';
import { printGame } from 'doctor-who-ai/utils';

export const BOT = {
  RNN: 'rnn',
  RANDOM: 'random',
  ASTAR: 'a-star',
} as const;

export const OPTIONS = {
  [BOT.RNN]: 'Reinforcement Learning Neural Network',
  [BOT.RANDOM]: 'Random',
  [BOT.ASTAR]: 'A*',
} as const;

export default class Bot extends Service {
  @service aiWorker!: AIWorker;
  // @service worker;
  @service game!: Game;
  @service history!: GameHistory;

  @tracked isAutoRetrying = false;
  @tracked currentBot: Algorithm = BOT.RNN;

  @action
  togglePlay() {
    let task = taskFor(this.gameLoop);

    if (task.isRunning) {
      task.cancelAll();
    } else {
      task.perform();
    }
  }

  @action
  toggleTraining() {
    let task = taskFor(this.trainTask);

    if (task.isRunning) {
      task.cancelAll();
    } else {
      task.perform();
    }
  }

  @action
  async requestMove() {
    let state = this.game.state;

    // if (state) {
    //   printGame(state);
    // }

    if (state !== null && !state.over) {
      if (!this.game.startTime) {
        this.game.startTime = new Date().getDate();
      }

      let { move, _rewardInfo } = await this.aiWorker.requestMove(state, this.currentBot);
      // let { reward, distanceNew, distanceOld } = rewardInfo;

      // console.log('^ from, to v', { reward, distanceNew, distanceOld });

      return { move };
    }
  }

  @restartableTask
  *trainTask() {
    yield this.aiWorker.train(this.game.state);
  }

  @restartableTask
  *gameLoop() {
    console.info('Starting Demonstration...');

    while (!this.game.isGameOver) {
      // let the external code calculate stuff?
      yield timeout(250);

      let data = yield this.requestMove();

      if (!data) {
        continue;
      }

      if (!data.move) {
        console.error(`No move was generated`, data);

        return;
      }


      this.game.pressKey(data.move);
    }

    yield timeout(5000);

    this.autoRetry();
  }

  @action
  async autoRetry() {
    if (!this.isAutoRetrying) {
      return;
    }

    if (this.game.isGameOver) {
      let stats = this.game.snapshot();

      this.history.addGame(stats);

      this.game.startNewGame();

      await timeout(1000);

      this.gameLoop.perform();
    }
  }
}

declare module '@ember/service' {
  interface Registry {
    bot: Bot;
  }
}
