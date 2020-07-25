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

export const BOT = {
  RNN: 'rnn',
  RANDOM: 'random',
} as const;

export const OPTIONS = {
  [BOT.RNN]: 'Reinforcement Learning Neural Network',
  [BOT.RANDOM]: 'Random',
} as const;

export default class Bot extends Service {
  @service aiWorker!: AIWorker;
  // @service worker;
  @service game!: Game;
  @service history!: GameHistory;

  @tracked isAutoRetrying = false;
  @tracked currentBot: Algorithm = BOT.RNN;

  @action
  play() {
    taskFor(this.gameLoop).perform();
  }

  @action
  stop() {
    taskFor(this.gameLoop).cancelAll();
  }

  @action
  async requestMove() {
    let state = this.game.state;

    if (state !== null && !state.over) {
      if (!this.game.startTime) {
        this.game.startTime = new Date().getDate();
      }

      let moveData = await this.aiWorker.requestMove(state, this.currentBot);

      return moveData;
    }
  }

  @restartableTask
  *gameLoop() {
    yield this.aiWorker.train(this.game.state);

    // console.info('Starting Demonstration...');

    // while (!this.game.isGameOver) {
    //   // let the external code calculate stuff?
    //   yield timeout(250);

    //   let data = yield this.requestMove();

    //   if (!data) {
    //     continue;
    //   }

    //   if (!data.move) {
    //     console.error(`No move was generated`, data);

    //     return;
    //   }

    //   if (data.trainingData) {
    //     this.aiWorker.trainingData = data.trainingData;
    //   }

    //   this.game.pressKey(data.move);
    // }

    // this.autoRetry();
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

      this.play();
    }
  }
}

declare module '@ember/service' {
  interface Registry {
    bot: Bot;
  }
}
