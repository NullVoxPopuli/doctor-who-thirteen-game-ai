import Service, { inject as service } from '@ember/service';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';
import { timeout } from 'ember-concurrency';
import { restartableTask } from 'ember-concurrency-decorators';

export const BOT = {
  RNN: 'rnn',
  RANDOM: 'random',
};

export default class Bot extends Service {
  @service aiWorker;
  @service game;
  @service history;

  @tracked isAutoRetrying = false;
  @tracked currentBot = BOT.RNN;

  @action
  play() {
    this.gameLoop.perform();
  }

  @action
  stop() {
    this.gameLoop.cancelAll();
  }

  @action
  async requestMove() {
    let state = this.game.state;

    if (state !== null && !state.over) {
      if (!this.game.startTime) {
        this.game.startTime = new Date();
      }

      let moveData = await this.aiWorker.requestMove(state, this.currentBot);

      return moveData;
    }
  }

  @restartableTask
  *gameLoop() {
    while (!this.game.isGameOver) {
      let data = yield this.requestMove();

      if (!data.move) {
        console.error(`No move was generated`, data);

        return;
      }

      if (data.trainingData) {
        this.aiWorker.trainingData = data.trainingData;
      }

      this.game.pressKey(data.move);

      // let the external code calculate stuff?
      yield timeout(50);
    }

    this.autoRetry();
  }

  async autoRetry() {
    if (!this.isAutoRetrying) {
      return;
    }

    if (this.game.isGameOver) {
      let stats = this.game.snapshot();

      this.history.add(stats);

      this.game.startNewGame();

      await timeout(1000);

      this.play.perform();
    }
  }
}
