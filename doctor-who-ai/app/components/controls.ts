import Component from '@glimmer/component';
import { inject as service } from '@ember/service';
import { action } from '@ember/object';

import { OPTIONS } from 'doctor-who-ai/services/bot';

import type Bot from 'doctor-who-ai/services/bot';

export default class Controls extends Component {
  @service bot!: Bot;

  botOptions = OPTIONS;

  @action
  toggleTrain() {
    this.bot.toggleTraining();
  }

  @action
  togglePlay() {
    this.bot.togglePlay();
  }

  @action
  selectAlgorithm(e) {
    let value = e.target.value;

    this.bot.currentBot = value;
  }

  @action
  toggleAutoRetry(e) {
    let value = e.target.checked;

    this.bot.isAutoRetrying = value;
  }
}
