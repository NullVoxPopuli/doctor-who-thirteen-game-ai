import Component from '@glimmer/component';
import { inject as service } from '@ember/service';
import { action } from '@ember/object';


import { OPTIONS } from 'doctor-who-ai/services/bot';

export default class Controls extends Component {
  @service bot;

  botOptions = OPTIONS;

  @action
  start() {
    this.bot.play();
  }

  @action
  stop() {
    this.bot.stop();
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
