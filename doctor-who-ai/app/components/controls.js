import Component from '@glimmer/component';
import { inject as service } from '@ember/service';
import { action } from '@ember/object';

export default class Controls extends Component {
  @service bot;

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
    console.log(e);
  }

  @action
  toggleAutoRetry(e) {
    let value = e.target.checked;

    this.bot.isAutoRetrying = value;
  }
}
