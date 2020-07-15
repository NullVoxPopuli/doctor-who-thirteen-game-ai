import Component from '@glimmer/component';
import { inject as service } from '@ember/service';

import type GameHistory from 'doctor-who-ai/services/history';
import type Game from 'doctor-who-ai/services/game';

export default class Stats extends Component {
  @service history!: GameHistory;
  @service game!: Game;

  get data() {
    return this.history.latest;
  }

  get topDoctor() {
    return this.game.topDoctor;
  }
}
