import Service from '@ember/service';
import { tracked } from '@glimmer/tracking';
import { action } from '@ember/object';

import { DOCTOR_NUMBER_MAP, biggestTile } from 'doctor-who-ai/utils';
import type { DirectionKey } from './ai/consts';

export default class Game extends Service {
  @tracked startTime?: number;
  @tracked topDoctor = '';

  get isGameOver() {
    return Boolean(document.querySelector('.game-over'));
  }

  get score() {
    return parseInt(document.querySelector('.score-container').textContent, 10);
  }

  get state() {
    return JSON.parse(localStorage.getItem('gameState')) as GameState;
  }

  get duration() {
    if (!this.startTime) return 0;

    return new Date().getDate() - this.startTime;
  }

  @action
  snapshot() {
    return {
      score: this.score,
      time: this.duration,
    };
  }

  startNewGame() {
    // set by moving (in case there is delay)
    this.startTime = undefined;

    document.querySelector('.retry-button').click();
  }

  pressKey(key: DirectionKey) {
    simulateKeyPress(key);

    // state will be empty at game's end
    if (this.state) {
      this.topDoctor = topDoctorFor(this.state);
    }
  }
}

function topDoctorFor(game: Game2048) {
  const biggest = biggestTile(game);

  return DOCTOR_NUMBER_MAP[biggest.num];
}

function simulateKeyPress(k: DirectionKey) {
  const oEvent = document.createEvent('KeyboardEvent');

  function defineConstantGetter(name: string, value: unknown) {
    Object.defineProperty(oEvent, name, {
      get() {
        return value;
      },
    });
  }

  defineConstantGetter('keyCode', k);
  defineConstantGetter('which', k);
  defineConstantGetter('metaKey', false);
  defineConstantGetter('shiftKey', false);
  defineConstantGetter('target', { tagName: '' });

  /* eslint-disable */
    oEvent.initKeyboardEvent('keydown',
      true, true, document.defaultView, false, false, false, false, k, k
    );
    /* eslint-enable */

  document.dispatchEvent(oEvent);
}

declare module '@ember/service' {
  interface Registry {
    game: Game;
  }
}
