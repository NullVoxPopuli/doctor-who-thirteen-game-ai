import Service from '@ember/service';
import { action, computed } from '@ember/object';
import { tracked } from '@glimmer/tracking';

type HistoryEntry = {
  bestScore?: number;
  score: number;
  averageScore: number;
  time: number;
  averageTime: number;
};

type GameInfo = {
  score: number;
  time: number;
};

const HISTORY_SIZE = 60;

const INITIAL = {
  bestScore: 0,
  averageScore: 0,
  averageTime: 0,
};

export default class GameHistory extends Service {
  @tracked totalGames = 0;
  @tracked history: HistoryEntry[] = [];

  // TODO: replace with @cached
  @computed('history.length')
  get latest() {
    return {
      ...(this.history[this.history.length - 1] || INITIAL),
      bestScore: Math.max(...this.history.map((h) => h.score)),
    };
  }

  get scores() {
    return this.history.map((h) => h.score);
  }

  get averageScores() {
    return this.history.map((h) => h.averageScore);
  }

  @action
  addGame({ score, time }: GameInfo) {
    const scores = [...this.scores, score];
    const times = [...this.history.map((h) => h.time), time];

    this.history.push({
      score,
      time,
      averageScore: average(scores),
      averageTime: average(times),
    });

    this.totalGames += 1;

    this.trimToWindow();
  }

  @action
  trimToWindow() {
    this.history = this.history.slice(Math.max(this.history.length - HISTORY_SIZE, 0));
  }
}

function average(numbers: number[]) {
  return numbers.reduce((a, b) => a + b, 0) / numbers.length;
}

declare module '@ember/service' {
  interface Registry {
    history: History;
  }
}
