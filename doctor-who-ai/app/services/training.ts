import Service from '@ember/service';
import { tracked } from '@glimmer/tracking';

class Stats {
  @tracked averageScore = 0;
  @tracked numberOfGames = 0;
  @tracked numberOfMovevs = 0;
}

export default class Training extends Service {
  // "Epsilon"
  // aka: Chance of random decision
  @tracked minLearningRate = 0.001;
  @tracked maxLearningRate = 0.8;
  // fow fast should we exponentially approach min learning rate
  @tracked learningDecay = 0.0000001;

  // Training Stats
  totalStats = new Stats();
}

declare module '@ember/service' {
  interface Registry {
    training: Training;
  }
}
