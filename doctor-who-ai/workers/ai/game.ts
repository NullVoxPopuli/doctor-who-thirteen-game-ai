/* global GameManager */
// ^ externally loaded

import { MOVE_KEY_MAP } from './consts';
import { voidFn, isEqual } from './utils';

import type { DirectionKey } from 'doctor-who-ai/services/ai/consts';

export function executeMove(gameManager: Game2048, move: DirectionKey) {
  let internalMove = MOVE_KEY_MAP[move];

  gameManager.actuate = voidFn;
  gameManager.keepPlaying = true;
  gameManager.move(internalMove);
}

export function imitateMove(model: Game2048, move: DirectionKey) {
  let gameManager = fakeGameFrom(model);

  executeMove(gameManager, move);

  let serialized = gameManager.serialize();

  return {
    move,
    score: gameManager.score,
    model: serialized,
    wasMoved: !isEqual(serialized.grid.cells, model.grid.cells),
  };
}

export function fakeGameFrom(model: Game2048) {
  class FakeInputManager {
    declare on: () => void;

    constructor() {
      this.on = voidFn;
    }
  }

  class FakeActuator {
    declare actuate: () => void;

    constructor() {
      this.actuate = voidFn;
    }
  }

  class FakeStorage {
    declare getGameState: () => Game2048;
    declare clearGameState: () => void;
    declare getBestScore: () => void;
    declare setGameState: () => void;

    constructor() {
      this.getGameState = () => model;
      this.clearGameState = voidFn;
      this.getBestScore = voidFn;
      this.setGameState = voidFn;
    }
  }

  let gameManager = new GameManager(model.grid.size, FakeInputManager, FakeActuator, FakeStorage);

  return gameManager;
}
