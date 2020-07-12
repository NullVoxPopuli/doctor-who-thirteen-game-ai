/* global GameManager */
// ^ externally loaded

import { MOVE_KEY_MAP } from './consts';
import { voidFn, isEqual } from './utils';

export function executeMove(gameManager, move) {
  let internalMove = MOVE_KEY_MAP[move];

  gameManager.actuate = voidFn;
  gameManager.keepPlaying = true;
  gameManager.move(internalMove);
}

export function imitateMove(model, move) {
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

export function fakeGameFrom(model) {
  class FakeInputManager {
    constructor() {
      this.on = voidFn;
    }
  }

  class FakeActuator {
    constructor() {
      this.actuate = voidFn;
    }
  }

  class FakeStorage {
    constructor() {
      this.getGameState = () => model;
      this.clearGameState = voidFn;
      this.getBestScore = voidFn;
      this.setGameState = voidFn;
    }
  }

  let gameManager = new GameManager(
    model.grid.size,
    FakeInputManager,
    FakeActuator,
    FakeStorage
  );

  return gameManager;
}
