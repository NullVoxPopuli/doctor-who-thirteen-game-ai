/* global GameManager */
// ^ externally loaded

import { MOVE_KEY_MAP } from './consts';
import { voidFn, isEqual } from './utils';

export function imitateMove(model, move) {
  let gameManager = fakeGameFrom(model);
  let internalMove = MOVE_KEY_MAP[move];

  gameManager.actuate = voidFn;
  gameManager.keepPlaying = true;
  gameManager.move(internalMove);

  let serialized = gameManager.serialize();

  // Object.freeze(serialized);

  return {
    move,
    score: gameManager.score,
    model: serialized,
    // NOTE: the score is not updated for the fake manager
    // wasMoved: serialized.score !== model.score,
    wasMoved: !isEqual(serialized.grid.cells, model.grid.cells),
  };
}

function fakeGameFrom(model) {
  class FakeInputManager {
    on = voidFn;
  }

  class FakeActuator {
    actuate = voidFn;
  }

  class FakeStorage {
    getGameState = () => model;
    clearGameState = voidFn;
    getBestScore = voidFn;
    setGameState = voidFn;
  }

  let gameManager = new GameManager(
    model.grid.size,
    FakeInputManager,
    FakeActuator,
    FakeStorage
  );

  return gameManager;
}
