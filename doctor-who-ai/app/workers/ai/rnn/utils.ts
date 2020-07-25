import * as tf from '@tensorflow/tfjs';

export const voidFn = () => undefined;
export const clone = <T>(obj: T): T => JSON.parse(JSON.stringify(obj));

export const isEqual = (a: GameCells, b: GameCells) => {
  // a and b have the same dimensions
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      let av = a[i][j];
      let bv = b[i][j];
      let avv = av?.value;
      let bvv = bv?.value;

      if (avv !== bvv) {
        return false;
      }
    }
  }

  return true;
};

export const gameTo1DArray = (game: GameState) => {
  return game.grid.cells.flat().map((cell) => (cell ? cell.value : 0));
};

export const groupByValue = (game: GameState) => {
  let values = gameTo1DArray(game);

  return values.reduce((group, value) => {
    group[value] = (group[value] || 0) + 1;

    return group;
  }, {});
};

export function gameToTensor(game: GameState) {
  let result: number[][] = [];
  let cells = game.grid.cells;

  for (let i = 0; i < cells.length; i++) {
    result[i] = [];

    for (let j = 0; j < cells.length; j++) {
      let cell = cells[i][j];

      let value = cell?.value || 0;
      let k = value === 0 ? 0 : Math.log2(value);

      result[i][j] = k;
    }
  }

  return tf.tensor2d(result);
}

export function printGame(game: GameState) {
  let grid: number[][] = [];

  let gameGrid = game.grid.cells;

  for (let x = 0; x < gameGrid.length; x++) {
    let row = gameGrid[x];

    grid[x] = [];

    for (let y = 0; y < row.length; y++) {
      grid[x][y] = gameGrid[x][y]?.value || 0;
    }
  }

  console.info(grid);
}
