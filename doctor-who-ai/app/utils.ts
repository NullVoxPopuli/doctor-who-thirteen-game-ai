export const DOCTOR_NUMBER_MAP = {
  1: '01 - William Hartnell',
  2: '02 - Patrick Troughton',
  3: '03 - Jon Pertwee',
  4: '04 - Tom Baker',
  5: '05 - Peter Davison',
  6: '06 - Colin Baker',
  7: '07 - Sylvester McCoy',
  8: '08 - Paul McGann',
  9: 'War - John Hurt',
  10: '09 - Christopher Eccleston',
  11: '10 - David Tennant',
  12: '11 - Matt Smith',
  13: '12 - Peter Capaldi',
  14: '13 - Jodie Whittaker',
  15: '14 - ???',
} as const;

type Index = keyof typeof DOCTOR_NUMBER_MAP;
export type DoctorLabel = typeof DOCTOR_NUMBER_MAP[Index];

export function biggestTile(game: Game2048) {
  let tiles = game.grid.cells.map((row) => row.map((cell) => (cell ? cell.value : 1))).flat();

  let value = Math.max(...tiles) as Value;

  return { value, num: Math.log2(value) };
}

export function round(num: number) {
  return Math.round(num * 100) / 100;
}

export function printGame(game: GameState, useNum = false) {
  let grid: number[][] = [];

  let gameGrid = game.grid.cells;

  for (let x = 0; x < gameGrid.length; x++) {
    let column = gameGrid[x];

    grid[x] = [];

    for (let y = 0; y < column.length; y++) {
      let value = gameGrid[y][x]?.value || 0;

      if (useNum) {
        value = value === 0 ? 0 : Math.log2(value);
      }

      grid[x][y] = value;
    }
  }

  let max = Math.max(...grid.flat());
  let width = `${max}`.length;

  let toString = (num: number) => {
    let result = `${num}`;
    let existing = result.length;

    for (let i = 0; i < width - existing; i++) {
      result = ` ${result}`;
    }

    return result;
  };

  let result = `Game State:\n`;

  for (let y = 0; y < grid.length; y++) {
    result += '  ';

    for (let x = 0; x < grid[y].length; x++) {
      result += ` ${toString(grid[y][x])}`;
    }

    result += `\n`;
  }

  console.info(result);
}
