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
