import { ALL_MOVES } from './consts';

export function run() {
  // only need to multiply by 3, because 0 counts as our fourth
  let moveIndex = Math.round(Math.random() * 3);

  let move = ALL_MOVES[moveIndex];

  return { move };
}
