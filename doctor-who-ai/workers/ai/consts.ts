export const MOVE = {
  LEFT: 37,
  UP: 38,
  RIGHT: 39,
  DOWN: 40,
} as const;

type Names = keyof typeof MOVE;
export type DirectionKey = typeof MOVE[Names];

export const ALL_MOVES = [MOVE.UP, MOVE.RIGHT, MOVE.DOWN, MOVE.LEFT] as const;
export const MOVE_NAMES = ['Up   ', 'Right', 'Down ', 'Left '] as const;

export const MOVE_KEY_MAP = {
  [MOVE.UP]: 0,
  [MOVE.RIGHT]: 1,
  [MOVE.DOWN]: 2,
  [MOVE.LEFT]: 3,
} as const;

export type InternalMove = typeof MOVE_KEY_MAP[DirectionKey];

// Math.log2(key)
export const VALUE_MAP = {
  0: 0,
  1: 0,
  /* eslint-disable prettier/prettier */
  2:     1, 4:      2, 8:      3, 16:    4,
  32:    5, 64:     6, 128:    7, 256:   8,
  512:   9, 1024:  10, 2048:  11, 4096: 12,
  8192: 13, 16384: 14, 32768: 15,
  /* eslint-enable prettier/prettier */
} as const;

type Value = keyof typeof VALUE_MAP;
export type ValueIndex = typeof VALUE_MAP[Value];
