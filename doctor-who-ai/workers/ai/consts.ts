export const MOVE = {
  LEFT: 37, UP: 38, RIGHT: 39, DOWN: 40
} as const;

type Names = keyof typeof MOVE;
export type DirectionKey =  typeof MOVE[Names]

export const ALL_MOVES = [MOVE.UP, MOVE.RIGHT, MOVE.DOWN, MOVE.LEFT] as const;

export const MOVE_KEY_MAP = {
  [MOVE.UP]: 0,
  [MOVE.RIGHT]: 1,
  [MOVE.DOWN]: 2,
  [MOVE.LEFT]: 3,
} as const;

export type InternalMove = typeof MOVE_KEY_MAP[DirectionKey];
