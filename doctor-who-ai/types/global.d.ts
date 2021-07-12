// Types for compiled templates
declare module 'doctor-who-ai/templates/*' {
  import { TemplateFactory } from 'htmlbars-inline-precompile';
  const tmpl: TemplateFactory;
  export default tmpl;
}

declare type Value =
  | 2
  | 4
  | 8
  | 16
  | 32
  | 64
  | 128
  | 256
  | 512
  | 1024
  | 2048
  | 4096
  | 8192
  | 16384
  | 32768;

declare type CellPosition = { x: number; y: number };
declare type Cell2048 = null | { value: Value; position: CellPosition };
declare type GameCells = [
  [Cell2048, Cell2048, Cell2048, Cell2048],
  [Cell2048, Cell2048, Cell2048, Cell2048],
  [Cell2048, Cell2048, Cell2048, Cell2048],
  [Cell2048, Cell2048, Cell2048, Cell2048]
];
declare type GameGrid = {
  cells: GameCells;
  size: number;
};

declare interface GameState {
  grid: GameGrid;
  over: boolean;
  won: boolean;
  keepPlaying: boolean;
}

declare interface Game2048 extends GameState {
  score: number;
  serialize: () => Game2048;
  actuate?: () => void;
  move: (move: 0 | 1 | 2 | 3) => void;
}

declare module 'ai/rnn/vendor/app.map-worker-edition' {
  export class GameManager implements Game2048 {
    constructor(size: number, inputManager: any, actuator: any, storage: any);
    score: number;
    serialize: () => Game2048;
    actuate?: (() => void) | undefined;
    move: (move: 0 | 1 | 2 | 3) => void;
    grid: GameGrid;
    over: boolean;
    won: boolean;
    keepPlaying: boolean;
    addStartTiles: () => void;
  }
}
