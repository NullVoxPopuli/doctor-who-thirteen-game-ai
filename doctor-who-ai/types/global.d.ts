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

declare type Cell2048 = null | { value: Value; position: Record<string, any> };
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
declare type Game2048 = {
  grid: GameGrid;
  score: number;
  over: boolean;
  won: boolean;
  serialize: () => Game2048;
  actuate?: () => void;
  keepPlaying?: boolean;
  move: (move: 0 | 1 | 2 | 3) => void;
};
