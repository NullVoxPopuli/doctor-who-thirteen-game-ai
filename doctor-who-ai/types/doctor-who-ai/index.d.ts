import Ember from 'ember';

declare global {
  interface Array<T> extends Ember.ArrayPrototypeExtensions<T> {}
  // interface Function extends Ember.FunctionPrototypeExtensions {}

  type Value =
  | 2    | 4     | 8    | 16
  | 32   | 64    | 128  | 256
  | 512  | 1024  | 2048 | 4096
  | 8192 | 16384 | 32768;

  type Cell2048 = null | { value: Value, position: Record<string, any> }
  type Game2048 = {
    grid: {
      cells: [
        [Cell2048, Cell2048, Cell2048, Cell2048],
        [Cell2048, Cell2048, Cell2048, Cell2048],
        [Cell2048, Cell2048, Cell2048, Cell2048],
        [Cell2048, Cell2048, Cell2048, Cell2048]
      ]
    }
    over: boolean;
    won: boolean;
  }

}

export {};
