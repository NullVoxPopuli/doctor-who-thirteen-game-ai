export const voidFn = () => undefined;
export const clone = <T>(obj: T): T => JSON.parse(JSON.stringify(obj));

export const isEqual = (a: GameCells, b: GameCells) => {
  // a and b have the same dimensions
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      let av = a[i][j];
      let bv = b[i][j];
      let avv = av && av.value;
      let bvv = bv && bv.value;

      if (avv !== bvv) {
        return false;
      }
    }
  }

  return true;
};

export const gameTo1DArray = (game: Game2048) => {
  return game.grid.cells.flat().map((cell) => (cell ? cell.value : 0));
};

export const groupByValue = (game: Game2048) => {
  let values = gameTo1DArray(game);

  return values.reduce((group, value) => {
    group[value] = (group[value] || 0) + 1;

    return group;
  }, {});
};

export async function loadDependencies(dependencies: string[]) {
  await Promise.all(
    dependencies.map(async (depUrl) => {
      let response = await fetch(depUrl);
      let script = await response.text();
      let blob = new Blob([script], { type: 'text/javascript' });
      let blobLink = URL.createObjectURL(blob);

      // yolo
      importScripts(blobLink);
    })
  );

  self.postMessage({ type: 'ack' });
}
