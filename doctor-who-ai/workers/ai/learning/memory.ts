import random from 'random';

export class Memory<T> {
  declare size: number;

  #memory: T[] = [];

  constructor(size: number) {
    this.size = size;
  }

  add(item: T) {
    this.#memory.push(item);

    if (this.#memory.length > this.size) {
      this.#memory.shift();
    }
  }

  recallRandomly(count: number) {
    return sample(this.#memory, count);
  }
}

function sample(arr: unknown[], count: number) {
  let results = [];
  let previousIndicies: number[] = [];

  while (results.length < count) {
    let index = random.int(0, arr.length - 1);

    if (previousIndicies.includes(index)) {
      continue;
    }

    previousIndicies.push(index);

    results.push(arr[index]);
  }

  return results;
}
