import random from 'random';

export class Memory<T> {
  declare size: number;
  declare keepBestPercent?: number;
  declare keepBestVia?: (mem: T) => number;

  #memory: T[] = [];

  constructor(size: number, keepBestPercent?: number, keepBestVia?: (mem: T) => number) {
    this.size = size;
    this.keepBestPercent = keepBestPercent;
    this.keepBestVia = keepBestVia;
  }

  add(item: T) {
    this.#memory.push(item);

    if (this.#memory.length > this.size) {
      if (this.keepBestVia && this.keepBestPercent) {
        let kept = this.recallTopBy(this.keepBestVia, this.keepBestPercent);

        for (let i = 0; i < this.#memory.length; i++) {
          if (!kept.includes(this.#memory[i])) {
            this.#memory.splice(i, 1);
          }
        }
      } else {
        this.#memory.shift();
      }
    }
  }

  recall() {
    return this.#memory;
  }

  recallRandomly(count: number) {
    return sample(this.#memory, count);
  }

  recallTopBy(getter: (item: T) => number, percent = 0.1) {
    let sorted = this.#memory.sort((a, b) => getter(b) - getter(a));

    let numItems = Math.ceil(sorted.length * percent);

    return sorted.slice(0, numItems);
  }
}

function sample(arr: unknown[], count: number) {
  let results = [];
  let previousIndicies: number[] = [];

  while (results.length < count && results.length !== arr.length) {
    let index = random.int(0, arr.length - 1);

    if (previousIndicies.includes(index)) {
      continue;
    }

    previousIndicies.push(index);

    results.push(arr[index]);
  }

  return results;
}
