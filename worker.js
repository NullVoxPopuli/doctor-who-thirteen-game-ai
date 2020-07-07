/* global importScripts, RL, GameManager */

const dependencies = [
  'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/vendor/rl.js',
  'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/vendor/game.js',
];

const MOVE = { LEFT: 37, UP: 38, RIGHT: 39, DOWN: 40 };
const ALL_MOVES = [MOVE.UP, MOVE.RIGHT, MOVE.DOWN, MOVE.LEFT];
const MOVE_KEY_MAP = {
  [MOVE.UP]: 0,
  [MOVE.RIGHT]: 1,
  [MOVE.DOWN]: 2,
  [MOVE.LEFT]: 3,
};
const MOVE_NAMES_MAP = {
  [MOVE.UP]: 'up',
  [MOVE.RIGHT]: 'right',
  [MOVE.DOWN]: 'down',
  [MOVE.LEFT]: 'left',
};

const voidFn = () => undefined;
const clone = (obj) => JSON.parse(JSON.stringify(obj));
const isEqual = (a, b) => {
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

const gameTo1DArray = (game) => {
  return game.grid.cells.flat().map((cell) => (cell ? cell.value : 0));
};

/////////////////////////////////////////////////////////////////////////
// Game Helper Code
/////////////////////////////////////////////////////////////////////////

function fakeGameFrom(model) {
  class FakeInputManager {
    on = voidFn;
  }

  class FakeActuator {
    actuate = voidFn;
  }

  class FakeStorage {
    getGameState = () => model;
    clearGameState = voidFn;
    getBestScore = voidFn;
    setGameState = voidFn;
  }

  let gameManager = new GameManager(
    model.grid.size,
    FakeInputManager,
    FakeActuator,
    FakeStorage
  );

  return gameManager;
}

function imitateMove(model, move) {
  let gameManager = fakeGameFrom(model);
  let internalMove = MOVE_KEY_MAP[move];

  gameManager.actuate = voidFn;
  gameManager.keepPlaying = true;
  gameManager.move(internalMove);

  let serialized = gameManager.serialize();

  // Object.freeze(serialized);

  return {
    move,
    score: gameManager.score,
    model: serialized,
    // NOTE: the score is not updated for the fake manager
    // wasMoved: serialized.score !== model.score,
    wasMoved: !isEqual(serialized.grid.cells, model.grid.cells),
  };
}

/////////////////////////////////////////////////////////////////////////
// Worker-related code
/////////////////////////////////////////////////////////////////////////

let rnn;

function createRnn() {
  // followed:
  //   https://codepen.io/Samid737/pen/opmvaR
  //   https://github.com/karpathy/reinforcejs

  let spec = {
    update: 'qlearn', // qlearn | sarsa algorithm
    gamma: 0.9, // discount factor, [0, 1)
    epsilon: 0.001, // initial epsilon for epsilon-greedy policy, [0, 1)
    alpha: 0.001, // value function learning rate
    experience_add_every: 5, // number of time steps before we add another experience to replay memory
    experience_size: 5000, // size of experience replay memory
    learning_steps_per_iteration: 20,
    tderror_clamp: 1.0, // for robustness
    num_hidden_units: Math.pow(2, 13), // number of neurons in hidden layer
  };

  let env = {
    getNumStates: () => 4,
    getMaxNumActions: () => 4,
  };

  return new RL.DQNAgent(env, spec);
}

function outcomesForEachMove(game) {
  let result = [];

  for (let move of ALL_MOVES) {
    let clonedGame = clone(game);
    let moveData = imitateMove(clonedGame, move);

    result.push(moveData);
  }

  // biggest first
  return result.sort((a, b) => b.score - a.score);
}

const calculateReward = (move, originalGame) => {
  let clonedGame = clone(originalGame);
  let moveData = imitateMove(clonedGame, move);

  if (clonedGame.over) {
    if (clonedGame.won) {
      return 1;
    } else {
      return -1;
    }
  }

  if (!moveData.wasMoved) {
    return -0.01;
  }

  let bestPossibleMove = outcomesForEachMove(originalGame)[0] || {};
  let bestPossibleScore = bestPossibleMove.score || 10000000;

  if (moveData.score >= bestPossibleScore) {
    return 1 - originalGame.score / moveData.score;
  }

  if (moveData.score > originalGame.score) {
    return (1 - originalGame.score / moveData.score) / 2;
  }

  // next score is equal to current
  // it's possible that we need to do something that doesn't
  // change our score before getting to something good
  // TODO: penalize more when thare are available moves of higher value
  return -0.01;
};

async function runRNN(game, trainingData) {
  Object.freeze(game.grid);

  if (!rnn) {
    rnn = createRnn();

    if (trainingData) {
      rnn.fromJSON(trainingData);
    }
  }

  let inputs = gameTo1DArray(game);

  // normalized to 0-1
  let moveIndex = await rnn.act(inputs);
  let move = ALL_MOVES[moveIndex];
  let reward = calculateReward(move, game);

  rnn.learn(reward);

  console.debug({ reward, move, moveName: MOVE_NAMES_MAP[move] });
  self.postMessage({ type: 'move', move, trainingData: rnn.toJSON() });
}

function run({ game, algorithm, trainingData }) {
  switch (algorithm) {
    case 'RNN':
      return runRNN(game, trainingData);
    default:
      console.error(...arguments);
      throw new Error('Unrecognized Algorithm', algorithm);
  }
}

async function loadDependencies() {
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

self.onmessage = function (e) {
  let { data } = e;

  switch (data.type) {
    case 'ready':
      return loadDependencies();

    case 'run':
      // it's possible to have ~ 3 moves of nothing happening
      return run(data);
    default:
      console.error(data);
      throw new Error('Unrecognized Message');
  }
};
