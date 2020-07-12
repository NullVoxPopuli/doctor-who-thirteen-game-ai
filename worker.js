/* global importScripts, RL, GameManager */
const window = {};
const dependencies = [
  'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/vendor/rl.js',
  'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/vendor/game.js',
  'https://cdn.jsdelivr.net/npm/reimprovejs@0/dist/reimprove.js',
  'https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/vendor/2048-ai/convnet-min.js'
];

const MOVE = { LEFT: 37, UP: 38, RIGHT: 39, DOWN: 40 };
const ALL_MOVES = [MOVE.UP, MOVE.RIGHT, MOVE.DOWN, MOVE.LEFT];
const MOVE_KEY_MAP = {
  [MOVE.UP]: 0,
  [MOVE.RIGHT]: 1,
  [MOVE.DOWN]: 2,
  [MOVE.LEFT]: 3,
};
// const MOVE_NAMES_MAP = {
//   [MOVE.UP]: 'up',
//   [MOVE.RIGHT]: 'right',
//   [MOVE.DOWN]: 'down',
//   [MOVE.LEFT]: 'left',
// };

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

const groupByValue = (game) => {
  let values = gameTo1DArray(game);

  return values.reduce((group, value) => {
    group[value] = (group[value] || 0) + 1;

    return group;
  }, {});
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

function executeMove(gameManager, move) {
  let internalMove = MOVE_KEY_MAP[move];

  gameManager.actuate = voidFn;
  gameManager.keepPlaying = true;
  gameManager.move(internalMove);

}

function imitateMove(model, move) {
  let gameManager = fakeGameFrom(model);
 
  executeMove(gameManager, move);

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

  /*
   *
   * spec.gamma is the discount rate. When it is zero, the agent will be maximally
   *            greedy and won't plan ahead at all. It will grab all the reward it
   *            can get right away. For example, children that fail the marshmallow
   *            experiment have a very low gamma. This parameter goes up to 1, but
   *            cannot be greater than or equal to 1 (this would make the discounted
   *            reward infinite).
   * spec.epsilon controls the epsilon-greedy policy. High epsilon (up to 1) will
   *              cause the agent to take more random actions. It is a good idea to
   *              start with a high epsilon (e.g. 0.2 or even a bit higher) and decay
   *              it over time to be lower (e.g. 0.05).
   * spec.num_hidden_units: currently the DQN agent is hardcoded to use a neural net
   *                        with one hidden layer, the size of which is controlled with
   *                        this parameter. For each problems you may get away with
   *                        smaller networks.
   * spec.alpha controls the learning rate. Everyone sets this by trial and error and
   *            that's pretty much the best thing we have.
   * spec.experience_add_every: REINFORCEjs won't add a new experience to replay every
   *                            single frame to try to conserve resources and get more
   *                            variaty. You can turn this off by setting this parameter
   *                            to 1. Default = 5
   * spec.experience_size: size of memory. More difficult problems may need bigger memory
   * spec.learning_steps_per_iteration: the more the better, but slower. Default = 20
   * spec.tderror_clamp: for robustness, clamp the TD Errror gradient at this value.
   *
   *
   */
  let spec = {
    update: 'qlearn', // qlearn | sarsa algorithm
    gamma: 0.9, // discount factor, [0, 1)
    epsilon: 0.2, // initial epsilon for epsilon-greedy policy, [0, 1)
    alpha: 0.005, // value function learning rate
    experience_add_every: 1, // number of time steps before we add another experience to replay memory
    experience_size: 100000, // size of experience replay memory
    learning_steps_per_iteration: 10,
    tderror_clamp: 1.0, // for robustness
    num_hidden_units: Math.pow(2, 13), // number of neurons in hidden layer
  };

  let env = {
    getNumStates: () => 4,
    getMaxNumActions: () => 4,
  };

  return new RL.DQNAgent(env, spec);
}

const calculateReward = (move, originalGame, currentGame) => {
  let moveData;
  let clonedGame;

  if (!currentGame) {
    clonedGame = clone(originalGame);
    moveData = imitateMove(clonedGame, move);
  } else {
    clonedGame = currentGame;
    moveData = {
      model: currentGame,
      score: currentGame.score,
      wasMoved: !isEqual(
        currentGame.serialize().grid.cells,
        originalGame.grid.cells,
      ),
    };
  }

  // if (clonedGame.over) {
  //   if (clonedGame.won) {
  //     return 1;
  //   } else {
  //     return -1;
  //   }
  // }

  // if (!moveData.wasMoved) {
  //   // strongly discourage invalid moves
  //   return -1;
  // }

  let grouped = groupByValue(originalGame);
  let newGrouped = groupByValue(moveData.model);

  let highest = Math.max(...Object.keys(grouped));
  let newHighest = Math.max(...Object.keys(newGrouped));

  // highest two were merged, we have a new highest
  if (newHighest > highest) {
    return 1;
  }

  // for each value, determimne if they've been merged
  // highest first
  // let currentValues = Object.keys(newGrouped).sort((a, b) => b - a);

  // let likelyWontMakeItTo = 15; // 2 ^ 30 -- need an upper bound for rewarding

  // for (let value of currentValues) {
  //   // what if it previously didn't exist? but still isn't highest?
  //   if (newGrouped[value] > (grouped[value] || 0)) {
  //     // log2 converts big number to small number
  //     // SEE: inverse of VALUE_MAP
  //     return Math.log2(value) / likelyWontMakeItTo;
  //   }
  // }

  // let bestPossibleMove = outcomesForEachMove(originalGame)[0] || {};
  // let bestPossibleScore = bestPossibleMove.score;

  // if (moveData.score >= bestPossibleScore) {
  //   return 1;
  // }

  if (moveData.score > originalGame.score) {
    return 1 - originalGame.score / moveData.score;

    // Provide a bigger reward the higher the merge value is

    // let additionalPoints = (moveData.score = originalGame.score);

    // let fractionalScore = additionalPoints / Math.pow(2, 13); // highest possible single merge score;

    // return fractionalScore > 1 ? 1 : fractionalScore;
  }

  // next score is equal to current
  // it's possible that we need to do something that doesn't
  // change our score before getting to something good
  return 0; // - originalGame.score / bestPossibleScore;
};

async function runRNN(game, trainingData) {
  // return runRL(game, trainingData);
  return runReImprove(game, trainingData);
}

let _reImprove = {};
let iterations = 0;

async function runReImprove(game, trainingData) {
  Object.freeze(game.grid);
  let ReImprove = window.ReImprove;

  function createNetwork() {
    const modelFitConfig = {              // Exactly the same idea here by using tfjs's model's
        epochs: 1,                        // fit config.
        stepsPerEpoch: 16
      };

    const numActions = 3; // (including 0?)                 // The number of actions your agent can choose to do
    const inputSize = 16;                // Inputs size (10x10 image for instance)
    const temporalWindow = 1;             // The window of data which will be sent yo your agent
                                          // For instance the x previous inputs, and what actions the agent took

    const totalInputSize = inputSize * temporalWindow + numActions * temporalWindow + inputSize;

    const network = new ReImprove.NeuralNetwork();
    network.InputShape = [totalInputSize];
    network.addNeuralNetworkLayers([
      {type: 'dense', units: Math.pow(2, 8), activation: 'relu'},
      {type: 'dense', units: Math.pow(2, 11), activation: 'relu'},
      {type: 'dense', units: Math.pow(2, 10), activation: 'relu'},
      {type: 'dense', units: Math.pow(2, 9), activation: 'relu'},
      {type: 'dense', units: Math.pow(2, 8), activation: 'relu'},
      {type: 'dense', units: Math.pow(2, 6), activation: 'relu'},
      {type: 'dense', units: numActions, activation: 'softmax'}
    ]);
    // Now we initialize our model, and start adding layers
    const model = new ReImprove.Model.FromNetwork(network, modelFitConfig);

    // Finally compile the model, we also exactly use tfjs's optimizers and loss functions
    // (So feel free to choose one among tfjs's)
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'})

    // Every single field here is optionnal, and has a default value. Be careful, it may not
    // fit your needs ...

    const teacherConfig = {
      lessonsQuantity: 10000,                  
      lessonLength: 20,                    
      lessonsWithRandom: 10,                  // We do not care about full random sessions
      epsilon: 0.5,                            // Maybe a higher random rate at the beginning ?
      epsilonDecay: 0.995,                   
      epsilonMin: 0.05,
      gamma: 0.9                            
    };

    const agentConfig = {
      model: model,                          // Our model corresponding to the agent
      agentConfig: {
          memorySize: 1000,                      // The size of the agent's memory (Q-Learning)
          batchSize: 128,                        // How many tensors will be given to the network when fit
          temporalWindow: temporalWindow         // The temporal window giving previous inputs & actions
      }
    };

    const academy = new ReImprove.Academy();    // First we need an academy to host everything
    const teacher = academy.addTeacher(teacherConfig);
    const agent = academy.addAgent(agentConfig);

    academy.assignTeacherToAgent(agent, teacher);

    return { model, academy, agent, teacher };
  }
 
  async function getMove(game) {
    let inputs = gameTo1DArray(game);

    let result = await _reImprove.academy.step([               // Let the magic operate ...
        { teacherName: _reImprove.teacher, agentsInput: inputs }
    ]);

    let moveIndex = result.get(_reImprove.agent);
    let move = ALL_MOVES[moveIndex];

    return move;
  }

  if (!_reImprove.agent) {
    Object.assign(_reImprove, createNetwork());
  }

  async function trainABit(originalGame) {
    console.debug('Running simulated game to completion...');
    let moves = 0;
    let start = new Date();
    // copy the game
    // run to completion
    let clonedGame = clone(originalGame);
    let gameManager = fakeGameFrom(clonedGame);


    while (!gameManager.over || moves > 10000) {
      moves++;

      if (moves % 100 === 0) {
        console.debug(`at ${moves} moves...`);
      }

      let previousGame = clone(gameManager);
      let move = await getMove(gameManager);

      executeMove(gameManager, move);

      let internalMove = MOVE_KEY_MAP[move];
      let reward = calculateReward(internalMove, previousGame, gameManager);

      _reImprove.academy.addRewardToAgent(_reImprove.agent, reward);    

    }

    console.debug('Simulation Finished', {
      moves,
      score: gameManager.score,
      time: new Date() - start,
    });  
  }

  // await trainABit(game);

  let move = await getMove(game);
  let reward = calculateReward(move, game);

  _reImprove.academy.addRewardToAgent(_reImprove.agent, reward);

  self.postMessage({ 
    type: 'move', 
    move,
    // trainingData: rnn.toJSON() 
  });
}

/**
 * Initially, this started out as an A* algorithm, constrained by depth
 *  - original version from https://github.com/nloginov/2048-ai
 *
 * Modifications:
 * - use weighted score, penalizing a higher number of moves to achieve a score
 * - instead of blindly searching until maxLevel,
 *   maxLevel will only be reached in the event of ties in score
 *
 */
function treeAI(model) {
  let bestNode;
  let treeSize = 0;
  let bestScore = 0;
  let bestHops = 1000;

  let rootNode = {
    value: { model },
    children: [],
  };

  function updateBest(childNode) {
    if (childNode === rootNode) {
      return;
    }

    if (childNode.weightedScore < bestScore) {
      return;
    }

    // if the score is equal, let's choose the least hops

    let root = childNode;
    let hops = 0;

    while (root.parent !== undefined && root.parent.move) {
      root = root.parent;
      hops++;
    }

    // if (hops < bestHops) {
    //   if (hops === 0) {
        if (childNode.weightedScore > bestScore) {
          bestNode = root;
          bestScore = childNode.weightedScore;
        }

        return;
    //   }

    //   bestHops = hops;
    //   bestNode = root;
    //   bestScore = childNode.weightedScore;
    // }
  }

  function expandTree(node, level) {
    updateBest(node);

    if (level >= 6) {
      return;
    }

    const enumerateMoves = () => {
      for (let move of ALL_MOVES) {
        let copyOfModel = clone(node.value);
        let moveData = imitateMove(copyOfModel.model, move);

        if (!moveData.wasMoved) {
          continue;
        }

        treeSize++;

        let scoreChange = moveData.score - model.score;

        // this is a very important strategy
        // let multiplier = edgeMultiplierFor(moveData.model);

        // let weightedScore = scoreChange / 1 / ((level + 1) * multiplier);

        let weightedScore = level === 0 ? scoreChange : scoreChange / level;

        // weightedScore = weightedScore * calculateReward(move, node.value.model);

        node.children.push({
          // penalize scores with higher depth
          // this takes the nth root of the score where n is the number of moves
          // weightedScore: moveData.score, //Math.pow(moveData.score, 1 / (level + 1)),
          // weightedScore: moveData.score / 1 / (level * 2 + 1),
          // weightedScore: moveData.score,
          weightedScore,

          value: moveData,
          children: [],
          move: move,
          parent: node,
        });
      }
    };

    // to try to account for misfortune
    enumerateMoves();
    // enumerateMoves();
    // enumerateMoves();

    for (let childNode of node.children) {
      expandTree(childNode, level + 1);
    }
  }

  let initialLevel = 0;

  while (bestNode === undefined || initialLevel < -3) {
    expandTree(rootNode, initialLevel);

    initialLevel = initialLevel - 1;
  }

  let bestMove = bestNode.move;

  // console.debug(
  //   `Best Move: ${bestMove} aka ${MOVE_NAMES_MAP[bestMove]} out of ${treeSize} options`
  // );
  // console.debug(
  //   `with expected score change of ${model.score} => ${bestNode.value.model.score}`
  // );

  return bestMove;
}

function runAStar(game, maxLevel) {
  Object.freeze(game.grid);

  console.debug('-------------- Calculate Move -----------------');
  let initialTime = new Date();

  let move = treeAI(game, Math.max(maxLevel, 4));

  console.debug(`Time: ${new Date() - initialTime}ms`);

  self.postMessage({ type: 'move', move });
}





async function runRL(game, trainingData) {
  Object.freeze(game.grid);

  if (!rnn) {
    rnn = createRnn();

    if (trainingData) {
      rnn.fromJSON(trainingData);
    }
  }

  let inputs = gameTo1DArray(game);

  // normalized to 0-getMaxNumAction() - 1
  let moveIndex = await rnn.act(inputs);
  let move = ALL_MOVES[moveIndex];
  let reward = calculateReward(move, game);

  rnn.learn(reward);

  self.postMessage({ type: 'move', move, trainingData: rnn.toJSON() });
}

function random() {
  // only need to multiply by 3, because 0 counts as our fourth
  let moveIndex = Math.round(Math.random() * 3);

  let move = ALL_MOVES[moveIndex];

  self.postMessage({ type: 'move', move });
}

function run({ game, algorithm, trainingData }) {
  switch (algorithm) {
    case 'RNN':
      return runRNN(game, trainingData);
    case 'random':
      return random();
    case 'a-star':
      return runAStar(game, 6);
    case '2048':
      return run2048(game);
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
      console.debug(`Loaded ${depUrl}`);
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

let net;
let netAi;

function run2048(game) {
  if (!net) {
    var layer_defs = [];
    layer_defs.push({type:'input', out_sx:4, out_sy:4, out_depth:16});
    var net_params=[256, 2400, 1600, 800, 400, 400, 200, 200, 100, 100, 4];
    for(var i= 1;i<net_params.length-1;i++){
        layer_defs.push({type:'fc', num_neurons:net_params[i], activation:'relu'});
    }

    layer_defs.push({type:'fc', num_neurons:4});

    net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    // document.getElementById('ai-info').innerHTML="Downloading model......(about 25M). <br /><strong>NN AI won't work until the model is loaded</strong><div class='progress-container'><div class='progressbar' id='progressbar' style='width: 0%'></div></div>";
    var oReq = new XMLHttpRequest();
    oReq.open("GET", "https://raw.githubusercontent.com/NullVoxPopuli/doctor-who-thirteen-game-ai/master/vendor/2048-ai/model.bin", true);
    oReq.responseType = "arraybuffer";
    var updateProgress = function (oEvent) {
        // var pb = document.getElementById('progressbar');
        // pb.innerHTML=""+ oEvent.loaded+"/25484016";
        // var width = 100 * oEvent.loaded/25484016;
        // pb.style.width = width + '%'; 
    }
    oReq.addEventListener("progress", updateProgress, false);

    oReq.onload = function (oEvent) {
        var arrayBuffer = oReq.response; // Note: not oReq.responseText
        if (arrayBuffer) {
            floatArray = new Float32Array(arrayBuffer);

            var ptr=0|0;
            for(var i=1;i<net_params.length;i++)
            {
                var l = i*2-1;
                for(var x=0;x<net_params[i-1];x++){
                    for(var y=0;y<net_params[i];y++){
                            net.layers[l].filters[y].w[x] = floatArray[ptr];
                            ptr++;
                    }
                }
                for(var y=0;y<net_params[i];y++){
                        net.layers[l].biases.w[y] = floatArray[ptr];
                            ptr++;
                }
            }
        }
    };

    oReq.send(null);
  }

  let grid = clone(game.grid);
  let netAi = new AI(grid);

  let best = netAi.getBest();

  let move;
  for(var i=0;i<4;i++){
    let m=best.moves[i];

    let cloned = { grid: clone(grid) };
    let moveData = imitateMove(cloned, m);

    if (moveData.wasMoved) {
      move = m;
      break;
    }
  }

  self.postMessage({ type: 'move', move: ALL_MOVES[best.move] });
}


function AI(grid) {
  this.grid = grid;
  this.v = new convnetjs.Vol(4, 4, 16, 0.0);
}
AI.prototype.make_input = function(){
    this.v.setConst(0.0);
    for(var i=0;i<4;i++){
        for(var j=0;j<4;j++){
            if(this.grid.cells[j][i] !== null){
                var v = this.grid.cells[j][i].value|0;
                var k =0;
                while(v>2){
                    v >>=1;
                    k+=1;
                };
                this.v.w[i*4*16+j*16+k]=1.;
            }
        }
    }

    
}

// performs a search and returns the best move
AI.prototype.getBest = function() {
 //console.log(this.grid);
 this.make_input()
 //console.log(this.v);
 var p = net.forward(this.v).w;
  var m=[0,1,2,3];
  m.sort(function(i,j){return p[j]-p[i]});
  //console.log(p);
  //console.log(m);
  m = m.map(function(x){ return (x+3)%4;})
  return {move:m[0], moves:m};
}


AI.prototype.translate = function(move) {
 return {
    0: 'up',
    1: 'right',
    2: 'down',
    3: 'left'
  }[move];
}
