import * as tf from '@tensorflow/tfjs';
import ReImprove from 'reimprovejs';

// import ml5 from 'ml5';

const fileName = 're-improve.model';
const dataLocation = `downloads://${fileName}`;
// const fileInfoLocation = `/${fileName}.json`;

// const fileInfo = {
//   model: '/model.json',
//   metadata: '/model.meta.json',
//   weights: '/model.weights.bin',
// };

export async function useGPU() {
  if (tf.getBackend() !== 'webgl') {
    await tf.setBackend('webgl');
  }

  await tf.ready();
}

export async function save(network) {
  // await network.export(fileName, 'downloads');

  await network.model.save(dataLocation);
}

export async function getNetwork() {
  let network = createNetwork();

  try {
    // await network.load(fileInfo);
    // ReImprove.JS is broken and can't handle loadLayersModel
    // network.model = await tf.loadLayersModel(fileInfoLocation)

    return network;
  } catch (e) {
    console.debug(e);

    return network;
  }
}

function createNetwork() {
  /**
   * ML5
   */
  // return ml5.neuralNetwork({
  //   debug: true,
  //   inputs: 16,
  //   outputs: 4,
  //   layers: [
  //     { type: 'dense', units: Math.pow(2, 8), activation: 'relu' },
  //     { type: 'dense', units: Math.pow(2, 11), activation: 'relu' },
  //     { type: 'dense', units: Math.pow(2, 10), activation: 'relu' },
  //     { type: 'dense', units: Math.pow(2, 9), activation: 'relu' },
  //     { type: 'dense', units: Math.pow(2, 8), activation: 'relu' },
  //     { type: 'dense', units: Math.pow(2, 6), activation: 'relu' },
  //     { type: 'dense', units: Math.pow(2, 5), activation: 'relu' },
  //     { type: 'dense', units: 4, activation: 'softmax' },
  //   ],
  // });

  /**
   * ReImprove
   */
  const modelFitConfig = {
    // Exactly the same idea here by using tfjs's model's
    epochs: 1, // fit config.
    stepsPerEpoch: 16,
  };

  const numActions = 3; // The number of actions your agent can choose to do
  const inputSize = 16; // Inputs size (10x10 image for instance)
  const temporalWindow = 1; // The window of data which will be sent yo your agent
  // For instance the x previous inputs, and what actions the agent took

  const totalInputSize = inputSize * temporalWindow + numActions * temporalWindow + inputSize;

  const network = new ReImprove.NeuralNetwork();

  network.InputShape = [totalInputSize];
  network.addNeuralNetworkLayers([
    { type: 'dense', units: Math.pow(2, 8), activation: 'relu' },
    { type: 'dense', units: Math.pow(2, 11), activation: 'relu' },
    { type: 'dense', units: Math.pow(2, 10), activation: 'relu' },
    { type: 'dense', units: Math.pow(2, 9), activation: 'relu' },
    { type: 'dense', units: Math.pow(2, 8), activation: 'relu' },
    { type: 'dense', units: Math.pow(2, 6), activation: 'relu' },
    { type: 'dense', units: Math.pow(2, 5), activation: 'relu' },
    { type: 'dense', units: numActions, activation: 'softmax' },
  ]);
  // Now we initialize our model, and start adding layers
  const model = new ReImprove.Model.FromNetwork(network, modelFitConfig);

  // Finally compile the model, we also exactly use tfjs's optimizers and loss functions
  // (So feel free to choose one among tfjs's)
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  return model;
}

const trainingOptions = {
  batchSize: 32,
  epochs: 16,
};

export async function train(network) {
  await network.train(trainingOptions);
}

export async function getAgent(model) {
  // Every single field here is optionnal, and has a default value. Be careful, it may not
  // fit your needs ...

  const teacherConfig = {
    lessonsQuantity: 100, // Number of training lessons before only testing agent
    lessonsLength: 100, // The length of each lesson (in quantity of updates)
    lessonsWithRandom: 5, // How many random lessons before updating epsilon's value
    epsilon: 1, // Q-Learning values and so on ...
    epsilonDecay: 0.995, // (Random factor epsilon, decaying over time)
    epsilonMin: 0.7,
    gamma: 1, // (Gamma = 1 : agent cares really much about future rewards)
  };

  const agentConfig = {
    model: model, // Our model corresponding to the agent
    agentConfig: {
      memorySize: 5000, // The size of the agent's memory (Q-Learning)
      batchSize: 16, // How many tensors will be given to the network when fit
      temporalWindow: 1, // The temporal window giving previous inputs & actions
    },
  };

  const academy = new ReImprove.Academy(); // First we need an academy to host everything
  const teacher = academy.addTeacher(teacherConfig);
  const agent = academy.addAgent(agentConfig);

  academy.assignTeacherToAgent(agent, teacher);

  return {
    reward: (value: number) => academy.addRewardToAgent(agent, value),
    step: async (inputs: number[]) => {
      let result = await academy.step([{ teacherName: teacher, agentsInput: inputs }]);

      return result.get(agent);
    },
  };
}
