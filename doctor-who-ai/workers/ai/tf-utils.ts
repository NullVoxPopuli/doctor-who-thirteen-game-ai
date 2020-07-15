import tf from '@tensorflow/tfjs';

const fileName = 're-improve.model';
const dataLocation = `downloads://${fileName}`;
const fileInfoLocation = `/${fileName}.json`;

export async function useGPU() {
  if (tf.getBackend() !== 'webgl') {
    await tf.setBackend('webgl');
  }

  await tf.ready();
}

export async function save(network: tf.LayersModel) {
  await network.save(dataLocation);
}

export async function getNetwork(): Promise<tf.LayersModel> {
  try {
    return await tf.loadLayersModel(fileInfoLocation);
  } catch (e) {
    console.debug(e);

    return createNetwork();
  }
}

function createNetwork() {
  let model = tf.sequential();

  model.add(tf.layers.dense({ units: Math.pow(2, 9), inputShape: [16], activation: 'relu' }));
  model.add(tf.layers.dense({ units: Math.pow(2, 11), activation: 'relu' }));
  model.add(tf.layers.dense({ units: Math.pow(2, 10), activation: 'relu' }));
  model.add(tf.layers.dense({ units: Math.pow(2, 9), activation: 'relu' }));
  model.add(tf.layers.dense({ units: Math.pow(2, 8), activation: 'relu' }));
  model.add(tf.layers.dense({ units: Math.pow(2, 6), activation: 'relu' }));
  model.add(tf.layers.dense({ units: Math.pow(2, 5), activation: 'relu' }));
  model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

  return model;
}

export async function train(
  network: tf.Sequential,
  gameState: tf.Tensor1D,
  rankedMoves: tf.Tensor1D
) {
  await network.fit(gameState, rankedMoves, { batchSize: 32, epochs: 1 });
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
