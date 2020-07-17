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

export function predict(network: tf.Sequential, inputs: tf.Tensor1D) {
  return tf.tidy(() => network.predict(inputs));
}

