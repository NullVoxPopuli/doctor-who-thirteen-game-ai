import * as tf from '@tensorflow/tfjs';

const fileName = 'conv-small.model';
const dataLocation = `indexeddb://${fileName}`;
// const fileInfoLocation = `http://localhost:4200/${fileName}.json`;

export async function useGPU() {
  if (tf.getBackend() !== 'webgl') {
    await tf.setBackend('webgl');
  }

  await tf.ready();

  console.debug(`TensorFlow Backend: `, tf.getBackend());
}

export async function save(network: tf.LayersModel) {
  await network.save(dataLocation);
}

export async function getNetwork(): Promise<tf.LayersModel> {
  let model;

  try {
    model = await tf.loadLayersModel(dataLocation);
  } catch (e) {
    console.debug(e);
  }

  if (!model) {
    model = createNetwork();
  }

  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
  model.summary();

  return model;
}

function createNetwork() {
  let layer = tf.layers.dense;

  let model = tf.sequential({
    name: '2048-move-network',
    layers: [
      tf.layers.conv2d({
        inputShape: [4, 4, 1],
        kernelSize: 2,
        filters: 1,
        padding: 'same',
        // strides: 1,
        activation: 'relu',
      }),
      tf.layers.maxPooling2d({ poolSize: 2, strides: 1 }),
      tf.layers.flatten(),
      layer({ name: 'hidden-0', units: Math.pow(2, 5), activation: 'relu' }),
      // layer({ name: 'hidden-1', units: Math.pow(2, 11), activation: 'relu' }),
      // layer({ name: 'hidden-2', units: Math.pow(2, 9), activation: 'relu' }),
      layer({ name: 'hidden-3', units: Math.pow(2, 8), activation: 'relu' }),
      layer({ name: 'hidden-4', units: Math.pow(2, 7), activation: 'relu' }),
      // layer({ name: 'hidden-5', units: Math.pow(2, 6), activation: 'relu' }),
      // layer({ name: 'hidden-6', units: Math.pow(2, 5), activation: 'relu' }),
      layer({ name: 'output', units: 4, activation: 'softmax' }),
    ],
  });

  return model;
}
