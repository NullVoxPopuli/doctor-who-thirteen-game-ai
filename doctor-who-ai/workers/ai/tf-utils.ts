import tf, { model } from '@tensorflow/tfjs';

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
  let model;

  try {
    // model = await tf.loadLayersModel(fileInfoLocation);
  } catch (e) {
    console.debug(e);
  }

  if (!model) {
    model = createNetwork();
  }

  model.summary();

  return model;
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

  let layer = tf.layers.dense;

  let model = tf.sequential({
    name: '2048-move-network',
    layers: [
      layer({ name: 'input', units: Math.pow(2, 9), inputShape: [16], activation: 'relu' }),
      layer({ name: 'hidden-1', units: Math.pow(2, 5), activation: 'relu' }),
      layer({ name: 'hidden-2', units: Math.pow(2, 6), activation: 'relu' }),
      layer({ name: 'hidden-3', units: Math.pow(2, 7), activation: 'relu' }),
      // layer({ units: Math.pow(2, 8), activation: 'relu' }),
      // layer({ units: Math.pow(2, 6), activation: 'relu' }),
      // layer({ units: Math.pow(2, 5), activation: 'relu' }),
      layer({ name: 'output', units: 4, activation: 'softmax' }),
    ],
  });

  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

  return model;
}
