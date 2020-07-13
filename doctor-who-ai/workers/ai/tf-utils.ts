import * as tf from '@tensorflow/tfjs';
import ml5 from 'ml5';

// const fileName = 're-improve.model';
// const dataLocation = `downloads://${fileName}`;
// const fileInfoLocation = `/${fileName}.json`;

const fileInfo = {
  model: '/model.json',
  metadata: '/model.meta.json',
  weights: '/model.weights.bin',
};

export async function useGPU() {
  if (ml5.tf.getBackend() !== 'webgl') {
    await ml5.tf.setBackend('webgl');
  }

  await ml5.tf.ready();
}

export async function save(network) {
  await network.save(dataLocation);
}

export async function getNetwork() {
  let network = createNetwork();

  try {
    await network.load(fileInfo);

    return network;
  } catch (e) {
    console.debug(e);

    return network;
  }
}


function createNetwork() {
  return ml5.neuralNetwork({
    debug: true,
    inputs: 16,
    outputs: 4,
    layers: [
      { type: 'dense', units: Math.pow(2, 8), activation: 'relu' },
      { type: 'dense', units: Math.pow(2, 11), activation: 'relu' },
      { type: 'dense', units: Math.pow(2, 10), activation: 'relu' },
      { type: 'dense', units: Math.pow(2, 9), activation: 'relu' },
      { type: 'dense', units: Math.pow(2, 8), activation: 'relu' },
      { type: 'dense', units: Math.pow(2, 6), activation: 'relu' },
      { type: 'dense', units: Math.pow(2, 5), activation: 'relu' },
      { type: 'dense', units: 4, activation: 'softmax' },
    ],
  });
}

const trainingOptions = {
  batchSize: 32,
  epochs: 16
}

export async function train(network) {
  await network.train(trainingOptions);
}
