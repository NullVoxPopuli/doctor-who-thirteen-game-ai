import * as tf from '@tensorflow/tfjs';

const fileName = 'dense-sm-distance1.model';
const dataLocation = `indexeddb://${fileName}`;
// const fileInfoLocation = `http://localhost:4200/${fileName}.json`;

export async function useGPU() {
  if (tf.getBackend() !== 'webgl') {
    await tf.setBackend('webgl');

    console.debug(`TensorFlow Backend: `, tf.getBackend());
  }

  await tf.ready();
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
  let input = tf.input({ shape: [4, 4] });

  let model = tf.sequential({ name: '2048-move-network' });

  let horizontalConv = tf.layers.conv2d({
    inputShape: [4, 4],
    kernelSize: [2, 1],
    strides: 1,
    padding: 'valid',
    filters: 512,
    activation: 'relu',
  });
  let verticalConv = tf.layers.conv2d({
    inputShape: [4, 4],
    kernelSize: [1, 2],
    strides: 1,
    padding: 'valid',
    filters: 512,
    activation: 'relu',
  });

  model.add(concatLayer);
  model.add(tf.layers.flatten())


  let output = tf.layers.dense({ name: 'output', units: 4, activation: 'softmax' });

  model.add(output);


  // let layer = tf.layers.dense;
  // let model = tf.sequential({
  //   name: '2048-move-network',
  //   layers: [
  //     layer({
  //       name: 'input-receive',
  //       units: Math.pow(2, 10),
  //       activation: 'relu',
  //       inputShape: [16],
  //     }),
  //     // tf.layers.conv2d({
  //     //   inputShape: [4, 4, 1],
  //     //   kernelSize: 2,
  //     //   filters: 3,
  //     //   padding: 'same',
  //     //   // strides: 1,
  //     //   activation: 'relu',
  //     // }),
  //     // tf.layers.maxPooling2d({ poolSize: 2, strides: 1 }),
  //     // tf.layers.flatten(),
  //     // layer({ name: 'hidden-0', units: Math.pow(2, 5), activation: 'relu' }),
  //     layer({ name: 'hidden-1', units: Math.pow(2, 11), activation: 'relu' }),
  //     layer({ name: 'hidden-2', units: Math.pow(2, 9), activation: 'relu' }),
  //     // layer({ name: 'hidden-3', units: Math.pow(2, 8), activation: 'relu' }),
  //     // layer({ name: 'hidden-4', units: Math.pow(2, 7), activation: 'relu' }),
  //     // layer({ name: 'hidden-5', units: Math.pow(2, 6), activation: 'relu' }),
  //     // layer({ name: 'hidden-6', units: Math.pow(2, 5), activation: 'relu' }),
  //     layer({ name: 'output', units: 4, activation: 'softmax' }),
  //   ],
  // });

  return model;
}
