import * as tf from '@tensorflow/tfjs';

const fileName = 'conv-small-distance1.model';
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
  let inputForHorizontal = tf.input({ shape: [4, 4, 1] });
  let inputForVertical = tf.input({ shape: [4, 4, 1] });

  let horizontalConv = tf.layers
    .conv2d({
      inputShape: [4, 4],
      kernelSize: [2, 1],
      strides: 1,
      padding: 'valid',
      filters: 32,
      activation: 'relu',
    })
    .apply(inputForHorizontal);

  let verticalConv = tf.layers
    .conv2d({
      inputShape: [4, 4],
      kernelSize: [1, 2],
      strides: 1,
      padding: 'valid',
      filters: 32,
      activation: 'relu',
    })
    .apply(inputForVertical);

  let horizontalDense = tf.layers
    .dense({ name: 'horizontal', units: 256, activation: 'relu' })
    .apply(horizontalConv);

  let verticalDense = tf.layers
    .dense({ name: 'vertical', units: 256, activation: 'relu' })
    .apply(verticalConv);

  let concatLayer = tf.layers
    .concatenate()
    .apply([tf.layers.flatten().apply(horizontalDense), tf.layers.flatten().apply(verticalDense)]);

  let hidden1 = tf.layers
    .dense({ name: 'hidden-1', units: 1024, activation: 'relu' })
    .apply(concatLayer);

  let hidden2 = tf.layers
    .dense({ name: 'hidden-2', units: 1024, activation: 'relu' })
    .apply(hidden1);

  let output = tf.layers.dense({ name: 'output', units: 4, activation: 'softmax' }).apply(hidden2);

  let model = tf.model({
    name: '2048-move-network',
    inputs: [inputForVertical, inputForHorizontal],
    outputs: output,
  });

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
