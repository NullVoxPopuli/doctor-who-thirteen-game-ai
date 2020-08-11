import * as tf from '@tensorflow/tfjs';

const fileName = 'conv-eh.model';
const dataLocation = `indexeddb://${fileName}`;
// const fileInfoLocation = `http://localhost:4200/${fileName}.json`;

export async function useGPU() {
  // try webgpu?
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
  let input = tf.input({ shape: [4, 4, 1] });

  let horizontalConv = tf.layers
    .conv2d({
      inputShape: [4, 4, 1],
      kernelSize: [2, 1],
      strides: 1,
      padding: 'valid',
      filters: 512,
      activation: 'relu',
    })
    .apply(input);

  let verticalConv = tf.layers
    .conv2d({
      inputShape: [4, 4, 1],
      kernelSize: [1, 2],
      strides: 1,
      padding: 'valid',
      filters: 512,
      activation: 'relu',
    })
    .apply(input);

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
    .dense({ name: 'hidden-1', units: 32, activation: 'relu' })
    .apply(concatLayer);

  // let hidden2 = tf.layers
  //   .dense({ name: 'hidden-2', units: 512, activation: 'relu' })
  //   .apply(hidden1);

  // let hidden3 = tf.layers
  //   .dense({ name: 'hidden-3', units: 256, activation: 'relu' })
  //   .apply(hidden2);

  // let hidden4 = tf.layers
  //   .dense({ name: 'hidden-4', units: 128, activation: 'relu' })
  //   .apply(hidden3);

  // let hidden5 = tf.layers.dense({ name: 'hidden-5', units: 64, activation: 'relu' }).apply(hidden4);

  // let hidden6 = tf.layers.dense({ name: 'hidden-6', units: 32, activation: 'relu' }).apply(hidden5);

  let output = tf.layers.dense({ name: 'output', units: 4, activation: 'softmax' }).apply(hidden1);

  let model = tf.model({
    name: '2048-move-network',
    inputs: input,
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
