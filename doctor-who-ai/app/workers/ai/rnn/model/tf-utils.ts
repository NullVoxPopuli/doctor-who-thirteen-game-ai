import * as tf from '@tensorflow/tfjs';

const fileName = 'dqn-sm.model';
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

export function createNetwork() {
  let input = tf.input({
    // x, y, channels?
    shape: [4, 4, 1],
  });

  let horizontalConv = tf.layers
    .conv2d({
      kernelSize: [2, 1],
      strides: 1,
      filters: 256,
      activation: 'relu',
      kernelInitializer: 'ones',
    })
    .apply(input);

  let verticalConv = tf.layers
    .conv2d({
      kernelSize: [1, 2],
      strides: 1,
      filters: 256,
      kernelInitializer: 'ones',
      activation: 'relu',
    })
    .apply(input);

  let horizontalDense = tf.layers
    .dense({
      name: 'horizontal',
      units: 128,
      activation: 'relu',
      kernelInitializer: 'ones',
    })
    .apply(horizontalConv);

  let verticalDense = tf.layers
    .dense({ name: 'vertical', units: 128, activation: 'relu', kernelInitializer: 'ones' })
    .apply(verticalConv);

  let concatLayer = tf.layers
    .concatenate()
    .apply([tf.layers.flatten().apply(horizontalDense), tf.layers.flatten().apply(verticalDense)]);

  let hidden1 = tf.layers
    .dense({ name: 'hidden-1', units: 16, activation: 'relu' })
    .apply(concatLayer);

  //   let hidden2 = tf.layers
  //     .dense({ name: 'hidden-2', units: 256, activation: 'relu' })
  //     .apply(hidden1);

  //   let hidden3 = tf.layers
  //     .dense({ name: 'hidden-3', units: 256, activation: 'relu' })
  //     .apply(hidden2);

  //   let hidden4 = tf.layers
  //     .dense({ name: 'hidden-4', units: 128, activation: 'relu' })
  //     .apply(hidden3);

  // let hidden5 = tf.layers.dense({ name: 'hidden-5', units: 64, activation: 'relu' }).apply(hidden4);

  // let hidden6 = tf.layers.dense({ name: 'hidden-6', units: 32, activation: 'relu' }).apply(hidden5);

  let output = tf.layers.dense({ name: 'output', units: 4, activation: 'softmax' }).apply(hidden1);

  let model = tf.model({
    name: '2048-move-network',
    inputs: input,
    outputs: output,
  });

  return model;
}

/**
 * from tfjs-examples/snake-dqn
 *
 * Copy the weights from a source deep-Q network to another.
 *
 * @param {tf.LayersModel} destNetwork The destination network of weight
 *   copying.
 * @param {tf.LayersModel} srcNetwork The source network for weight copying.
 */
export function copyWeights(destNetwork: tf.LayersModel, srcNetwork: tf.LayersModel) {
  // https://github.com/tensorflow/tfjs/issues/1807:
  // Weight orders are inconsistent when the trainable attribute doesn't
  // match between two `LayersModel`s. The following is a workaround.
  // TODO(cais): Remove the workaround once the underlying issue is fixed.
  let originalDestNetworkTrainable;

  if (destNetwork.trainable !== srcNetwork.trainable) {
    originalDestNetworkTrainable = destNetwork.trainable;
    destNetwork.trainable = srcNetwork.trainable;
  }

  destNetwork.setWeights(srcNetwork.getWeights());

  // Weight orders are inconsistent when the trainable attribute doesn't
  // match between two `LayersModel`s. The following is a workaround.
  // TODO(cais): Remove the workaround once the underlying issue is fixed.
  // `originalDestNetworkTrainable` is null if and only if the `trainable`
  // properties of the two LayersModel instances are the same to begin
  // with, in which case nothing needs to be done below.
  if (originalDestNetworkTrainable != null) {
    destNetwork.trainable = originalDestNetworkTrainable;
  }
}
