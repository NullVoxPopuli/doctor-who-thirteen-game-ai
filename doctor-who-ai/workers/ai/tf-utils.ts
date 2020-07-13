import * as tf from '@tensorflow/tfjs';

export async function useGPU() {
  if (tf.getBackend() !== 'webgl') {
    await tf.setBackend('webgl');
  }

  await tf.ready();
}
