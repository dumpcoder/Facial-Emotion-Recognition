
"""Convolutional Neural Network Estimator for facial emotion, built with tf.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model import *
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)



def main(unused_argv):
    # Load training and eval data
    data1 = np.loadtxt('Dataset/train_image_data.csv', delimiter=',', dtype = np.float32)
    data2 = np.loadtxt('Dataset/test_image_data.csv', delimiter=',', dtype = np.float32)
    train_data = data1[0:,1:]
    train_labels = data1[0:,0:1].flatten()
    eval_data = data2[0:, 1:]
    eval_labels = data2[0:, 0:1].flatten()

    # Cast labels to int32
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_labels = np.asarray(eval_labels, dtype=np.int32)
    
    # Create the Estimator
    emotion_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/users/tomas/Code/Machine_Learning/Project/emotion_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=128,
        num_epochs=124,
        shuffle=True)
    emotion_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = emotion_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
 

if __name__ == "__main__":
    tf.app.run()