import os
import random
import numpy as np
import tensorflow as tf
import vggish.vggish_input as vggish_input
import vggish.vggish_slim as vggish_slim
import vggish.vggish_params as vggish_params

def vggish(model_path):

    with tf.Graph().as_default() as default_grapth:
        sess=tf.Session()
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, model_path)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

    return sess, embedding_tensor, embedding_tensor, features_tensor


def extract_vggish_features(sess, input_data, embedding_tensor, features_tensor): 

    extracted_feat = sess.run([embedding_tensor], feed_dict={features_tensor: input_data})
    feature = np.squeeze(np.asarray(extracted_feat))
    return feature


def get_features(paths, verbose=True):


    if verbose:
        print('Loading graph, Opening session')
    sess, embedding_tensor, embedding_tensor, features_tensor  = vggish(model_path="models/vggish_model.ckpt")


    if verbose:
        print('Extracting features')

    features = []
    for p in paths:
        if verbose:
            print('Extracting featues for: {}'.format(p))
        input_data = vggish_input.wavfile_to_examples(p)
        feature = extract_vggish_features(sess, input_data, embedding_tensor, features_tensor)
        featuresa.append(feature)

    for i in features:
        print(i.shape)

    return features

