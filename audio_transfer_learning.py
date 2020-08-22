import os
import random
import numpy as np
import tensorflow as tf
import vggish.vggish_input as vggish_input
import vggish.vggish_slim as vggish_slim
import vggish.vggish_params as vggish_params
from utils import wavefile_to_waveform, softmax
from scipy import spatial
import matplotlib.pyplot as plt
import datetime
plt.style.use('ggplot')


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

if __name__ == '__main__':


    print('Loading graph, Opening session')
    sess, embedding_tensor, embedding_tensor, features_tensor  = vggish(model_path="models/vggish_model.ckpt")

    print('Extracting pattern features')
    pat_input_data = vggish_input.wavfile_to_examples("tracks/pattern1.wav")
    pat_feature = extract_vggish_features(sess, pat_input_data, embedding_tensor, features_tensor)


    print("Extracting reference features")
    ref_input_data = vggish_input.wavfile_to_examples("tracks/reference.wav")
    ref_feature = extract_vggish_features(sess, ref_input_data, embedding_tensor, features_tensor)


    def print_feature_shapes(name, shape):
        print(f"{name} feature shape: {shape}")

    print_feature_shapes("pattern", pat_feature.shape)
    print_feature_shapes("reference", ref_feature.shape)

    scores = []
    i = 0

    flattend_pat_feature = pat_feature.flatten()
    N = ref_feature.shape[0]
    P = pat_feature.shape[0]
    hisotry = []
    while i+P <= N:

        result = 1 - spatial.distance.cosine(flattend_pat_feature, ref_feature[i:i+P].flatten())
        time = i*96*10/60000
        minute = i*96*10//60000
        second = (time - minute)*60
        print(f"Step: {i} Minute: {minute} Second: {second}   Cosine Dist: {result}")
        i+=1
        hisotry.append((datetime.time(minute=minute, second=int(second), microsecond=int(1000*(second-int(second)))), result))

    # Create figure and plot space
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.tick_params(axis='both', which='major', labelsize=10) 

    # Add x-axis and y-axis
    times = [t for i,(t, r) in enumerate(hisotry)]
    scores = softmax(np.array([r for t, r in hisotry]), 0.05)
    
    ax.plot(np.arange(len(scores)), scores)
    ax.set_xticks(np.arange(len(scores))[::20])
    ax.set_xticklabels(times[::20])

    ax.set(xlabel="Time",title="Audio Snipet Detection")
    ax.grid()
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im, cax=cax)

    plt.savefig('pattern1-result.png', dpi=150)











