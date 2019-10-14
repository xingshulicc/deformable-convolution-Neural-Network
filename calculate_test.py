import tensorflow as tf
from keras.datasets import mnist

batch_size = 1
width = 1
height = 1
channels = 4

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images[0])

indices = tf.constant([
#        [1, 3, 2, 0],
#        [0, 1, 3, 2],
#        [2, 3, 1, 0],
        [3, 1, 2, 0]])

def feature_sort(inputs, indices):
    inputs_batch = inputs[0]
    indices_batch = indices[0]

    print(inputs_batch.eval())
    print(inputs_batch.eval())
    
    return 0
            


with tf.Session() as sess:
    sess.run(feature_sort(inputs, indices))