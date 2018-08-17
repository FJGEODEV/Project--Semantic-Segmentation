import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from docutils.languages.af import labels
from jinja2.optimizer import optimize
from jinja2 import optimizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LEARN_RATE = 1e-4

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    print('Start to load vgg model!')
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #vgg_model = 
    tf.saved_model.loader.load(sess, [vgg_tag],vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    print('load_vgg is done!')
    return w1, keep, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    print('Fully convolutional network calculation...')

    # 1x1 convolution for spatial information layer 7
    conv7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes,kernel_size=1,
                                  strides=(1, 1),padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

    # 2c Upsample 
    conv7_2x = tf.layers.conv2d_transpose(conv7_1x1, num_classes,kernel_size=4,
                                  strides=(2, 2),padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

    # 1x1 convolution for spatial information layer4
    conv4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes,kernel_size=1,
                                  strides=(1, 1),padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

    # construct skip layer
    skip_4to7 = tf.add(conv7_2x, conv4_1x1)

    # Upsample the combined layer4 and 1x1 by 2,2
    upsample2x_skip_4to7 = tf.layers.conv2d_transpose(skip_4to7, num_classes,
                                                        kernel_size=4,
                                                        strides=(2, 2),
                                                        padding='same',
                                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                        kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

    # 1x1 convolution for spatial information layer 3
    conv3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                  kernel_initializer=tf.random_normal_initializer(stddev=1e-3))

    # Add layer 3 with upsampled skip4to7 layer
    skip_3 = tf.add(upsample2x_skip_4to7, conv3_1x1)

    # Upsample by (8,8) to obtian to original image size
    output = tf.layers.conv2d_transpose(skip_3, num_classes,
                                        kernel_size=16,
                                        strides=(8, 8),
                                        padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
    
    print('Calculation is done!')
    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    print('start to optimize tensorflow operation!')
    #try to reshape parameters
    #Given tensor, this operation returns a tensor that has the same values as tensor with shape shape
    #create logits as 2D tensor
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    #correct_label = tf.reshape(correct_label, (-1, num_classes))
    
    #Define cost function by cross entropy
    cross_entropy_loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                       ( logits=logits,labels=correct_label))
    
    # As reviewed by Udacity reviewer, here is a place to add loss term
    #
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_constant = 0.01  
    loss = cross_entropy_loss + regularization_constant * sum(regularization_losses)
    
    
    #Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    print('Optimization is done!')
    return logits, train_op, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    print('Training Neural Network!')
    for epoch in range(epochs):
      for image, label in get_batches_fn(batch_size):
        # Training

        _, loss = sess.run([train_op, cross_entropy_loss], 
            feed_dict={input_image:image, correct_label:label, keep_prob:0.5, learning_rate:LEARN_RATE })   
        print('\nTraining Loss = {:.4f}'.format(loss))
    
    
    print('Training is finished!')
    
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 10
    batch_size = 4
    print('Start to run FCN!')
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        
        #create laer:
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        #create loss and optimizer:
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        #saver = tf.train.Saver()
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, 
                 input_image, correct_label, keep_prob, learning_rate)


        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        
        
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
