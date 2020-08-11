'''
Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

'''


#from __future__ import print_function
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf2onnx
#from tf2onnx import loader

# Import MNIST data
#from tensorflow.compat.v1.examples.tutorials.mnist import input_data
import input_data
mnist = input_data.read_data_sets('./data/tensorflow/', one_hot=True)
#mnist = tf.keras.datasets.mnist


def convert_to_onnx(output_graph_def):
    
    #output_graph_def = tf2onnx.loader.freeze_session(sess, output_names=["output:0"])

    tf.reset_default_graph()
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(output_graph_def, name='input')

        onnx_graph = process_tf_graph(tf_graph, 
                                      input_names=["input:0"],
                                      output_names=["output:0"])
        model_proto = onnx_graph.make_model("mnist")
        with open("tensorflow1_mnist.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())


# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [batch_size, num_input], name="input")
Y = tf.placeholder("float", [batch_size, num_classes], name="output")

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    batch_x, batch_y = mnist.test.next_batch(batch_size)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: batch_x,
                                      Y: batch_y}))

    saver = tf.train.Saver()
    save_path = saver.save(sess, './tensorflow/tensorflow_model.ckpt')


    #x = tf.placeholder_with_default(1.0, shape=(), name="input")
    #y = tf.placeholder_with_default(1.0, shape=(), name="output")

    # freeze the graph so that it can be converted to onnx
    # Originally used tf.graph_util.convert_variables_to_constants
    #names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #print(names)
    #output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
    #            sess,
    #            sess.graph.as_graph_def(),  
    #            names)


    #x = tf.placeholder(tf.float32, [batch_size, num_input], name="input")
    #y = tf.placeholder(tf.float32, [batch_size, num_classes], name="output")
    #x = tf.placeholder_with_default(1.0, shape=(), name="input")
    #y = tf.placeholder_with_default(1.0, shape=(), name="output")
    #keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_probability")  # dropout (keep probability
    #_ = tf.identity(x, name="input")
    #_ = tf.identity(y, name="output")

    #x = tf.placeholder(tf.float32, [2, 3], name="input")
    #x_ = tf.add(x, x)
    #_ = tf.identity(x_, name="output")

    #onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=["input:0"], output_names=["output:0"])
    #model_proto = onnx_graph.make_model('MNIST')
    #with open('./tensorflow1_mnist.onnx', 'wb') as f:
    #    f.write(model_proto.SerializeToString())

#convert_to_onnx(output_graph_def)

