'''
This is a Tensorflow 1.0 Neural Network that creates a model for predicting the
MNIST dataset.

'''

#from __future__ import print_function
#import tensorflow as tf
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tf2onnx
#from tf2onnx import loader
#from tensorflow.compat.v1.examples.tutorials.mnist import input_data
import input_data


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
X = tf.compat.v1.placeholder("float", [batch_size, num_input], name="input")
Y = tf.compat.v1.placeholder("float", [batch_size, num_classes], name="output")


def load_data():
    mnist = input_data.read_data_sets('./data/tensorflow/', one_hot=True)
    return mnist.train, mnist.test


# Create model
def build_model(x):
    weights = {
        'h1': tf.Variable(tf.random.normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random.normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden_1])),
        'b2': tf.Variable(tf.random.normal([n_hidden_2])),
        'out': tf.Variable(tf.random.normal([num_classes]))
    }

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    model = tf.nn.softmax(out_layer)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(
        logits=out_layer, labels=tf.stop_gradient(Y)))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    prediction_op = tf.equal(tf.argmax(input=model, axis=1), tf.argmax(input=Y, axis=1))
    accuracy_op = tf.reduce_mean(input_tensor=tf.cast(prediction_op, tf.float32))

    return model, loss_op, train_op, prediction_op, accuracy_op

# Construct model
#logits = build_model(X)
#prediction = tf.nn.softmax(logits)
#model, loss_op, train_op = build_model(X)


def train_model(train, test, model, loss_op, train_op):

    # Evaluate model

    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Start training
    #with tf.Session() as sess:
    sess = tf.compat.v1.Session()

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_train_x, batch_train_y = train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_train_x, Y: batch_train_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy_op], feed_dict={X: batch_train_x,
                                                                Y: batch_train_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                "{:.4f}".format(loss) + ", Training Accuracy= " + \
                "{:.3f}".format(acc))

    return sess


def evaluate_model(test, sess, accuracy_op):

    batch_test_x, batch_test_y = test.next_batch(batch_size)
    accuracy = sess.run(accuracy_op, feed_dict={X: batch_test_x,
                                                Y: batch_test_y})
    return accuracy


def predict(sess, prediction_op, images, labels):
    y = sess.run(prediction_op, feed_dict={X: images})
    return y


def save_framework_model(sess):
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(sess, './tensorflow/tensorflow_model.ckpt')


def convert_to_onnx(output_graph_def):
    
    #output_graph_def = tf2onnx.loader.freeze_session(sess, output_names=["output:0"])

    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(output_graph_def, name='input')

        onnx_graph = process_tf_graph(tf_graph, 
                                      input_names=["input:0"],
                                      output_names=["output:0"])
        model_proto = onnx_graph.make_model("mnist")
        with open("tensorflow1_mnist.onnx", "wb") as f:
            f.write(model_proto.SerializeToString())


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


if __name__ == '__main__':
    train, test = load_data()

    model, loss_op, train_op, prediction_op, accuracy_op = build_model(X)
    sess = train_model(train, test, model, loss_op, train_op)
    accuracy = evaluate_model(test, sess, accuracy_op)
    print('Testing Accuracy: ' + str(accuracy))
    y = predict(sess, model, train.images[0:128], train.labels[0:128])
    print(y)
