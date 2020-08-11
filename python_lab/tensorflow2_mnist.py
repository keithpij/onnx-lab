import tensorflow as tf
import tensorflow_datasets as tfds


# Declare training variables
padding = "SAME"  #@param ['SAME', 'VALID' ]
num_output_classes = 102  #@param {type: "number"}
batch_size = 32  #@param {type: "number"}
learning_rate = 0.001  #@param {type: "number"}


# Fetch the dataset.
dataset_name = 'horses_or_humans'  #@param {type: "string"}

dataset = tfds.load(name=dataset_name, split=tfds.Split.TRAIN)
dataset = dataset.shuffle(1024).batch(batch_size)
print(type(dataset))


# Define CNN operations
leaky_relu_alpha = 0.2 #@param {type: "number"}
dropout_rate = 0.5 #@param {type: "number"}

def conv2d( inputs , filters , stride_size ):
    out = tf.nn.conv2d(inputs, filters, strides=[1, stride_size, stride_size, 1], padding=padding) 
    return tf.nn.leaky_relu(out, alpha=leaky_relu_alpha) 

def maxpool(inputs, pool_size, stride_size):
    return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1] ,padding='VALID' ,strides=[1, stride_size, stride_size, 1])

def dense(inputs, weights):
    x = tf.nn.leaky_relu(tf.matmul(inputs, weights), alpha=leaky_relu_alpha )
    return tf.nn.dropout(x, rate=dropout_rate)


# Initialize CNN weights
output_classes = 3
initializer = tf.initializers.glorot_uniform()
def get_weight( shape , name ):
    return tf.Variable(initializer(shape), name=name, trainable=True, dtype=tf.float32)

shapes = [
    [ 3 , 3 , 3 , 16 ] , 
    [ 3 , 3 , 16 , 16 ] , 
    [ 3 , 3 , 16 , 32 ] , 
    [ 3 , 3 , 32 , 32 ] ,
    [ 3 , 3 , 32 , 64 ] , 
    [ 3 , 3 , 64 , 64 ] ,
    [ 3 , 3 , 64 , 128 ] , 
    [ 3 , 3 , 128 , 128 ] ,
    [ 3 , 3 , 128 , 256 ] , 
    [ 3 , 3 , 256 , 256 ] ,
    [ 3 , 3 , 256 , 512 ] , 
    [ 3 , 3 , 512 , 512 ] ,
    [ 8192 , 3600 ] , 
    [ 3600 , 2400 ] ,
    [ 2400 , 1600 ] , 
    [ 1600 , 800 ] ,
    [ 800 , 64 ] ,
    [ 64 , output_classes ] ,
]

weights = []
for i in range(len(shapes)):
    weights.append(get_weight(shapes[i], 'weight{}'.format(i)))


# Assemble operations
def model( x ) :
    x = tf.cast( x , dtype=tf.float32 )
    c1 = conv2d( x , weights[ 0 ] , stride_size=1 ) 
    c1 = conv2d( c1 , weights[ 1 ] , stride_size=1 ) 
    p1 = maxpool( c1 , pool_size=2 , stride_size=2 )
    
    c2 = conv2d( p1 , weights[ 2 ] , stride_size=1 )
    c2 = conv2d( c2 , weights[ 3 ] , stride_size=1 ) 
    p2 = maxpool( c2 , pool_size=2 , stride_size=2 )
    
    c3 = conv2d( p2 , weights[ 4 ] , stride_size=1 ) 
    c3 = conv2d( c3 , weights[ 5 ] , stride_size=1 ) 
    p3 = maxpool( c3 , pool_size=2 , stride_size=2 )
    
    c4 = conv2d( p3 , weights[ 6 ] , stride_size=1 )
    c4 = conv2d( c4 , weights[ 7 ] , stride_size=1 )
    p4 = maxpool( c4 , pool_size=2 , stride_size=2 )

    c5 = conv2d( p4 , weights[ 8 ] , stride_size=1 )
    c5 = conv2d( c5 , weights[ 9 ] , stride_size=1 )
    p5 = maxpool( c5 , pool_size=2 , stride_size=2 )

    c6 = conv2d( p5 , weights[ 10 ] , stride_size=1 )
    c6 = conv2d( c6 , weights[ 11 ] , stride_size=1 )
    p6 = maxpool( c6 , pool_size=2 , stride_size=2 )

    flatten = tf.reshape( p6 , shape=( tf.shape( p6 )[0] , -1 ))

    d1 = dense( flatten , weights[ 12 ] )
    d2 = dense( d1 , weights[ 13 ] )
    d3 = dense( d2 , weights[ 14 ] )
    d4 = dense( d3 , weights[ 15 ] )
    d5 = dense( d4 , weights[ 16 ] )
    logits = tf.matmul( d5 , weights[ 17 ] )

    return tf.nn.softmax( logits )


# Define the loss function and the optimizer using tf.GradientTape
def loss(pred, target):
    return tf.losses.categorical_crossentropy( target , pred )

optimizer = tf.optimizers.Adam( learning_rate )

def train_step( model, inputs , outputs ):
    with tf.GradientTape() as tape:
        current_loss = loss( model( inputs ), outputs)
    grads = tape.gradient( current_loss , weights )
    optimizer.apply_gradients( zip( grads , weights ) )
    print( tf.reduce_mean( current_loss ) )


# Train the model
num_epochs = 256 #@param {type: "number"}

for e in range( num_epochs ):
    for features in dataset:
        image , label = features[ 'image' ] , features[ 'label' ]
        train_step( model , image , tf.one_hot( label , depth=3 ) )
