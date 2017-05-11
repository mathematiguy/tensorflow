import tensorflow as tf
import tflearn

def build_MNIST_net(n_units, activations=None, learning_rate=0.001, 
                    regularizer=None, weight_decay=0.001, 
                    loss='categorical_crossentropy', optimizer='adam'):
    '''Receives a list of n_units, activations and hyperparameters
       and constructs a Fully Connected Neural Network'''
    
    if activations is None:
        # Set relu as default
        activations = ['relu'] * len(n_units)
        
    # Construct input layer
    net = tflearn.input_data(shape=[None, 784], name="inputs")
    
    # Construct hidden layers
    for i, n_unit in enumerate(n_units):
        net = tflearn.fully_connected(net, 
                                      n_unit, 
                                      activation=activations[i],
                                      name="fully_connected_%d" %(i + 1),
                                      bias_init='truncated_normal',
                                      regularizer=regularizer,
                                      weight_decay=weight_decay)
        
    # End with a softmax layer
    net = tflearn.fully_connected(net, 10,  
                                  activation='softmax',
                                  bias_init='truncated_normal',
                                  name="softmax")
    
    # Perform regression
    net = tflearn.regression(net, 
                             optimizer=optimizer, 
                             loss=loss,
                             learning_rate=learning_rate,
                             name='regression')
    
    return net


def bad_MNIST_net():
    # Build neural network ~92% accuracy
    net = tflearn.input_data(shape=[None, 784])
    net = tflearn.fully_connected(net, 16)
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net)
    return net


def better_MNIST_net():
    # Build neural network ~99% accuracy
    net = tflearn.input_data(shape=[None, 784])
    net = tflearn.fully_connected(net, 1024, activation='relu')
    net = tflearn.fully_connected(net, 512,  activation='relu')
    net = tflearn.fully_connected(net, 512,  activation='relu')
    net = tflearn.fully_connected(net, 512,  activation='relu')
    net = tflearn.fully_connected(net, 128,  activation='relu')
    net = tflearn.fully_connected(net, 64,  activation='relu')
    net = tflearn.fully_connected(net, 32,  activation='relu')
    net = tflearn.fully_connected(net, 10,  activation='softmax')
    net = tflearn.regression(net)
    return net