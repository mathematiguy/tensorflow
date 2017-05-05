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