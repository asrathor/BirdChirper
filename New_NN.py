import tensorflow as tf
import numpy as np
import argparse
import csv
from sklearn.model_selection import train_test_split

def import_data(DataFile, LabelFile, random_seed):
    # Import Data
    train_data = np.genfromtxt(DataFile, delimiter= ",")
    train_label = np.genfromtxt(LabelFile, delimiter= ",")

    return train_test_split(train_data, train_label, test_size=0.33, random_state=random_seed)

def import_test_data(DataFile, LabelFile):
    # Import Data
    test_data = np.genfromtxt(DataFile, delimiter= ",")
    test_label = np.genfromtxt(LabelFile, delimiter= ",")
    return test_data, test_label

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def init_weights(shape):
    # Weight initialization
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    # Forward-propagation
    h_1 = tf.nn.sigmoid(tf.matmul(X, w_1))
    y_out = tf.nn.softmax(tf.matmul(h_1, w_2))
    return y_out

def init_existing_weights(file):
    with open(str(file), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = [r for r in reader]
        x_size = int(rows[0][0])
        h_size = int(rows[0][1])
        y_size = int(rows[0][2])
        w_1 = np.array(rows[1])
        w_1 = np.reshape(w_1.astype(np.float32), (x_size, h_size))
        w_2 = np.array(rows[2])
        w_2 = np.reshape(w_2.astype(np.float32), (h_size, y_size))
    csvfile.close()
    return x_size, h_size, y_size, tf.Variable(w_1), tf.Variable(w_2)

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_positive_float(value):
    ivalue = float(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def Train_NN(max_epochs, n_hidden, DataFile, LabelFile, random_seed, learning_rate, show_accuracy, output_file):

    # Import Data
    train_data, test_data, train_label, test_label = import_data(DataFile, LabelFile, random_seed)

    # Layer Parameters
    n_input = train_data.shape[1]   # Number of input nodes
    n_classes = train_label.shape[1]   # Number of outcomes

    # Weight initializations
    w_1 = init_weights((n_input, n_hidden))
    w_2 = init_weights((n_hidden, n_classes))

    # Create the model
    x = tf.placeholder(tf.float32, [None, n_input])

    # Forward propagation
    pred = forwardprop(x, w_1, w_2)
    predict = tf.argmax(pred, axis=1)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #Initialize Variables
    init = tf.global_variables_initializer()

    #Launch Graph
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    #Train
    print("Begin Optimization")

    for epoch in range(max_epochs):
       shuffled_data, shuffled_label = shuffle_in_unison(train_data, train_label)
       sess.run(train_step, feed_dict={x: shuffled_data, y_: shuffled_label})
       if show_accuracy:
           current_accuracy = np.mean(np.argmax(test_label, axis=1) ==
                                 sess.run(predict, feed_dict={x: test_data, y_: test_label})*100)
           print("Epoch: ", epoch + 1, "Accuracy: ", current_accuracy)

    print("Optimization Completed\n")

    # Save Weights
    with open(output_file, 'w') as weightsfile:
        writer = csv.writer(weightsfile, delimiter=',')
        writer.writerow([n_input, n_hidden, n_classes])
        writer.writerow(sess.run(w_1).flatten())
        writer.writerow(sess.run(w_2).flatten())
    weightsfile.close()

    print("Neural Network Parameters:\n")
    print("Seed:", random_seed)
    print("Epochs: ", max_epochs)
    print("Learning Rate:", learning_rate)
    print("Hidden Layer Nodes:", n_hidden)
    print("Input Nodes:", n_input)
    print("Output Nodes:", n_classes)

    ## Predicted and true labels used for statistics for test data
    predlabels = sess.run(predict, feed_dict={x: test_data, y_: test_label})[np.newaxis, :]
    truelabels = np.argmax(test_label, axis=1)
    labels, counts = np.unique(truelabels, return_counts=True)

    ## Predicted and true labels used for statistics for training data
    trainpredlabels = sess.run(predict, feed_dict={x: train_data, y_: train_label})[np.newaxis, :]
    traintruelabels = np.argmax(train_label, axis=1)
    trainlabels, traincounts = np.unique(traintruelabels, return_counts=True)

    # Bird labels
    bird_label = np.genfromtxt('BirdLabel.csv', dtype="S", delimiter=",")[np.newaxis, :]

    # Print accuracy for each bird in training data
    print("\nTraining Data:\n")
    print("Training Accuracy: ", np.mean(np.argmax(train_label, axis=1) == sess.run(predict, feed_dict={x: train_data, y_: train_label})))
    for label in trainlabels:
        correctpred = 0
        indices = np.array(np.where(traintruelabels == label))
        for index in indices[0]:
            if trainpredlabels[0][index] == label:
                correctpred += 1
        print("%s Accuracy: %.2f%% with %i samples" % (str(bird_label[0][label]), 100 * correctpred / traincounts[label], traincounts[label]))

    # Print accuracy for each bird in testing data
    print("\nTesting Data:\n")
    print("Test Accuracy: ", np.mean(np.argmax(test_label, axis=1) == sess.run(predict, feed_dict={x: test_data, y_: test_label}))*100)
    for label in labels:
        correctpred = 0
        indices = np.array(np.where(truelabels == label))
        for index in indices[0]:
            if predlabels[0][index] == label:
                correctpred += 1
        print("%s Accuracy: %.2f%% with %i samples" % (str(bird_label[0][label]), 100 * correctpred / counts[label], counts[label])*100)

    sess.close()

def Test_NN(weights, DataFile, LabelFile):
    test_data, test_label = import_test_data(DataFile, LabelFile)
    n_input, n_hidden, n_classes, w_1, w_2 = init_existing_weights(weights)

    # Create the model
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Forward propagation
    pred = forwardprop(x, w_1, w_2)
    predict = tf.argmax(pred, axis=1)

    #Initialize Variables
    init = tf.global_variables_initializer()

    #Launch Graph
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    print("Accuracy: ", np.mean(np.argmax(test_label, axis=1) == sess.run(predict, feed_dict={x: test_data, y: test_label}))*100)
    sess.close()


#Default Parameters
random_seed = 100
epochs = 1000
learning_rate = 1e-4
n_hidden = 256
show_accuracy = False
output_file = "weights.csv"

#Command Line Parser
parser = argparse.ArgumentParser(description='Neural Network.')
parser.add_argument('data', help='csv file containing data')
parser.add_argument('labels', help='csv file containing corresponding data labels')
parser.add_argument('-w', '--weights', help='csv file containing weights')
parser.add_argument('-e', '--epoch', help='The number of training epochs', type=check_positive)
parser.add_argument('-s', '--seed', help='Random Seed variable',type=check_positive)
parser.add_argument('-l', '--learn', help='Learning Rate',type=check_positive_float)
parser.add_argument('-n', '--nodes', help='Nodes in each hidden layer',type=check_positive)
parser.add_argument('-a', '--accuracy', help='Print accuracy during each training step',type=bool)
parser.add_argument('-o', '--output', help='Name of output csv file containing weights')
args = parser.parse_args()

#Check for optional parameters
if args.epoch:
    epochs = args.epoch

if args.seed:
    random_seed = args.seed

if args.learn:
    learning_rate = args.learn

if args.nodes:
    n_hidden = args.nodes

if args.accuracy:
    show_accuracy = args.accuracy

if args.output:
    output_file = args.output + '.csv'

#Run Neural Network
if args.weights:
    Test_NN(args.weights, args.data, args.labels)
else:
    Train_NN(epochs, n_hidden, args.data, args.labels, random_seed, learning_rate, show_accuracy, output_file)
