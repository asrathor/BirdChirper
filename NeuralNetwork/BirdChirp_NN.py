import tensorflow as tf
import numpy as np
import argparse
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def import_test(DataFile, LabelFile):
    # Import Data
    train_data = np.genfromtxt(DataFile, delimiter= ",")
    train_label = np.genfromtxt(LabelFile, delimiter= ",")

    return train_data, train_label

def import_data(DataFile, LabelFile, seed, split_by_train=True, split=10, split_data = True):
    # Import Data
    train_data = np.genfromtxt(DataFile, delimiter= ",")
    train_label = np.genfromtxt(LabelFile, delimiter= ",")

    if(np.ndim(train_data) > 1):
        train_data, train_label = shuffle(train_data, train_label, random_state=seed)
    else:
        train_data = np.matrix(train_data)
        train_label = np.matrix(train_label)

    if split_data == False:
        return train_data, train_label

    test_data = np.zeros((1,train_data.shape[1]))
    test_label = np.zeros((1,train_label.shape[1]))


    labels, count = np.unique(np.argmax(train_label, axis=1),return_counts=True)

    print count
    if (count <= split).any():
        raise AssertionError("Not enough data to split")

    for i in range(train_label.shape[1]):
        index = 0
        confirmed = 0

        while(confirmed < split and index < train_label.shape[0]):
            if(train_label[index][i] == 1):
                test_data = np.append(test_data, [train_data[index]], axis=0)
                test_label = np.append(test_label, [train_label[index]], axis=0)

                train_data = np.delete(train_data, index, axis=0)
                train_label = np.delete(train_label, index, axis=0)
                confirmed += 1
            else:
                index += 1


    test_data = np.delete(test_data, 0, axis=0)
    test_label = np.delete(test_label, 0, axis=0)

    if split_by_train == True:
        return train_data, test_data, train_label, test_label
    else:
        return test_data, train_data, test_label, train_label


    # if(split_data):
    #     return train_test_split(train_data, train_label, test_size=0.1, random_state=seed)
    # else:
    #     return train_data, train_label

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

def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_positive_float(value):
    ivalue = float(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def Train_NN(max_epochs, n_hidden, DataFile, LabelFile, seed, learning_rate, show_accuracy, output_file, splitby, split, train_all):

    # Import Data
    if train_all:
        train_data, train_label = import_data(DataFile, LabelFile, seed, split_data=False)
    else:
        train_data, test_data, train_label, test_label = import_data(DataFile, LabelFile, seed, splitby, split)

    # Define Layer Parameters
    n_input = train_data.shape[1]   # Number of input nodes
    n_classes = train_label.shape[1]   # Number of outcomes

    # Weight initializations
    w_1 = init_weights((n_input, n_hidden))
    w_2 = init_weights((n_hidden, n_classes))

    # Create TF placeholders for features and labels
    x = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    # Define Forward propagation variable
    pred = forwardprop(x, w_1, w_2)
    predict = tf.argmax(pred, axis=1)

    # Define loss and optimizer variables
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=pred))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    #Initialize Variables in TensorFlow graph
    init = tf.global_variables_initializer()

    #Launch TensorFlow Graph
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

    #Train Model
    print "Begin Optimization"

    for epoch in range(max_epochs):
       shuffled_data, shuffled_label = shuffle(train_data, train_label)
       sess.run(train_step, feed_dict={x: shuffled_data, y_: shuffled_label})
       if show_accuracy:
           current_accuracy = np.mean(np.argmax(train_label, axis=1) ==
                                 sess.run(predict, feed_dict={x: train_data, y_: train_label}))*100
           print "Epoch: %i Accuracy: %0.2f%%" % (epoch + 1, current_accuracy)


    print "Optimization Completed\n"

    # Save Weights to CSV file
    with open(output_file, 'w') as weightsfile:
        writer = csv.writer(weightsfile, delimiter=',')
        writer.writerow([n_input, n_hidden, n_classes])
        writer.writerow(sess.run(w_1).flatten())
        writer.writerow(sess.run(w_2).flatten())
    weightsfile.close()

    if train_all == False:
        # Print Neural Network Info
        print "Neural Network Parameters:\n"
        print "Seed:", seed
        print "Epochs: ", max_epochs
        print "Learning Rate:", learning_rate
        print "Hidden Layer Nodes:", n_hidden
        print "Input Nodes:", n_input
        print "Output Nodes:", n_classes

        ## Predicted and true labels used for statistics for test data
        predlabels = sess.run(predict, feed_dict={x: test_data, y_: test_label})[np.newaxis, :]
        truelabels = np.argmax(test_label, axis=1)
        labels, counts = np.unique(truelabels, return_counts=True)

        # Bird labels
        bird_label = np.genfromtxt('BirdLabel.csv', dtype="S", delimiter=",")[np.newaxis, :]

        # Print accuracy for each bird in testing data
        print "\nTesting Data:\n"
        print "Test Accuracy: ", np.mean(np.argmax(test_label, axis=1) == sess.run(predict, feed_dict={x: test_data, y_: test_label}))*100
        temp = 0
        for label in labels:
            correctpred = 0
            indices = np.array(np.where(truelabels == label))
            for index in indices[0]:
                if predlabels[0][index] == label:
                    correctpred += 1
            print "%s Accuracy: %.2f%% with %i samples" % (str(bird_label[0][label]), 100 * correctpred / counts[temp], counts[temp])
            temp += 1

    ## Predicted and true labels used for statistics for training data
    trainpredlabels = sess.run(predict, feed_dict={x: train_data, y_: train_label})[np.newaxis, :]
    traintruelabels = np.argmax(train_label, axis=1)
    trainlabels, traincounts = np.unique(traintruelabels, return_counts=True)

    # Bird labels
    bird_label = np.genfromtxt('BirdLabel.csv', dtype="S", delimiter=",")[np.newaxis, :]

    # Print accuracy for each bird in training data
    print "\nTraining Data:\n"
    print "Training Accuracy: ", np.mean(np.argmax(train_label, axis=1) == sess.run(predict, feed_dict={x: train_data, y_: train_label}))*100
    temp = 0
    for label in trainlabels:
        correctpred = 0
        indices = np.array(np.where(traintruelabels == label))
        for index in indices[0]:
            if trainpredlabels[0][index] == label:
                correctpred += 1
        print "%s Accuracy: %.2f%% with %i samples" % (str(bird_label[0][label]), 100 * correctpred / traincounts[temp], traincounts[temp])
        temp += 1

    sess.close()

def Test_NN(weights, DataFile, LabelFile, seed):
    test_data, test_label = import_data(DataFile, LabelFile, seed, split_data=False)
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

    # Bird labels
    bird_label = np.genfromtxt('BirdLabel.csv', dtype="S", delimiter=",")[np.newaxis, :]


    predlabels = np.matrix(sess.run(predict, feed_dict={x: test_data, y: test_label}))

    # print np.argmax(test_label, axis=1)
    # print "Accuracy:", np.mean(np.argmax(test_label, axis=1) == sess.run(predict, feed_dict={x: test_data, y: test_label}))*100

    for item in predlabels[0]:
        print "Predicted Bird: %s" % (str(bird_label[0][item]))
    sess.close()


#Default Parameters
seed = 100
epochs = 100000
learning_rate = 1e-4
n_hidden = 150
show_accuracy = False
output_file = "weights.csv"
split = 15
splitby = False
train_all = False

#Command Line Parser
parser = argparse.ArgumentParser(description='Neural Network.')
parser.add_argument('data', help='csv file containing data')
parser.add_argument('labels', help='csv file containing corresponding data labels')
parser.add_argument('-o', '--output', help='Name of output csv file containing weights')
parser.add_argument('-sp', '--split', help='Number of data seperated from original data set',type=check_positive_int)
parser.add_argument('-w', '--weights', help='csv file containing weights')
parser.add_argument('-e', '--epoch', help='The number of training epochs', type=check_positive_int)
parser.add_argument('-s', '--seed', help='Random Seed variable',type=check_positive_int)
parser.add_argument('-l', '--learn', help='Learning Rate',type=check_positive_float)
parser.add_argument('-n', '--nodes', help='Nodes in each hidden layer',type=check_positive_int)
parser.add_argument('-a', '--accuracy', help='Print accuracy during each training step', action="store_true")
parser.add_argument('-t', '--splitby', help='Return split data as test data', action="store_true")
parser.add_argument('-d', '--trainall', help='Use all data as training data', action="store_true")
args = parser.parse_args()

#Check for optional parameters
if args.epoch:
    epochs = args.epoch

if args.seed:
    seed = args.seed

if args.learn:
    learning_rate = args.learn

if args.nodes:
    n_hidden = args.nodes

if args.accuracy:
    show_accuracy = args.accuracy

if args.output:
    output_file = args.output + '.csv'

if args.split:
    split = args.split

if args.splitby:
    splitby = args.splitby

if args.trainall:
    train_all = args.trainall

#Run Neural Network
if args.weights:
    Test_NN(args.weights, args.data, args.labels, seed)
else:
    Train_NN(epochs, n_hidden, args.data, args.labels, seed, learning_rate, show_accuracy, output_file, splitby, split, train_all)
