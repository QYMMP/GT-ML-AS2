"""
Backprop NN training on Madelon data (Feature selection complete)

"""
import os
import csv
import time
import sys
sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU



# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 50
HIDDEN_LAYER1 = 50
HIDDEN_LAYER2 = 50
HIDDEN_LAYER3 = 50
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1000
OUTFILE = './output/NN_OUTPUT/NN_{}_LOG.csv'


def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) < 0 else 1))
            instances.append(instance)

    return instances


def f1_score(labels, predicted):
    get_count = lambda x: sum([1 for i in x if i is True])

    tp = get_count([predicted[i] == x and x == 1.0 for i, x in enumerate(labels)])
    tn = get_count([predicted[i] == x and x == 0.0 for i, x in enumerate(labels)])
    fp = get_count([predicted[i] == 1.0 and x == 0.0 for i, x in enumerate(labels)])
    fn = get_count([predicted[i] == 0.0 and x == 1.0 for i, x in enumerate(labels)])

    if tp == 0:
        return 0, 0, 0

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return precision, recall, 0.0
    return precision, recall, f1


def errorOnDataSet(network, ds, measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    actuals = []
    predicteds = []

    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted, 1), 0)
        predicteds.append(round(predicted))
        actuals.append(max(min(actual, 1), 0))
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    precision, recall, f1 = f1_score(actuals, predicteds)

    return MSE, acc, f1


def train(oa, network, oaName, training_ints, validation_ints, testing_ints, measure):
    """Train a given network on a set of instances.
    """
    print "\nError results for %s\n---------------------------" % (oaName,)
    times = [0]
    for iteration in xrange(TRAINING_ITERATIONS):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
        times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
            MSE_trg, acc_trg, f1_trg = errorOnDataSet(
                network, training_ints, measure)
            MSE_val, acc_val, f1_val = errorOnDataSet(
                network, validation_ints, measure)
            MSE_tst, acc_tst, f1_tst = errorOnDataSet(
                network, testing_ints, measure)
            txt = '{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                iteration, MSE_trg, MSE_val, MSE_tst, acc_trg, acc_val, acc_tst, f1_trg, f1_val, f1_tst, times[-1])
            print txt
            with open(OUTFILE.format(oaName), 'a+') as f:
                f.write(txt)


def main(CE):
    """Run this experiment"""
    training_ints = initialize_instances('datasets/tic-tac-toe_trg.csv')
    testing_ints = initialize_instances('datasets/tic-tac-toe_test.csv')
    validation_ints = initialize_instances('datasets/tic-tac-toe_val.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    rule = RPROPUpdateRule(0.064, 50, 0.000001)
    oa_name = "SA_{}".format(CE)
    with open(OUTFILE.format(oa_name), 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst',
                                                            'elapsed'))
    classification_network = factory.createClassificationNetwork(
        [INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, HIDDEN_LAYER3, OUTPUT_LAYER], relu)
    nnop = NeuralNetworkOptimizationProblem(
        data_set, classification_network, measure)
    oa = SimulatedAnnealing(1E12, CE, nnop)
    train(oa, classification_network, oa_name, training_ints,
          validation_ints, testing_ints, measure)


if __name__ == "__main__":
    for CE in [0.1, 0.3, 0.5, 0.7, 0.9]:
        main(CE)
