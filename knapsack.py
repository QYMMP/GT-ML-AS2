from array import array
from itertools import product
from time import clock

import sys
sys.path.append("./ABAGAIL.jar")

import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.prob.MIMIC as MIMIC
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.DiscreteDependencyTree as DiscreteDependencyTree
import java.util.Random as Random



"""
Commandline parameter(s):
    none
"""
random = Random()
maxIters = 200
iterStep = 10
numTrials = 5
# The number of items
NUM_ITEMS = 40
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME

# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
outfile = './output/KNAPSACK/KNAPSACK_{}_{}_LOG.csv'

# Randomized Hill Climbing
for t in range(numTrials):
    fname = outfile.format('RHC', str(t + 1))
    with open(fname, 'w') as f:
        f.write('iterations,fitness,time\n')
    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, iterStep)
    times = [0]

    for i in range(0, maxIters, iterStep):
        start = clock()
        fit.train()
        elapsed = clock() - start
        times.append(times[-1] + elapsed)
        score = ef.value(rhc.getOptimal())
        st = '{},{},{}\n'.format(i, score, times[-1])
        print 'RHC {} '.format(t) + st
        with open(fname, 'a') as f:
            f.write(st)

# Simulated Annealing
for t in range(numTrials):
    # CE = Cooling Exponent
    for CE in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fname = outfile.format('SA{}'.format(CE), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time\n')
        ef = KnapsackEvaluationFunction(
            weights, volumes, KNAPSACK_VOLUME, copies)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        # Starting Temp, Cooling Exponent, Hill Climbing Problem
        sa = SimulatedAnnealing(1E12, CE, hcp)
        fit = FixedIterationTrainer(sa, iterStep)
        times = [0]

        for i in range(0, maxIters, iterStep):
            start = clock()
            fit.train()
            elapsed = clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(sa.getOptimal())
            st = '{},{},{}\n'.format(i, score, times[-1])
            print 'SA {} CE {} '.format(t, CE) + st
            with open(fname, 'a') as f:
                f.write(st)

# Genetic Algorithm
cf = UniformCrossOver()
mf = DiscreteChangeOneMutation(ranges)
pop = 100
for t in range(numTrials):
    for toMate, toMutate in product([50, 30, 10], [50, 30, 10]):
        fname = outfile.format('GA{}_{}_{}'.format(
            pop, toMate, toMutate), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time\n')
        ef = KnapsackEvaluationFunction(
            weights, volumes, KNAPSACK_VOLUME, copies)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, toMate, toMutate, gap)
        fit = FixedIterationTrainer(ga, iterStep)
        times = [0]

        for i in range(0, maxIters, iterStep):
            start = clock()
            fit.train()
            elapsed = clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(ga.getOptimal())
            st = '{},{},{}\n'.format(i, score, times[-1])
            print 'GA {} toMate {} toMutate {} '.format(
                t, toMate, toMutate) + st
            with open(fname, 'a') as f:
                f.write(st)

# MIMIC
ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
# top half
samples = 100
toKeep = 50
for t in range(numTrials):
    for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fname = outfile.format('MIMIC{}_{}_{}'.format(
            samples, toKeep, m), str(t + 1))
        df = DiscreteDependencyTree(m, ranges)
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time\n')
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, toKeep, pop)
        fit = FixedIterationTrainer(mimic, iterStep)
        times = [0]
        for i in range(0, maxIters, iterStep):
            start = clock()
            fit.train()
            elapsed = clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(mimic.getOptimal())
            st = '{},{},{}\n'.format(i, score, times[-1])
            print 'MIMIC {} m {} '.format(t, m) + st
            with open(fname, 'a') as f:
                f.write(st)
