import sys
import math
from operator import itemgetter, attrgetter
from decimal import *

def generate_model(train_data_filename, model_filename, prior_delta, cond_delta):
    classLabels = {}
    featureCounts = {}
    instances = {}
    allFeatures = set()
    featuresInLabel = {}

    train_data = open(train_data_filename, 'r')
    for line in train_data:
        lineArray = line.split()
        instanceName = lineArray[0]
        label = lineArray[1]
    
        instances[instanceName] = {}
        instances[instanceName]['classLabel'] = label
        
        if not (label in classLabels):
            classLabels[label] = {}
            classLabels[label]['numInstances'] = 1
            featuresInLabel[label] = 0
        else:
            classLabels[label]['numInstances'] += 1
    
        features = lineArray[2::2] # every other word in line starting with third
        values = lineArray[3::2] # every other word in line starting with fourth
        
        for f, v in zip(features, values):
            allFeatures.add(f)
            featuresInLabel[label] += 1;
            # ignore v in the case of a binomial trial
            if not(f in classLabels[label]):
                classLabels[label][f] = 1
            else:
                classLabels[label][f] += 1
            if not (f in featureCounts):
                featureCounts[f] = 1  # would be v in multinomial
            else:
                featureCounts[f] += 1
            if not (f in instances[instanceName]):
                instances[instanceName][f] = 1
            else:
                instances[instanceName][f] += 1
    
    train_data.close()
    
    # cache logs for faster calculations
    logs = {}
    biggestLog = sum(featuresInLabel.values())
    if prior_delta >= cond_delta: 
        biggestLog += int(prior_delta*2)
    else:
        biggestLog += int(cond_delta*2)
    print biggestLog
    for i in range(1,biggestLog+1):
        logs[i] = math.log(i)
    
    #store probabilities
    labelProbs = {}
    labelLogProbs = {}
    featureProbs = {}
    featureLogProbs = {}
    for label in classLabels:
        # get label probabilities
        lc = float(len(classLabels))
        labelProbs[label] = (classLabels[label]['numInstances'] + prior_delta) \
        / (len(instances)+(prior_delta*lc)) 
        labelLogProbs[label] = logs[classLabels[label]['numInstances'] + \
        prior_delta] - logs[len(instances) + (prior_delta*lc)]
        # get P(f|c)
        featureProbs[label] = {}
        featureLogProbs[label] = {}
        for feature in classLabels[label]:
            if feature in classLabels[label]:
                featureProbs[label][feature] = (classLabels[label][feature] + \
                cond_delta) / (featuresInLabel[label] + (cond_delta*2.0))
                featureLogProbs[label][feature] = logs[classLabels[label][feature] \
                + cond_delta] - logs[featuresInLabel[label] + cond_delta*2]
#            else:
#                featureProbs[label][feature] = cond_delta / (featuresInLabel[label]\
#                + (cond_delta*2.0))
#                featureLogProbs[label][feature] = logs[cond_delta] - \
#                logs[featuresInLabel[label] + cond_delta*2]
        
    
    # now print the probabilities we got to the model file
    model_file = open(model_filename, 'w')
    
    # prior probs
    model_file.write("%%%%% prior prob P(c) %%%%%\n")
    for label in classLabels:
        model_file.write(str(label) + "\t")
        model_file.write(str(labelProbs[label]) + "\t")
        model_file.write(str(labelLogProbs[label])+ "\n")
    #conditional probs
    model_file.write("%%%%% conditional prob P(f|c) %%%%%\n")
    for label in classLabels:
        model_file.write("%%%%% conditional prob P(f|c) c=" + label + "  %%%%%\n")
        for feature in featureProbs[label]:
            model_file.write(feature + "\t" + label + "\t")
            model_file.write(str(featureProbs[label][feature]) + "\t")
            model_file.write(str(featureLogProbs[label][feature]) + "\n")
    return [instances, allFeatures, labelLogProbs, featureLogProbs]

def classify(instances, classLogProbs, featureLogProbs):
    output = {}
    for instance in instances:				
        output[instance] = {}
        argmax = ""
        maxProb = 0
        for label in classLogProbs:
            #print featureLogProbs
            # P(c)
            classLogProb = classLogProbs[label]

            # get P(f|c)
            classProduct = 1
            ## can't tell... should this be a loop over every feature,
            ## every feature in the class,
            ## or just the features in the document
            ## waiting on a thread in gopost about it, for now do every
            ## feature ever seen in this class label
            for feature in featureLogProbs[label]:
                # skip the utility variables
                if (feature == 'classLabel') or (feature == 'numInstances'):
                    continue 

                # skip words we haven't seen before
                if not (feature in featureLogProbs[label]):
                    continue

                # use optimization on slide 38 of NB slides
                # using addition and subtraction instead of multiplication
                # and division because we are working with logs of probs

                # get prob for words not found in the document 
                featurelogprob = featureLogProbs[label][feature]
                prob2 = 1 - featurelogprob
                # get prob for words found in the document
                prob1 = 0 #just initializing
                prob1 =  featurelogprob - \
                (1 - featurelogprob)
                classProduct += prob1 + prob2
            docClassLogProb = classLogProb + classProduct
            output[instance][label] = docClassLogProb

            # use less than because we're dealing with log probs
            if (docClassLogProb < maxProb):
                argmax = label
                maxProb = docClassLogProb
        output[instance]['winner'] = argmax
        output[instance]['winnerLogProb'] = maxProb
    return output

def print_sys(output, sys_file, labels):
    for instance in output:
        sys_file.write(instance + " ")
        sys_file.write(output[instance]['winner'])
        for label in labels:
            sys_file.write(" " + label)
            logprob = output[instance][label]
            prob = str(Decimal(10)**Decimal(str(logprob)))
            sys_file.write(" " +prob)
        sys_file.write("\n")

if (len(sys.argv) < 7):
    print "Not enough args."
    sys.exit(1)

# The format is: build_NB1.sh training_data test_data prior_delta cond_prob_delta model_file sys_output > acc_file
# python build_NB1.py examples/train.vectors.txt examples/test.vectors.txt 0 0.1 examples/model1 examples/sys1

train_data_filename = sys.argv[1]
test_data_filename = sys.argv[2]
prior_delta = float(sys.argv[3])
cond_delta = float(sys.argv[4])
model_filename = sys.argv[5]
sys_filename = sys.argv[6]


data_list = generate_model(train_data_filename, model_filename, prior_delta,\
cond_delta)

train_results = classify(data_list[0], data_list[2], data_list[3])
sys_file = open(sys_filename, 'w')
sys_file.write("%%%%% training data:\n")
print_sys(train_results, sys_file, data_list[2])

#print train_results
