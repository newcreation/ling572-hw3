import sys
import math
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
    # logs = {}
    # biggestLog = sum(featuresInLabel.values())
    # if prior_delta*len(classLabels.keys()) >= (cond_delta*2): 
    #     biggestLog += int(math.ceil(prior_delta)*len(classLabels.keys()))
    # else:
    #     biggestLog += int(math.ceil(cond_delta)*2)
    # for i in range(1,biggestLog+1):
    #     logs[i] = math.log(i)
    
    #store probabilities
    labelProbs = {}
    labelLogProbs = {}
    featureProbs = {}
    featureLogProbs = {}
    featureNegLogProbs = {} # cache these for the classify function
    for label in classLabels:
        # get label probabilities
        lc = float(len(classLabels))
        labelProbs[label] = (classLabels[label]['numInstances'] + prior_delta) \
        / (len(instances)+(prior_delta*lc)) 
        labelLogProbs[label] = math.log(classLabels[label]['numInstances'] + \
        prior_delta) - math.log(len(instances) + (prior_delta*lc))
        # get P(f|c)
        featureProbs[label] = {}
        featureLogProbs[label] = {}
        featureNegLogProbs[label] = {}
        for feature in classLabels[label]:
            if feature in classLabels[label]:
                featureProbs[label][feature] = (classLabels[label][feature] + \
                cond_delta) / (featuresInLabel[label] + (cond_delta*2.0))
                featureLogProbs[label][feature] = math.log(classLabels[label][feature] \
                + cond_delta) - math.log(featuresInLabel[label] + cond_delta*2)
                featureNegLogProbs[label][feature] =\
                math.log(1-featureProbs[label][feature])
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
    return [instances, allFeatures, labelLogProbs, featureLogProbs,\
    featuresInLabel, featureNegLogProbs]

def classify(instances, classLogProbs, featureLogProbs, featureNegLogProbs):
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

                prob = 0
                # skip words we haven't seen before
                if (feature in instances[instance]):
                    prob = featureLogProbs[label][feature] -\
                    featureNegLogProbs[label][feature]
                else:
                    prob = featureNegLogProbs[label][feature]

                # use optimization on slide 38 of NB slides
                # using addition and subtraction instead of multiplication
                # and division because we are working with logs of probs

                # get prob for words not found in the document 
                #featurelogprob = featureLogProbs[label][feature]
                #featureneglogprob = featureNegLogProbs[label][feature]
                # get prob for words found in the document
                #prob1 =  featurelogprob - featureneglogprob 
                classProduct += prob
            docClassLogProb = classLogProb + classProduct
            output[instance][label] = docClassLogProb

            # use less than because we're dealing with log probs
            if (docClassLogProb < maxProb):
                argmax = label
                maxProb = docClassLogProb
        output[instance]['winner'] = argmax
        output[instance]['winnerLogProb'] = maxProb
    return output

def print_sys(output, sys_file, labels, instances):
    for instance in output:
        sys_file.write(instance + " ")
        sys_file.write(instances[instance]['classLabel']+ " ")
        #sys_file.write(output[instance]['winner']+ " ")
        for label in labels:
            sys_file.write(" " + label)
            logprob = output[instance][label]
            prob = str(Decimal(10)**Decimal(str(logprob)))
            sys_file.write(" " +prob)
        sys_file.write("\n")
        
def print_acc(output, instances, labels):
    #print labels.keys()
    #print instances.keys()
    #print labels.keys()[0]
    # print output
    # print instances
    print "Confusion matrix for the training data:"
    print "row is the truth, column is the system output\n"
    toprow = "\t\t"
    for label in labels:
        toprow+=label+"\t\t"
    #toprow+="|total"
    print toprow

    cells = {}
    numRight = 0
    for actuallabel in labels:
        cells[actuallabel] = {}
        for expectedlabel in labels:
            cells[actuallabel][expectedlabel] = 0

    for instance in output:
        cells[instances[instance]['classLabel']][output[instance]['winner']] +=1
        if instances[instance]['classLabel'] == output[instance]['winner']:
            numRight += 1.0

    for actuallabel in labels: #loop for row
        #curr_row = str(index_row)+" "
        curr_row = actuallabel + "\t\t"

        for expectedlabel in labels: #loop for column
            curr_row += str(cells[actuallabel][expectedlabel])+"\t\t"
#but only if it's the index representing the winner
        print curr_row
    
    # confusion_matrix = [][]
    # for output in output.keys():
    # 	    print output
    return numRight/len(instances)   

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

# return [instances, allFeatures, labelLogProbs, featureLogProbs,\
# featuresInLabel]

train_results = classify(data_list[0], data_list[2], data_list[3], data_list[5])
sys_file = open(sys_filename, 'w')
sys_file.write("%%%%% training data:\n")
print_sys(train_results, sys_file, data_list[2], data_list[0])

print "class_num=", len(data_list[2]), ", feat_num=", len(data_list[1])
train_acc = print_acc(train_results,data_list[0], data_list[2])
print "Training accuracy =", train_acc
print "\n\n"

test_data = open(test_data_filename, 'r')
instances = {}
for line in test_data:
    lineArray = line.split()
    instanceName = lineArray[0]
    label = lineArray[1]

    instances[instanceName] = {}
    instances[instanceName]['classLabel'] = label
        
    features = lineArray[2::2] # every other word in line starting with third
    values = lineArray[3::2] # every other word in line starting with fourth
        
    for f, v in zip(features, values):
        if not (f in instances[instanceName]):
            instances[instanceName][f] = 1
        else:
            instances[instanceName][f] += 1
    
test_data.close()


test_results = classify(instances, data_list[2], data_list[3], data_list[5])
test_acc = print_acc(test_results,instances, data_list[2])
print "Testing accuracy =", test_acc
