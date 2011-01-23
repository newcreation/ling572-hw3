import sys
import math

def generate_model(train_data_filename, model_filename, prior_delta, cond_delta):
    classLabels = {}
    featureCounts = {}
    instances = {}
    allFeatures = set()
    featuresInLabel = {}

    train_data = open(train_data_filename, 'r')
    for line in train_data:
        lineArray = line.split(' ')
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
                instances[instanceName][f] = 1  # would be v in multinomial
            else:
                instances[instanceName][f] += 1
    
    train_data.close()
    
    # cache logs for faster calculations
    logs = {}
    biggestLog = sum(featuresInLabel.values())
    if prior_delta >= cond_delta: 
        biggestLog += (prior_delta*2)
    else:
        biggestLog = (cond_delta*2)
    for i in range(1,biggestLog+1):
        logs[i] = math.log(i)
    
    #store probabilities
    labelProbs = {}
    labelLogProbs = {}
    featureProbs = {}
    featureLogProbs = {}
    for label in classLabels:
        # get label probabilities
        labelProbs[label] = (classLabels[label]['numInstances'] + prior_delta) \
        / (len(instances)+(prior_delta*2.0)) #2 for binomial
        labelLogProbs[label] = logs[classLabels[label]['numInstances'] + \
        prior_delta] - logs[len(instances) + (prior_delta*2)]
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
        output[instance] = []

#Hey, this looks like great code to me, much cleaner and more succinct than I write.  I'm not sure why you're putting a list as the value here, so I don't want to mess up what you're doing.
#I started writing something like this:
#in your instance loop, split the instance if necessary, extract the features
#loop through the classes, outside this loop initialize a variable called argmax to "" and one called prob to 0
#loop through all possible features
#get the p(f|c)'s from featureLogProbs, add them all up
#if it's greater than prob, overwrrite argmax with the label of the class
#when done, write argmax to output[instance]

#thanks for working with me - you're obviously a better coder.  if you want to keep working with me after this assignment, i think i can get up to speed, but if not, i can understand.

#let's talk about the multinomial model tomorrow, looks like you already have an implementation in mind by your comments. feel free to delete these when read.
		



if (len(sys.argv) < 7):
    print "Not enough args."
    sys.exit(1)

train_data_filename = sys.argv[1]
test_data_filename = sys.argv[2]
prior_delta = int(sys.argv[3])
cond_delta = int(sys.argv[4])
model_filename = sys.argv[5]
sys_filename = sys.argv[6]


list = generate_model(train_data_filename, model_filename, prior_delta,\
cond_delta)
print list[0]


