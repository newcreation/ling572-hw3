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
            v = int(v)
            allFeatures.add(f)
            featuresInLabel[label] += v;
            if not(f in classLabels[label]):
                classLabels[label][f] = v
            else:
                classLabels[label][f] += v
            if not (f in featureCounts):
                featureCounts[f] = v  
            else:
                featureCounts[f] += v
            if not (f in instances[instanceName]):
                instances[instanceName][f] = v  
            else:
                instances[instanceName][f] += v
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
        li = float(len(instances))
        labelProbs[label] = (classLabels[label]['numInstances'] + prior_delta) \
        / li+(prior_delta*len(classLabels)) 
        labelLogProbs[label] = logs[classLabels[label]['numInstances'] + \
        prior_delta] - logs[li + (prior_delta*len(classLabels))]

        # conditional probabilities
        featureProbs[label] = {}
        featureLogProbs[label] = {}
        for feature in classLabels[label]:
            if feature in classLabels[label]:
                #calculate P(feature|label)
                l = float(len(allFeatures))
                featureProbs[label][feature] = (classLabels[label][feature] + \
                cond_delta) / (featuresInLabel[label] + \
                (cond_delta*l))

                #calculate log10(P(feature|label))
                featureLogProbs[label][feature] = logs[classLabels[label][feature] + \
                cond_delta] - logs[featuresInLabel[label] + \
                cond_delta*l]

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
        print featureProbs[label]
        for feature in featureProbs[label]:
            print "hello"
            model_file.write(feature + "\t" + label + "\t")
            model_file.write(str(featureProbs[label][feature]) + "\t")
            model_file.write(str(featureLogProbs[label][feature]) + "\n")
    return [instances, allFeatures, labelLogProbs, featureLogProbs]
# main
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
