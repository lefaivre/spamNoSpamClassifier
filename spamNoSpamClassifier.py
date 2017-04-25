#-------------------------------------------------------------------------------------------------
# Adam Lefaivre - 001145679
# CPSC 5310 - Dr. Yllias Chali
# Programming Portion Assn. 2 - Spam classifier
#-------------------------------------------------------------------------------------------------
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import stopwords
import random
import os
import glob
import re

# A function to check for all non-stop words (like 'the', 'is', etc.) given an input email as a string
# The function returns feature words that are important for the classification
def getFeatures(emailAsInputString):
    # get all of the stop words as a set (i.e. remove any duplicate stop words, if need be)
    stopWords = set(stopwords.words('english'))

    # Declare a lemmatizer so that the input string can be lemmatized
    # Here we are using the word net lemmatizer which is based on the word net corpora
    lemmatizer = WordNetLemmatizer()

    # Now lemmatize the tokens from the email
    wordtokens = []
    wordTokenizationResult = word_tokenize(emailAsInputString)
    for word in wordTokenizationResult:
        wordtokens.append(lemmatizer.lemmatize(word.lower()))

    # Now loop through the potential non-stop words to check to see if they
    # are actually non-stop words. If they are in fact non-stop words
    # then we can make return a dictionary with each value for that word
    # as true, meaning that yes, it is a non-stop word
    dictToReturn = {}
    for word in wordtokens:
        if word not in stopWords:
            # Assign the value true if the word is not a stop word
            dictToReturn[word] = True
    return dictToReturn


#Beggining of the main portion of this program:

#Get the corpus from the enron folder.
#If the folder is not there ask the user to give the path for it.
print ("This program uses the Enron1 Spam dataset. To see a list of all of the other spam/ham datasets go here: http://www.aueb.gr/users/ion/data/enron-spam/")
print("Now loading in the enron1 dataset, that was supposed to be in the .tar submission file.")
parentPath = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
enronPath = parentPath + "/enron1/"

if (not os.path.isdir(enronPath)):
    print("For some reason the enron1 folder cannot be found")
    print "Please download the enron1 folder (from: http://www.aueb.gr/users/ion/data/enron-spam/) and place it inside this directory:", parentPath
    print "Exiting.  Please try this program again with the enron1 folder in its correct place!"
    exit()

# Declare lists for the two different types of emails
spamEmails = []  # spam
realEmails = []  # ham

print("Acquiring spam and non-spam emails from corpora")
# We must now get all of the REAL ("ham") emails using glob.
for emailFile in glob.glob(os.path.join(enronPath + "ham", '*.txt')):
    # Open this file and append its contents to the realEmails list declared above
    # Strip away all punctuation
    fileToBeRead = open(emailFile, "r")
    stringFromFileToBeAppended = fileToBeRead.read()
    stringFromFileToBeAppended = re.sub("[^a-zA-Z\d\s]+", "", stringFromFileToBeAppended)
    realEmails.append(stringFromFileToBeAppended)
    fileToBeRead.close()

# We must now get all of the SPAM emails using glob.
for emailFile in glob.glob(os.path.join(enronPath + "spam", '*.txt')):
    # Open this file and append its contents to the spamEmails list declared above
    # Strip away all punctuation
    fileToBeRead = open(emailFile, "r")
    stringFromFileToBeAppended = fileToBeRead.read()
    stringFromFileToBeAppended = re.sub("[^a-zA-Z\d\s]+", "", stringFromFileToBeAppended)
    spamEmails.append(stringFromFileToBeAppended)
    fileToBeRead.close()

# Store the email (which, recall, is a string) as a tuple, with the spam or ham keywords for our classifier
allEmails = ([(email, 'spam') for email in spamEmails])
allEmails += ([(email, 'ham') for email in realEmails])
random.shuffle(allEmails)

# Now, get all non-stop words per email (as string), if the word is a non-stop word it will be of the type: (word, true)
# Note that getFeatures returns a dictionary, so featureSets will be a list of dictionaries of tuples.
print("Acquiring feature words (i.e. non-stop words), for each email...")
featureSets = [(getFeatures(k), v) for (k, v) in allEmails]

# initialize SPAM calculation vals
SPAMaccuracyTotal = 0.0
SPAMprecisionTotal = 0.0
SPAMrecallTotal = 0.0

# initialize HAM calculation vals
HAMaccuracyTotal = 0.0
HAMprecisionTotal = 0.0
HAMrecallTotal = 0.0

# initialize Microaverage totals
MICROAVERAGEprecisionTotal = 0.0
MICROAVERAGErecallTotal = 0.0

#Declare the number of folds needed for classification, as well as the window size.
print("Now evaluating the Naive Bayes Classifier and classifying tester data, results will be printed shortly...")
folds = 10
windowSize = len(featureSets) / folds
if __name__ == '__main__':
    for i in range(folds):
        # So allow the tester data to be the feature sets within the window size
        # Slide the window along per iteration
        testerData = featureSets[i * windowSize:i * windowSize + windowSize]

        # Exclude all data within the window (everything up to the start of the window,
        # plus everything from the end of the window onwards)
        trainerData = featureSets[:i * windowSize] + featureSets[i * windowSize + windowSize:]

        # Train the classifier using the training data
        classifier = NaiveBayesClassifier.train(trainerData)

        # Note: Precision = TP/(TP + FP)
        # and,     Recall = TP/(TP + FN)

        # We can generate the following values based on these cases:

        # Spam Class

        #                  | spam | ham  |
        # -----------------|------|------|
        # classified spam  |  TP  |  FP  |
        # -----------------|------|------|
        # classified ham   |  FN  |  TN  |
        # -------------------------------|

        # Ham Class

        #                  | ham  | spam |
        # -----------------|------|------|
        # classified ham   |  TP  |  FP  |
        # -----------------|------|------|
        # classified spam  |  FN  |  TN  |
        # -------------------------------|

        SPAMtruePos = 0.0
        SPAMfalsePos = 0.0
        SPAMfalseNeg = 0.0

        HAMtruePos = 0.0
        HAMfalsePos = 0.0
        HAMfalseNeg = 0.0

        MICROAVERAGEtruePos = 0.0
        MICROAVERAGEfalsePos = 0.0
        MICROAVERAGEfalseNeg = 0.0

        SPAMaccuracy_ = 0.0
        SPAMprecision_ = 0
        SPAMrecall_ = 0.0

        HAMaccuracy_ = 0.0
        HAMprecision_ = 0
        HAMrecall_ = 0.0

        MICROAVERAGErecall_ = 0.0
        MICROAVERAGEprecision_ = 0.0

        # Loop through testing data and increment the TP, FP, and FN values accordingly
        for testerEmails in testerData:

            # We can now directly input new email and have it classified as either Spam or Ham,
            classifierVal = classifier.classify(testerEmails[0])
            actualVal = testerEmails[1]

            #SPAM CLASS CASES

            # case: TP
            if ((classifierVal is 'spam') and (actualVal is 'spam')):
                SPAMtruePos += 1
                MICROAVERAGEtruePos += 1

            # case: FP
            elif ((classifierVal is 'spam') and (actualVal is 'ham')):
                SPAMfalsePos += 1
                MICROAVERAGEfalsePos += 1

            # case: FN
            elif ((classifierVal is 'ham') and (actualVal is 'spam')):
                SPAMfalseNeg += 1
                MICROAVERAGEfalseNeg += 1


            #HAM CLASS CASES

            # case: TP
            elif ((classifierVal is 'ham') and (actualVal is 'ham')):
                HAMtruePos += 1
                MICROAVERAGEtruePos += 1

            # case: FP
            elif ((classifierVal is 'ham') and (actualVal is 'spam')):
                HAMfalsePos += 1
                MICROAVERAGEfalsePos += 1

            # case: FN
            elif ((classifierVal is 'spam') and (actualVal is 'ham')):
                HAMfalseNeg += 1
                MICROAVERAGEfalseNeg += 1


        #SPAM calculations
        SPAMaccuracy_ = classify.accuracy(classifier, testerData)
        SPAMaccuracyTotal += SPAMaccuracy_

        SPAMprecision_ = SPAMtruePos / (SPAMtruePos + SPAMfalsePos)
        SPAMprecisionTotal += SPAMprecision_

        SPAMrecall_ = SPAMtruePos / (SPAMtruePos + SPAMfalseNeg)
        SPAMrecallTotal += SPAMrecall_

        #HAM calculations
        HAMaccuracy_ = classify.accuracy(classifier, testerData)
        HAMaccuracyTotal += HAMaccuracy_

        HAMprecision_ = HAMtruePos / (HAMtruePos + HAMfalsePos)
        HAMprecisionTotal += HAMprecision_

        HAMrecall_ = SPAMtruePos / (HAMtruePos + HAMfalseNeg)
        HAMrecallTotal += HAMrecall_

        #Micro average calculations
        MICROAVERAGEprecision_ = MICROAVERAGEtruePos / (MICROAVERAGEtruePos + MICROAVERAGEfalsePos)
        MICROAVERAGErecall_ = MICROAVERAGEtruePos / (MICROAVERAGEtruePos + MICROAVERAGEfalseNeg)
        MICROAVERAGEprecisionTotal += MICROAVERAGEprecision_
        MICROAVERAGErecallTotal += MICROAVERAGErecall_

        print "|-----------------------------------------|"
        print " SPAM Accuracy for fold", i, "is: ", SPAMaccuracy_
        print " SPAM Precision for fold", i, "is: ", SPAMprecision_
        print " SPAM Recall for fold:", i, "is: ", SPAMrecall_
        print " SPAM F measure for fold:", i, "is: ", (2 * ((SPAMprecision_ * SPAMrecall_) / (SPAMprecision_ + SPAMrecall_)))
        print
        print " HAM Accuracy for fold", i, "is: ", HAMaccuracy_
        print " HAM Precision for fold", i, "is: ", HAMprecision_
        print " HAM Recall for fold:", i, "is: ", HAMrecall_
        print " HAM F Measure for fold:", i, "is: ", (2 * ((HAMprecision_ * HAMrecall_) / (HAMprecision_ + HAMrecall_)))
        print
        print " Macro-Average Precision for fold:", i, "is: ", ((SPAMprecision_ + HAMprecision_)/ 2)
        print " Macro-Average Recall for fold:", i, "is: ", ((SPAMrecall_ + HAMrecall_) / 2)
        print " Micro-Average Precision for fold:", i, "is: ", MICROAVERAGEprecision_
        print " Micro-Average Recall for fold:", i, "is: ", MICROAVERAGErecall_



# Calculate the averages across all folds
print "|-----------------------------------------|"
print " SPAM Average Accuracy is: ", (SPAMaccuracyTotal / folds)
print " SPAM Average Precision is: ", (SPAMprecisionTotal / folds)
print " SPAM Average Recall is: ", (SPAMrecallTotal / folds)
print " SPAM Overall F Measure is: ", (2 * ((SPAMprecisionTotal * SPAMrecallTotal) / (SPAMprecisionTotal + SPAMrecallTotal)))
print
print " HAM Average Accuracy is: ", (HAMaccuracyTotal / folds)
print " HAM Average Precision is: ", (HAMprecisionTotal / folds)
print " HAM Average Recall is: ", (HAMrecallTotal / folds)
print " HAM Overall F Measure is: ", (2 * ((HAMprecisionTotal * HAMrecallTotal) / (HAMprecisionTotal + HAMrecallTotal)))
print
print " Macro-Average Precision across folds is:", ((SPAMprecisionTotal + HAMprecisionTotal) / 2)
print " Macro-Average Recall for fold:", i, "is: ", ((SPAMrecallTotal + HAMrecallTotal) / 2)
print " Micro-Average Precision for fold:", i, "is: ", MICROAVERAGEprecisionTotal / folds
print " Micro-Average Recall for fold:", i, "is: ", MICROAVERAGErecallTotal / folds
print "|-----------------------------------------|"


allClassifier = NaiveBayesClassifier.train(featureSets)
#Go ahead and allow for user input!
print("Evaluation complete, user can now classify her/his own documents")
while(True):
    controlInput = raw_input('Please enter your own email to classify (as a text file, for ex. /dir1/dir2/something.txt), or type q to quit: ')
    if(controlInput == "q"):
        print("good bye!")
        exit()

    while(not os.path.isfile(controlInput)):
        controlInput = raw_input('This is not a file.  Please enter your own email to classify (as a text file, for ex. /dir1/dir2/something.txt), or type q to quit: ')
        if(controlInput == "q"):
            print("good bye!")
            exit()

    fileToBeRead = open(controlInput, "r")
    stringFromFileToBeAppended = fileToBeRead.read()
    stringFromFileToBeAppended = re.sub("[^a-zA-Z\d\s]+", "", stringFromFileToBeAppended)

    keyWords = getFeatures(stringFromFileToBeAppended)
    print ("\n")
    print "This email is", allClassifier.classify(keyWords)
    print ("\n")
