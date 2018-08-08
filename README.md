# DecisionList-Classification-WordSenseDisambiguation-NLP-Python
Yarowsky's log-Likelihood decision ist model

Task at hand:
The task at hand is a classification system based on a decision list that defines best log likelihood valued rules with preference to highest rule. The algorithm begins by listing a simple bag of words which helps in determining the ambiguities in the classes. The next step here is to collect the training context which is creating a labeled data with the class presence removed(Removal of accents) and having them in separate labeled set of data, after that we measure the collocational distribution by extracting features from both the class data available to us, the features taken in to considerations are linguistic features for that for the model are listed below-

1]  Word to right
2] Word to left
3] Pair of words at offset
4]  Word found in +-K window

After careful, feature extraction we compile them to get the collocational distribution, also the paper encourages use of more linguistic based features if possible to extract like Parts of Speech, lemmatization etc. The next step is very important the log-likelihood for all features is extracted and a rank-order list called decision rule list is created with best log-likelihood having the highest spot. Now, the testing is done on this basis and the decision rule list is used upon the extracted features of the testing set to determine the class for that test data. A further optional pruning step is also defined. The main step is that of training the decision lists for general classes of ambiguity which determines which feature rules are to be used for evaluation.
The primary assumption being made here is that the model will work quite well even on test data with no matching decision rule and other assumption is that of a binary classification system which might be very narrowed for classification in general.
The decision list approach can be used for many different spectrums of classification even incorporated into non-binary classification, a key application of such a list can be in clustering of text of images where the decision list approach will be very important in determining which cluster to place the object in from the domain. The decision list with modification can also be used for prediction based tasks like regression or decision trees.

My model and conceptual decisions:

Conceptually, the task was to implement a binary classification that will align itself to a class based on features partial towards the class. This is achieved first by extracting the features which is a bag -of – words form of values for many linguistics based data, then this data is used upon the training context developed and finally a value is achieved which determines which features are more likely to align with which class for that train data. That is different training data will have different decision lists, the decision list is used to match the same features for a test data and if the same features are detected then based on the rules a class is assigned to that test data instance. The features, used by my model are listed below-

1-POS at -k
2-POS at +k
3-last word in context
4- Word at +1
5- Word at -1

A major decision that I took was related to the baseline, when the feature extracted seem sparse that is many instances of test data do not match to that on the decision list, the model after considering all features on the list assign the baseline majority class to that data. This although makes the data more like a baseline model, improves on it in some instances.

Reflection on results:
The results achieved are self-proving that is the metric is an indication for values found in confusion matrix, the confusion matrix shows that the low numbers of *Sake is reflected with 50% of it being false negative. While the near perfect distribution for bass is also evident with equal numbers of bass and *bass in the set. Also, the metrics show that the sake has almost near perfect recall which suggest that the model trained well but what is not observed is the sparsity of one class compared to other class in sake database. The absolute change suggests the model accuracy is not far from baseline which suggest average performance which should be improved by interpolation or pruning or selection of better features. The primary concern is for the distribution of data in the data set, which was sparse in one while very well distributed in other. The sake testing although seem right is inherently very inaccurate while the bass feels wrong but is considerably accurate.
