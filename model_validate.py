# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:22:21 2019

@author: rmahajan14
"""
#import pyspark
#from pyspark.ml import Pipeline
#from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark.ml.feature import HashingTF, Tokenizer
#from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


from model1 import model, training, test

crossval = CrossValidator(estimator=model,
                          estimatorParamMaps=[0,0.1,0.2,0.3,1],
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=10)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(training)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(test)
#selected = prediction.select("id", "text", "probability", "prediction")
#for row in selected.collect():
#    print(row)