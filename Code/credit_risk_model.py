import pandas as pd
import numpy as np
import seaborn as sns;
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Creating the Spark Session


spark = SparkSession.builder 
        .master("local") 
        .appName("Credit_risk_modelling_app") 
        .getOrCreate()
	
# Reading dataframe via Spark
loan_df = spark.read.csv(“./dataset/loans.csv”, inferSchema = True, header = True, sep=”,”)


loan_df.printSchema()

loan_df.select(“Property_Area”).distinct().show()




#loan_df = loan_df.withColumn('Married', when(col('Married')=='No', N).otherwise(Y))
#loan_df = loan_df.withColumn('Education', when(col('Education')=='Not Graduate', N).otherwise(Y))
#loan_df = loan_df.withColumn('Self_Employed', when(col('Self_Employed')=='No', N).otherwise(Y))


loan_df = loan_df.withColumn('Married', when(col('Married')=='No', 0).otherwise(1))
loan_df = loan_df.withColumn('Education', when(col('Education')=='Not Graduate', 0).otherwise(1))
loan_df = loan_df.withColumn('Self_Employed', when(col('Self_Employed')=='No', 0).otherwise(1))


drop_cols_list = [‘Loan_ID’, ‘Property_Area’]
loan_df = loan_df.select([column for column in loan_df.columns if column not in drop_cols_list])


loan_df = loan_df.select('Gender',
               'Married',
               col('Dependents').cast('int').alias('Dependents'),
              'Education',
              'Self_Employed',
              'ApplicantIncome',
              'CoapplicantIncome',
              'LoanAmount',
              'Loan_Amount_Term',
              'Credit_History',
              'Urban',
              'Semiurban',
              'Rural',
              'Loan_Status')
			  

## gather numerical features
numerical_features = [t[0] for t in loan_df.dtypes if t[1] == ‘int’ or t[1]==’double’]
loan_df_numeric = loan_df.select(numerical_features).describe().toPandas().transpose()


sns.set()
correlation_loan_df = loan_df.select(numeric_features).toPandas().corr()

sns.set(rc={"font.style":"normal",
            "axes.titlesize":20,
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':30,
            'figure.figsize':(8.0, 8.0),
            'xtick.labelsize':15,
            'ytick.labelsize':15})
            
#sns.heatmap(correlation_loan_df, annot = False, annot_kws={"size": 7}, cmap="YlGnBu", linewidths=.6)
sns.heatmap(correlation_loan_df, annot = True, annot_kws={"size": 7}, cmap="YlGnBu", linewidths=.5)
#sns.heatmap(correlation_loan_df, annot = False, annot_kws={"size": 8}, cmap="YlGnBu", linewidths=.5)
#sns.heatmap(correlation_loan_df, annot = True, annot_kws={"size": 10}, cmap="YlGnBu", linewidths=.9)


# check categorical columns in dataframe :
categorical_cols = [i[0] for i in loan_df.dtypes if i[1].startswith(‘string’)][:1]

# check numerical columns in dataframe:
numerical_cols = [i[0] for i in loan_df.dtypes if i[1].startswith(‘int’) | i[1].startswith(‘double’)]



def total_null_count(c):
    null_counts = []          
    for col in c.dtypes:     
        cname = col[0]            
        ctype = col[1]        
        null_values = c.where( c[cname].isNull()).count() 
        result = tuple([cname, null_values])  
        null_counts.append(result)      
    null_counts=[(x,y) for (x,y) in null_counts if y!=0]  
    return null_counts
    
 miss_counts = total_null_count(loan_df)



missing_value_list=[x[0] for x in miss_counts] 
loan_df_missing= loan_df.select(*missing_value_list)

# categorical columns
categorical_colums_miss=[i[0] for i in loan_df_missing.dtypes if i[1].startswith('string')] 

# numerical columns
numerical_columns_miss = [i[0] for i in loan_df_missing.dtypes if i[1].startswith('int') | i[1].startswith('double')] 


## Creating pipeline stages


stages = []
for categoricalCol in cat_cols:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
labelIndexer=StringIndexer(inputCol='Loan_Status',outputCol='label')
stages+=[labelIndexer]
#numeric_cols=['minimum_nights_int', 'number_of_reviews_int', 'reviews_per_month_int']
assemblerInputs = [c + "classVec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="unscaled_features")
stages += [assembler]

#standardize and scale features to mean 0 and variance 1
standardScaler = StandardScaler(inputCol="unscaled_features",
                                outputCol="features",
                                withMean=True,
                                withStd=True)
stages += [standardScaler]




cols = loan_df.columns
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(loan_df)
loan_df = pipelineModel.transform(loan_df)
selectedCols = ['label','features']+cols
loan_df = loan_df.select(selectedCols)
pd.DataFrame(loan_df.take(5), columns=loan_df.columns).transpose()


(train, test) = loan_df.randomSplit([0.7, 0.3], seed = 1)


## Implementing Logistic Regression

lr = LogisticRegression(featuresCol = ‘features’, labelCol = ‘label’, maxIter=100)
lrModel = lr.fit(train)



beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel(‘Beta Coefficients’)
plt.show()


trainSet = lrModel.summary
roc = trainSet.roc.toPandas()
plt.plot(roc[‘FPR’],roc[‘TPR’])
plt.ylabel(‘False Positive Rate’)
plt.xlabel(‘True Positive Rate’)
plt.title(‘ROC Curve’)
plt.show()
print(‘TrainSet areaUnderROC: ‘ + str(trainSet.areaUnderROC))


pr = trainSet.pr.toPandas()
plt.plot(pr[‘recall’],pr[‘precision’])
plt.ylabel(‘Precision’)
plt.xlabel(‘Recall’)
plt.show()


## Predicting Values
predictions = lrModel.transform(test)
predictions.select(‘ApplicantIncome’, ‘CoapplicantIncome’, ‘Loan_Amount_Term’, ‘Credit_History’, ‘prediction’, ‘probability’).show(10)



## Implementing Random Forest Classifier

# Creating RandomForest model.
rf = RandomForestClassifier(labelCol=”label”, featuresCol=”features”)

rfModel = rf.fit(train)
## make predictions
predictions = rfModel.transform(test)
rfPredictions = predictions.select("label", "prediction", "probability")
rfPredictions.show(10)



## Random Forest Classifier

evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)
print(‘Random Forest Test areaUnderROC: {}’.format(evaluator.evaluate(predictions)))



## Gradient-Boosted Tree Classifier

gbt = GBTClassifier(labelCol=”label”, featuresCol=”features”,maxIter=10)
pipeline = Pipeline(stages=stages+[gbt])
(traininggbt, testgbt) = loan_df_copy.randomSplit([0.7, 0.3], seed=1)
gbtModel = pipeline.fit(traininggbt)




predictions =gbtModel.transform(testgbt)
predictions.select(‘label’, ‘prediction’, ‘probability’).show(10)


evaluator = BinaryClassificationEvaluator()
print(“GBT Test Area Under ROC: “ + str(evaluator.evaluate(predictions, {evaluator.metricName: “areaUnderROC”})))



paramGrid = ParamGridBuilder()
 .addGrid(lr.aggregationDepth,[2,5,10])
 .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])
 .addGrid(lr.fitIntercept,[False, True])
 .addGrid(lr.maxIter,[10, 100, 1000])
 .addGrid(lr.regParam,[0.01, 0.5, 2.0]) 
 .build()
 
 
 cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Cross validation
cv_model = cv.fit(train)
predict_train=cv_model.transform(train)
predict_test=cv_model.transform(test)
print(“Cross-validation areaUnderROC for train set is {}”.format(evaluator.evaluate(predict_train)))
print(“Cross-validation areaUnderROC for test set is {}”.format(evaluator.evaluate(predict_test)))




# Output 
#Cross-validation areaUnderROC for train set is 0.8362328401543963
#Cross-validation areaUnderROC for test set is 0.815426590786455