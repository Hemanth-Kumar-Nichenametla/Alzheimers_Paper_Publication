# Predictive Analytics for Diagnosing Alzheimer's Disease using Artificial Intelligence and Machine Learning Algorithms.

## Project Abstract:

Predictive analytics plays a vital role in the healthcare industry in providing timely insights and decisions to all the stakeholders of the healthcare industry. Machine Learning has become predominant in the healthcare industry for predicting various diseases that cause severe health issues in humans. One such disease is Alzheimer’s, a common neurological condition that affects a sizable number of the elderly population. Although the symptoms of this disease are very minimal in the beginning, they become critical over time and can lead to dementia. Early prediction of this disease can improve a patient’s health status and facilitate the improvement of targeted interventions. This study uses various machine learning algorithms such as Logistic Regression, KNN, Decision Tree, SVC, Random Forest, Hard Voting Classifier, Soft Voting Classifier, Gradient Boosting Classifier, Extreme Gradient Boosting Classifier, and Ensemble Learning Model to predict Alzheimer’s disease. This research uses the Open Access Series of Imaging Studies (OASIS) longitudinal dataset to build different machine learning models by using multiple machine learning algorithms to predict Alzheimer’s disease. The obtained results clearly indicate that ensemble model achieved a higher validation accuracy of 87.3% and test accuracy of 85.3%, in comparison with the other machine learning algorithms used in this work.

## Dataset Preparation

In this research work, we have used the Open Access Series of Imaging Studies
(OASIS) longitudinal dataset consists of details about Alzimer’s diseased (Demented
Group) patient and a normal patient (Non-Demented Group) feature. This data set
comprises of approximately 15 features, comprising of both numerical and categorical
data. The Open Access Series of Imaging Studies (OASIS) dataset, with its 150 rows
and 15 columns, is a pivotal resource in neuroimaging and cognitive research. Its
longitudinal design allows researchers to study changes over time in individuals,
including those with conditions like Alzheimer's disease. Despite its smaller size, the
dataset offers a rich set of features and clinical data, supporting the development of
machine learning models, diagnostic tools, and novel analytical approaches for
cognitive disorder detection and monitoring. The OASIS dataset's columns provide
essential information about subjects, their demographics, and neuroimaging and
cognitive assessment measures, fostering advancements in brain health and cognitive
disorder research. A snapshot of the dataset is provided in Table 1 and a
comprehensive description of its features is provided in the below tables.

### Snapshot of the dataset

![Alt text](<Important_images/Screenshot 2024-03-10 at 1.15.44 PM.png>)

### Dataset Description

![Alt text](<Important_images/Screenshot 2024-03-10 at 1.16.39 PM.png>)



Few features contain the missing values, and everything can be dealt with in data
preprocessing phase and is explained in the below phase. Neat data preparation is
crucial for accurate Early prediction of Alzheimer's disease using machine learning
models. In this case study, we have implemented key preprocessing techniques on the
Open Access Series of Imaging Studies (OASIS) longitudinal dataset. These
techniques include handling missing values, addressing categorical data, and scaling
or normalizing features.
There are few columns such as “SES” and “MMSE” consisting of null values, we
employed imputation techniques such as median and mean imputation to eliminate
null values from the data set. Categorical data was encoded using one-hot encoding or
label encoding, depending on the cardinality of the variables. In this case the target
variable that is “Group” column contains categorical data with three unique values
“Demented”, “Non-Demented”, “Converted”. Converted values are replaced with
Demented as it is the status of showing dementia in a person. Now there are two
unique values present in the target variable and one-hot encoding is implemented on
this column. Finally, demented is converted into ‘1’ and non-Demented is converted
to ‘0’. These Data Preprocessing techniques ensured a complete and suitable dataset
for training machine learning models, enhancing the accuracy and reliability of
Alzheimer's disease predictions in early stages.
The following Table 3 shows the minimum, maximum, mean and median values of
each column of the data set to see how the data is distributed and getting to know
important data columns in the given data set.

### Minimum, Maximum, Median and Mean Values of each column.

![Alt text](<Important_images/Screenshot 2024-03-10 at 1.18.44 PM.png>)

Further, we implemented all the machine learning classification models and evaluated
their performance using validation accuracies and test accuracies. We applied feature
scaling to the model that achieves the highest accuracy. To ensure robustness of our
evaluation, we will employ the 5-fold cross-validation technique and obtain validation
accuracies.

## Methodology

The following list of machine learning algorithms are used in our research to classify
Alzheimer’s disease and evaluated the performance of these algorithms to diagnose
the disease with greatest accuracy. The achieved results of these algorithms are
discussed in the results section of this paper.

### Logistic Regression

The Logistic Regression is a Classification Machine Learning model used to classify
the two-class categorical (Yes or No, Y or N, True or False) or numerical (0 or 1)
data. It can be implemented in all the cases where the values of the target variable are
in binary choices like as mentioned above True or False, Yes or No, 0 or 1 etc. This
logistic Regression model mainly gets the probability of the input with which it
belongs to the specific class. Based on these probability values the class name can be
determined. The activation function to determine probabilities used is a sigmoid
function. To be specific this function has the range of values between 0 and 1 so the
probabilities. And the output from sigmoid function is nothing but the required
probability values. So, if the aim is to predict the probabilities of the output in that
case Logistic regression model can be used undoubtedly.

### Decision Tree
A decision tree is a type of classification technique that has a flowchart-like visual
representation. Internal nodes in this structure act as key nodes for running tests on
features. The results of these tests decide the direction to travel, leading us to the
suitable leaf node that corresponds to a specific class label. The mix of features that
go into the classification process is represented by the branches that join the nodes. As
a result, by exploring the routes from the root to the leaf nodes, we reveal the
fundamental principles behind efficient categorization.

### Support Vector Machine
SVM is a machine learning model which is used for both regression and classification
tasks. They are especially used for classification of complex small and medium size
data. Through non-linear mapping, it transforms the original training data into
elevated higher dimensions. It looks for a linear optimal separating hyperplane
through those new dimensions. With the help of nonlinear mapping, the data from
both the classes can be separated by using a hyperplane. The hyperplane can be found
using the support vectors and margins. By maximizing the margin and minimizing the
classification errors, SVM implements the classification task.

### Random Forest
Random Forest is one of the easy-to-use algorithms in machine learning which
provides accurate results even when hyper-tuning is not used. Random Forest is a
kind of ensemble learning which overcomes the limitation of decision trees of
overfitting. Ensemble learning uses multiple algorithms or same algorithms multiple
times. Random forest is a collection of the decision trees. By harnessing the collective
wisdom of these trees, random forest surpasses the performance of individual trees by
mitigating errors and reducing prediction correlations. To achieve accurate class
predictions, it requires informative features and strives to maintain low correlations
among the trees by randomly sampling the training data and selecting a subset of
features for each tree.

### Soft Voting Classifier
Soft voting is an ensemble learning approach used to combine the probabilities
predicted by different ML algorithms, resulting in a more refined and probabilistic
final prediction. Unlike hard voting, which relies on majority voting, soft voting
calculates the average of the class probabilities assigned by each model. By
considering the weighted average of these probability values, soft voting provides a
more nuanced decision-making process. For instance, if one model predicts a stone
with a 40% probability and another predicts it as a oil with an 80% probability, the
ensemble prediction would assign a probability of 60% for the object being an oil.
This technique is applicable in fields such as machine learning classification tasks,
accounts for the confidence levels of each algorithm and enhances accuracy and
robustness by leveraging the collective knowledge of multiple algorithms. In simple
words it is the combination of multiple classifiers to make collective predictions. By
aggregating the predicted probabilities from each classifier, the Soft Voting Classifier
achieves accurate and robust predictions.

### Hard Voting Classifier
Hard voting is a widely employed technique. It falls under the family of ensemble
learning, Here the final prediction is determined by selecting the class with the
highest number of votes among individual ML algorithms. This approach involves
independent predictions from each algorithm, and the ensemble combines these
predictions to arrive at a collective decision in terms of voting. For instance, in the
case of predicting the color of a specific drink, if three algorithms individually predict
"Pink", "blue" and "blue" the ensemble will ultimately predict "blue" due to its
majority number of votes. Hard voting simplifies the decision-making process by
considering the majority consensus among the algorithms. This technique finds
extensive application across various fields, particularly in machine learning
classification tasks, where the combination of predictions from multiple models
enhances accuracy and robustness. Leveraging the collective wisdom of the ensemble,
hard voting offers a straightforward and effective approach to making class
predictions.

### Gradient Boosting Classifier
Gradient Boosting is a powerful classification algorithm that iteratively improves
predictions by fitting new predictors to the residual errors made by previous
predictors. It starts by calculating the log odds and converting them to probabilities
using a logistic function. Residuals are then computed and used to build a decision
tree. To make predictions, the algorithm combines the base log odds with scaled
output values from each tree, adjusting for bias and variance using a learning rate.
This process continues until a stopping criterion is met. The resulting algorithm
provides accurate predictions for unseen instances and is a valuable tool in various
classification tasks.

### Extreme Gradient Classifier
Extreme Gradient Boosting (XGBoost) is a highly optimized and efficient algorithm
that offers a unique approach to tree construction and pruning, resulting in accelerated
training speed and improved performance, particularly when dealing with large-scale
datasets. XGBoost employs the Similarity Score and Gain metrics to determine the
most effective node splits by considering the residuals (the disparity between actual
and predicted values) and the previous probability calculated at each iteration. By
incorporating advanced techniques such as the Approximate Greedy Algorithm,
Parallel Learning, Sparsity-Aware Split Finding, and Cash-Aware Access, XGBoost
achieves faster processing and higher accuracy compared to conventional Gradient
Boosting methods. This algorithm has emerged as a crucial component in enhancing
model performance and has gained substantial popularity across diverse machine
learning applications.
### Ensemble Classifier Model
Ensemble learning, a prominent machine learning technique, has gained widespread
recognition for its ability to significantly improve model accuracy. It accomplishes
this by amalgamating multiple machine learning models. This approach encompasses
four key features: Bagging, which employs majority voting and selects random
subsets of training data to train diverse base learners, ensuring their dissimilarity;
AdaBoost, a boosting technique that sequentially creates base classifiers using
bootstrap samples with instance-weight adjustments based on misclassification rates;
and Stacking, a strategy that combines various machine learning models to enhance
predictive accuracy by leveraging the strengths of diverse algorithms. These ensemble
methods collectively contribute to achieving superior model accuracy and
performance.
### KNN Classifier Model
K-Nearest Neighbors (KNN) is a versatile and non-parametric supervised learning
method widely applied in areas like pattern recognition, data mining, and intrusion
detection. It groups unclassified data points based on their proximity to training set
neighbors, utilizing distance metrics like Euclidean, Manhattan, and Minkowski
distances. By analyzing clusters of data points, KNN intuitively determines a point's
classification based on its nearest neighbors' group. Selecting the right 'k' value
(number of neighbors) is vital and can be determined through cross-validation. KNN
is used in data preprocessing, pattern recognition, and recommendation engines,
offering benefits like ease of implementation, adaptability to new data, and minimal
hyperparameters. Nonetheless, it faces scalability, dimensionality, and overfitting
challenges, which can be addressed using techniques like feature selection and
dimensionality reduction to enhance its performance.
### Naïve Bayes Classifier
It is a supervised machine learning model used for a classification problem designed
to differentiate objects based on specific features. It utilizes algorithms to assign
objects to predefined categories, playing a crucial role in various applications. Among
the classifiers, the Naive Bayes classifier operates on the principle of Bayes theorem.
This probabilistic model assumes feature independence and calculates the probability
of a hypothesis given the evidence. Naive Bayes classifiers find extensive application
in sentiment analysis, spam filtering, and recommendation systems, offering fast
implementation and ease of use. However, their performance may be impacted when
predictors are not truly independent, highlighting a limitation in real-life scenarios.
### Multilayer Perceptron Classifier
The Multilayer Perceptron (MLP) classifier is a powerful feed-forward artificial
neural network that is widely used for supervised machine learning tasks. It utilizes
the back propagation learning technique, making it capable of handling complex non-
linear problems. The MLP demonstrates excellent performance, even with large input
datasets, and maintains a high accuracy ratio even with smaller data. These qualities
make the MLP highly valuable in various classification tasks within the field of
machine learning.

The dataset is split into an 80:20 ratio, with 80% allocated to the training data and the
remaining 20% designated as the testing data set. Following this, all the classification
models are fitted using the training data, and their performance is evaluated using the
test data. To ensure robustness in the evaluation process, a 5-fold cross-validation
technique is employed on the training data.

### Metrics details of all the Machine Learning Models.

![Alt text](<Important_images/Screenshot 2024-03-10 at 1.07.52 PM.png>)

## Conclusion

Alzheimer's Disease is a severe neurodegenerative disorder that greatly impacts
individuals and their families. Detecting the disease in its early stages is of paramount
importance, as it allows for timely medical interventions and support services. In this
research, we aimed to develop multiple machine learning models to diagnose
Alzheimer’s Disease (AD) very early and evaluated the performance of each
algorithm with different performance metrics. Among the models evaluated, the
ensemble model emerged as the top performer, exhibiting the highest accuracy,
validation accuracy, and area under the curve (AUC). The ensemble model's superior
performance provides a valuable tool for identifying AD cases at an early stage,
potentially leading to improved patient outcomes and quality of life. Furthermore, the
ensemble model's excellent AUC highlights its capacity to discriminate between AD
and non-AD individuals accurately. This discrimination ability is crucial for accurate
diagnosis and the development of targeted treatment strategies. In conclusion, our
research demonstrates that the ensemble model surpasses other models in terms of
highest accuracy of 85.3%, validation accuracy of 87.3%, and AUC of 86%. The
ensemble model offers a promising technique for reliable and accurate AD prediction.
Further research and validation studies are encouraged to validate its effectiveness
and explore its integration into clinical practice.