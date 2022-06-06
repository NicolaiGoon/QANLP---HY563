# Nikos Gounakis , HY563 Project , csdp1254

## Question Type Prediction & Entity Type Prediction

We used a machine learning approach to predict the question and the entity type. In order to produce training data for each of the two problems we used the questions from the provided collection along with their labels (target classes). We feed a sentence-transformer (`sentence-transformers/all-MiniLM-L6-v2`) model with the question to get numeric features and we export a weka file containing the features and the corresponding label for each sample. So at the end we have two weka files, one with features and question types and one with features and entity types.

Note: We merged the entity types in questions that had two entity types.

We used the `AUC` metric because we have a multiclass classification and class imbalances.
For training method we used 10-fold cross validation.

We had some problems using the exported Weka model file in python, so we used Weka only to compare the classifiers. Then using sklearn we build the models.

### Question Type Prediction

| Model          | AUC     |
| -------------- | ------- |
| Vote           | 0.486   |
| `RandomForest` | `0.792` |
| SMO            | 0,770   |

### Entity Type Prediction

| Model        | AUC     |
| ------------ | ------- |
| Vote         | 0,432   |
| RandomForest | 0,796   |
| `SMO`        | `0,820` |


