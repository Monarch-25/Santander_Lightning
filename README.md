# Santander_Lightning (This is long after the competition results were declared mainly for my own learnings)

Achieving top 1% like AUC using only vanilla neural network with the help of feature engineering and observation.

## Feature Engineering

### Technical Part:

The "magic" lies in the count of values, particularly the fact that some are unique. We have created 200 categorical features (one per raw feature), referred to as "has one feat". These features have 5 categories that correspond to different characteristics for training data:

1. This value appears at least another time in data with target==1 and no 0.
2. This value appears at least another time in data with target==0 and no 1.
3. This value appears at least two more times in data with target==0 & 1.
4. This value is unique in data.
5. This value is unique in data + test (only including real test samples).

Additionally, the other 200 features (one per raw feature) are numerical, referred to as "not unique feat". These features correspond to the raw feature replacing values that are unique in data + test with the mean of the feature.

## Journey to Finding

1. Initially prepared a baseline logistic model with batch normalization and achieved an AUC of 0.81.
2. Noticed lack of correlation between features, leading to treating each example as its own unique predictor.
3. Created features around uniqueness, achieving a .910 LB using only training data and the "has one feat".
4. Added the other 200 "not unique feat" features, leading to a .914 LB.
5. Utilized data + test to spot unique values, resulting in >.92x CV results.
6. Collaborated with Silogram and leveraged insights from the kernel "List of Fake Samples and Public/Private LB split" by @YaG320.
7. Achieved an LB of .921 using NN and finalized the set of features.

These features collectively contributed to the successful outcome of the project.
