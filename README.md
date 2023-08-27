# Santander_Lightning
Achieving top 1% like AUC using only vanilla neural network using feature engineering and observation.

## Feature engineering:
Technical part:
The "magic" is about count of values, especially the fact that some are unique.
We created 200 (one per raw feature) categorical features, let's call them "has one feat", with 5 categories that corresponds (for train data) to:

This value appears at least another time in data with target==1 and no 0;
This value appears at least another time in data with target==0 and no 1;
This value appears at least two more time in data with target==0 & 1;
This value is unique in data;
This value is unique in data + test (only including real test samples);
The other 200 (one per raw feature) features are numerical, let's call them "not unique feat", and correspond to the raw feature replacing values that are unique in data + test with the mean of the feature.

Journey to finding :
Initially prepared a baseline logistic model with batchNormalization got an AUC of 0.81.
After some EDA I noticed that the lack of correlation between the features since there was not correlation I thought of treating each example as its own unique predictor and that number of different values in train and test was not the same.

After this, I started to build features around uniqueness. Using only training data and the "has one feat", I could reach .910 LB. Adding the other 200 "not unique feat", .914LB.
The next move was to use data + test to spot unique values. It worked really well on CV, giving >.92x results but didn't apply to test as is!
As many people noticed, the count of unique values per feature in data and test is very different! So I knew that there was a subset of samples in test that I couldn't identify yet that would bring >.92x LB. I teamed with Silogram at this moment. The day after he sent me a link to the beautiful and very important kernel of @YaG320 (rick and morty's fans are the best!) "List of Fake Samples and Public/Private LB split". I immediately understood that this was the key to spot values that are unique in data + test!
Finally got LB .921 using LGBM at this time, and these are the features we used at the end.
