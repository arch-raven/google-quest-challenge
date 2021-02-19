# google-quest-challenge
The google-quest-challenge is an NLP competition that was hosted on Kaggle. 

 [Link to Kaggle page](https://www.kaggle.com/c/google-quest-challenge/)
```console
kaggle competitions download -c google-quest-challenge
```

 ### Things to try in priority order

 - [x] Stable cross validation split, used GroupKFold with question_title group
 - [x] Separate learning rate 3-e5 for transformers 0.005 for heads with cosine schedule
    - [x] Freeze the transformer layer and train head for 1 epoch
    - [x] train transformer layer and header layer for different learning rate
 - [ ] Using weighted sum of CLS outputs form all BERT layers rather than using only the last one, constraining the weights to be positive and sum to 1
 - [ ] Truncate the input text in different way when input is exceed the limit of input length
 - [ ] Different learning rate for each bert layer
 - [ ] Multi-Sample Dropout
 - [ ] Separate model for question and answer
