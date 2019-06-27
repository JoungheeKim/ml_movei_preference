# ml_movei_preference
This project is to find movie preference by using BTV dataset which is given by "DATA Analysis Contest"

### How to use it?
Follow the example below. There is 3 process - Scrap, Train, Test.

#### 1 Scrap additional movie infromation
To scrap additional movie information from 'naver movie web' you need to put some options
1. src_path : Source path that has source csv file
2. tgt_path : Target path that file is saved

##### EXAMPLE
```python
python scrap.py --src_path data/SKB_DLP_MOVIES.csv --tgt_path data/NEW_MOVIES.csv
```

#### 2 Train Model
To train model set options
1. movie_path : Path that has 'movie infomation csv' file which is generated from "Scrap.py"
2. view_path : Path that has 'SKB_DLP_VIEWS.csv' file
3. question_path : Path that has 'SKB_DLP_QUESTION.csv' file
4. batch_size : Batch size of training
5. window_size : Sequence size that will loaded for training
6. test_portion : Test portion that split dataset into train, valid dataset
7. hidden_size : Hidden size of RNN model
8. word_vec_dim : Embedding size of movieID
9. n_epochs : Max epoch to train
10. early_stop : Early stop condtion If there is no progress after epochs
11. target : Folder path where result model saved

##### EXAMPLE
```python
python train.py
```

#### 3 Test with trained model
To test model set options




