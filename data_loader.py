import os
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import logging
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader

class btvDataLoader():

    def __init__(self,
                 movie_path=None,
                 view_path=None,
                 question_path=None,
                 options = None,
                 batch_size=64,
                 window_size=10,
                 device='cpu',
                 shuffle=True,
                 test_portion=0.05
                 ):
        super(btvDataLoader, self).__init__()

        logging.info("##init DataLoader")

        self.movie_path = movie_path
        self.view_path = view_path
        self.question_path = question_path
        self.batch_size = int(batch_size)
        self.window_size = int(window_size)
        self.device = device
        self.shuffle = shuffle
        self.test_portion = float(test_portion)
        self.options = options

        self.data = None
        self.train = None
        self.valid = None
        self.test = None

        ## Load data
        if self.movie_path:
            logging.debug("Load data : " + str(self.movie_path))
            self.movies_df = self.load_csv(self.movie_path)

        if self.view_path:
            logging.debug("Load data : " + str(self.view_path))
            self.views_df = self.load_csv(self.view_path)

        if self.question_path:
            logging.debug("Load data : " + str(self.question_path))
            self.questions_df = self.load_csv(self.question_path)

        self.movie_size = None
        self.extra_size = None
        self.nation_size = None
        self.genre_size = None

    def get_movie_size(self):
        return self.movie_size

    def get_extra_size(self):
        return self.extra_size

    def get_nation_size(self):
        return self.nation_size

    def get_genre_size(self):
        return self.genre_size

    def preprocess(self, views_df, movies_df):
        ##장르 보정
        def convert_genre(genre, genre_list):
            for index, item in enumerate(genre_list):
                if genre == item:
                    return index
            return len(genre_list) - 1

        genre_list = movies_df['genre'].unique().tolist()
        movies_df['genre'] = movies_df['genre'].apply(convert_genre, args=(genre_list,))

        ##국가 보정
        def convert_nation(genre, nation_list):
            for index, item in enumerate(nation_list):
                if genre == item:
                    return index
            return len(nation_list) - 1

        nation_list = movies_df['nation'].unique().tolist()
        movies_df['nation'] = movies_df['nation'].apply(convert_nation, args=(nation_list,))

        ##MOVIE_DF와 VIEW_DF정보 합치기
        result_df = pd.merge(views_df, movies_df, on='MOVIE_ID')

        ## 정렬
        result_df = result_df.sort_values(by=['USER_ID', 'WATCH_DAY', 'WATCH_SEQ'], ascending=True)

        ##SEQ 보정
        def convert_seq(seq):
            if seq > 1:
                return 1
            return 0

        result_df['WATCH_SEQ'] = result_df['WATCH_SEQ'].apply(convert_seq)

        self.nation_size = len(nation_list)
        self.genre_size = len(genre_list)
        self.movie_size = len(movies_df['MOVIE_ID'].unique().tolist())

        return result_df



    def load_train_data(self):
        assert self.movie_path, "movie_path 가 없습니다."
        assert self.view_path, "view_path 가 없습니다."

        logging.info("call load_train_data with window_size : " + str(self.window_size))

        ## Load data
        movies_df = self.movies_df
        views_df = self.views_df

        ## 처리
        result_df = self.preprocess(views_df, movies_df)

        ## 데이터 생성
        logging.info('generate Data')
        # selected_column = ['MOVIE_ID', 'WATCH_SEQ', 'score', 'nation', 'genre']
        selected_column = self.options.get_name()

        raw_data = result_df[selected_column].values.tolist()
        iterator = result_df['USER_ID'].values.tolist()
        ##TEST용
        progress_bar = tqdm(iterator, desc='DATA_LOADING: ', unit='user')
        # progress_bar = tqdm(iterator, desc='DATA_LOADING: ', unit='user')
        datas = []
        labels = []
        temp_user_id = -1
        temp_count = 0
        for index, user_id in enumerate(progress_bar):
            if not temp_user_id == user_id:
                temp_count = 0
                temp_user_id = user_id

            temp_count += 1

            if temp_count > self.window_size:
                datas.append(raw_data[index - self.window_size:index])
                labels.append(raw_data[index])

        progress_bar.close()

        dataset = SequenceDataset(datas, labels)

        test_portion = self.test_portion
        test_size = int(test_portion * len(dataset))
        train_size = len(dataset) - test_size
        train, valid = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.valid = DataLoader(valid, batch_size=self.batch_size, shuffle=False)

        return self.train, self.valid

    def load_test_data(self):
        assert self.movie_path, "movie_path 가 없습니다."
        assert self.question_path, "question_path 가 없습니다."

        logging.info("call load_test_data with window_size : " + str(self.window_size))

        movies_df = self.movies_df
        questions_df = self.questions_df

        ## 정렬
        result_df = self.preprocess(questions_df, movies_df)

        ## 데이터 생성
        logging.info('generate Data')
        # selected_column = ['MOVIE_ID', 'WATCH_SEQ', 'score', 'nation', 'genre']
        selected_column = InputFeatures().get_columns()

        raw_data = result_df[selected_column].values.tolist()
        iterator = result_df['USER_ID'].values.tolist()

        progress_bar = tqdm(iterator, desc='DATA_LOADING: ', unit='user')
        # progress_bar = tqdm(iterator, desc='DATA_LOADING: ', unit='user')
        datas = []
        temp_user_id = -1
        temp_count = 0
        for index, user_id in enumerate(progress_bar):
            if not temp_user_id == user_id:
                temp_count = 0
                temp_user_id = user_id

            temp_count += 1

            if temp_count == self.window_size:
                datas.append(raw_data[index - self.window_size +1:index+1])
            elif temp_count > self.window_size:
                print("박살낫는데?")

        print(torch.tensor(datas).size())
        progress_bar.close()

        dataset = SequenceDataset(datas)

        self.test = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        return self.test


    def load_csv(self, path):
        return pd.read_csv(path, encoding='euc_kr')


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 movie_id=None,
                 seq=None,
                 score=None,
                 nation=None,
                 genre=None):
        self.movie_id = movie_id
        self.seq = seq
        self.score = score
        self.nation = nation
        self.genre = genre

        self.column_list = ['MOVIE_ID', 'WATCH_SEQ', 'score', 'nation', 'genre']

    def get_columns(self):
        return self.column_list


class sourceFeatures(object):
    """A single set of features of data."""

    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt


class SequenceDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, datas, labels=None):
        self.datas = datas
        self.labels = labels
        self.len = len(datas)

    def __getitem__(self, index):
        if not self.labels:
            return torch.tensor(self.datas[index])
        else:
            return torch.tensor(self.datas[index]), torch.tensor(self.labels[index])

    def __len__(self):
        return self.len





