import os
import pandas as pd
import logging
ENCODING = 'euc_kr'
CONSOLE_LEVEL = logging.INFO
LOGFILE_LEVEL = logging.DEBUG


def load_csv(path):
    return pd.read_csv(path, encoding=ENCODING)

OPTIONS = {
    'USER_ID' : False,
    'MOVIE_ID' : True,
    'WATCH_SEQ' : True,
    'TITLE' : False,
    'RELEASE_MONTH' : False,
    'score' : False,
    'participa' : False,
    'nation' : False,
    'genre': False,
    'showtime': False,
    'exist': False
}

class optionStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def set_movie_size(self, size):
        self.movie_size = size

    def get_movie_size(self):
        return self.movie_size

    def set_nation_size(self, size):
        self.nation_size = size

    def get_nation_size(self):
        return self.nation_size

    def set_genre_size(self, size):
        self.genre_size = size

    def get_genre_size(self):
        return self.genre_size

    def get_name(self):
        return [name for name, value in self.__dict__.items() if value == True]

    def get_input_size(self):
        size = 0
        if hasattr(self, 'movie_size') and hasattr(self, 'MOVIE_ID') and self.MOVIE_ID:
            size += self.movie_size
        if hasattr(self, 'nation_size') and hasattr(self, 'nation') and self.nation:
            size += self.nation_size
        if hasattr(self, 'genre_size') and hasattr(self, 'genre') and self.genre:
            size += self.genre_size

        if hasattr(self, 'score') and self.score:
            size += 1
        if hasattr(self, 'WATCH_SEQ') and self.WATCH_SEQ:
            size += 1
        if hasattr(self, 'RELEASE_MONTH') and self.RELEASE_MONTH:
            size += 1
        if hasattr(self, 'participa') and self.participa:
            size += 1
        if hasattr(self, 'showtime') and self.showtime:
            size += 1
        if hasattr(self, 'exist') and self.exist:
            size += 1

        return size