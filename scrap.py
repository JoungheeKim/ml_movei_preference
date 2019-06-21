import os
import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
import webbrowser
import urllib.parse
import pandas as pd
from tools import load_csv, ENCODING
import logging


class Scrapper():

    def __init__(self,
                 src_path=None,
                 tgt_path=None
                 ):
        super(Scrapper, self).__init__()

        self.src_path = src_path
        self.tgt_path = tgt_path


    def scrap_naver_data(self):
        logging.info("########## Scrapper[START] - scrap_naver_data ##########")
        assert self.src_path, "src_path 가 없음"
        assert self.tgt_path, "tgt_path 가 없음"

        movie_df = load_csv(self.src_path)
        movie_df['score'], movie_df['participa'], movie_df['nation'], movie_df['genre'], movie_df['showtime'], movie_df['exist'] = zip(
            *movie_df['TITLE'].apply(get_more_info))

        movie_df.to_csv(self.tgt_path, encoding=ENCODING)
        logging.info("########## Scrapper[END] - scrap_naver_data ##########")

def make_url(root_url, options):
    return root_url + urllib.parse.urlencode(options)

def clean_blacket(text):
    ret = ''
    skip1c = 0
    skip2c = 0
    skip3c = 0
    for i in text:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == '<':
            skip3c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')' and skip2c > 0:
            skip2c -= 1
        elif i == '>' and skip3c > 0:
            skip3c -= 1
        elif skip1c == 0 and skip2c == 0 and skip3c == 0:
            ret += i

    ret = ret.replace("예고편:", '')
    ret = ret.replace("디지털특별판", '')
    ret = ret.replace("영화제수상작", '')
    ret = ret.replace("싱어롱추가팩", '')
    ret = ret.replace("패키지", '')
    if len(ret.split(':'))>1:
        ret = ret.split(':')[0]
    return ret


def get_more_info(movie_name):
    root_url = 'https://movie.naver.com/movie/search/result.nhn?'
    options = {
        "section": 'movie',
        'query': '동갑내기과외하기레슨2',
        'ie': 'utf8'
    }
    options['query'] = clean_blacket(movie_name)
    url = make_url(root_url, options)
    param_list = {
        "nation=": "NN",
        "genre=": '99'
    }
    try:
        with urlopen(url) as response:
            html = response.read()
            soup = BeautifulSoup(html, 'html5lib')
            ul = soup.find("ul", {"class": "search_list_1"})

            ##점수
            score = ul.find('li').find('dd', {'class': 'point'}).find('em', {'class': 'num'}).text

            ##참여자수
            participation = ul.find('li').find('dd', {'class': 'point'}).find('em', {'class': 'cuser_cnt'}).text
            participation = re.findall("\d+", participation)[0]

            etc_dds = ul.find('li').find('dd', {'class': 'etc'}).find_all('a', href=True)

            ##국가, 장르
            for etc in etc_dds:
                etc = etc['href']
                for key, value in param_list.items():
                    items = str(etc).split(key)
                    if len(items) > 1:
                        param_list[key] = items[-1]

            ##상영 시간
            time_text = ul.find('li').find('dd', {'class': 'etc'}).text
            time_text = str(time_text).split("|")

            showtime = 0
            for time in time_text:
                if "분" in time:
                    showtime = re.findall("\d+", time)[0]

        ## return type : 점수, 참여자수, 국가, 장르, 상영시간, 조회(boolean)
        return score, participation, param_list['nation='], param_list['genre='], showtime, 1
    except Exception as e:
        print('error', options['query'])
        return '0', '0', param_list['nation='], param_list['genre='], 0, 0
        
if __name__ == '__main__':
    ##PATH
    src_path= os.path.join('data', 'SKB_DLP_MOVIES.csv')
    tgt_path = os.path.join('data', 'NEW_MOVIES.csv')

    ##LOGGER
    if not logging.getLogger() == None:
        for handler in logging.getLogger().handlers[:]:  # make a copy of the list
            logging.getLogger().removeHandler(handler)
    console_level = logging.INFO
    logfile_level = logging.DEBUG
    logging.basicConfig(filename="scrap_log", level=logfile_level)  # logging의 config 변경
    console = logging.StreamHandler()  # logging을 콘솔화면에 출력
    console.setLevel(console_level)  # log level 설정
    logging.getLogger().addHandler(console)  # logger 인스턴스에 콘솔창의 결과를 핸들러에 추가한다.


    scrapper = Scrapper(src_path=src_path, tgt_path=tgt_path)
    scrapper.scrap_naver_data()




