from argparse import ArgumentParser
import sys
import torch
import torch.nn as nn
from data_loader import btvDataLoader
import logging
from tools import LOGFILE_LEVEL, CONSOLE_LEVEL


def build_config():
    parser = ArgumentParser()

    parser.add_argument("--model_path", dest="model_path", metavar="model_path", default="model/model1.pwf")
    parser.add_argument("--device", dest="device", default="gpu")
    parser.add_argument("--question_path", dest="question_path", default='data/SKB_DLP_QUESTION.csv')
    parser.add_argument("--movie_path", dest="movie_path", default='data/NEW_MOVIES.csv')
    parser.add_argument("--batch_size", dest="batch_size", default=32)
    parser.add_argument("--n_epochs", dest="n_epochs", default=18)
    parser.add_argument("--test_num", dest="test_num", default=5)
    parser.add_argument("--model", dest="model", default="SeqModel3")
    config = parser.parse_args()
    return config

def load_training(path):
    checkpoint = torch.load(path)
    return checkpoint

def _print_config(config):
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(config))


if __name__ == "__main__":
    config = build_config()
    ## gpu사용할 것인지 cpu사용할 것인지 확인.
    config.device = torch.device("cuda" if torch.cuda.is_available() and (config.device == 'gpu' or config.device == 'cuda') else "cpu")

    if not logging.getLogger() == None:
        for handler in logging.getLogger().handlers[:]:  # make a copy of the list
            logging.getLogger().removeHandler(handler)

    logging.basicConfig(filename='test_log', level=LOGFILE_LEVEL)  # logging의 config 변경
    console = logging.StreamHandler()  # logging을 콘솔화면에 출력
    console.setLevel(CONSOLE_LEVEL)  # log level 설정
    logging.getLogger().addHandler(console)  # logger 인스턴스에 콘솔창의 결과를 핸들러에 추가한다.

    logging.info("loading saved data [" + str(config.model_path) + "]")
    checkpoint = load_training(config.model_path)
    options = checkpoint['options']
    saved_config = checkpoint['config']
    loader = btvDataLoader(movie_path=config.movie_path,
                           view_path=None,
                           question_path=config.question_path,
                           batch_size=config.batch_size,
                           window_size=saved_config.window_size,
                           device=config.device
                           )

    test = loader.load_test_data()
    if config.model == "SeqModel3":
        from model import SeqModel3
        from sequence_trainer import seq_Trainer
        model = SeqModel3(
            input_size=options.get_movie_size(),
            word_vec_dim=saved_config.word_vec_dim,
            hidden_size=saved_config.hidden_size,
            output_size=options.get_movie_size(),
            device=config.device
        ).to(config.device)

        crits = nn.NLLLoss()
        trainer = seq_Trainer(model, crits, config, options)
    else:
        from model import SeqModel
        from trainer import Trainer

        model = SeqModel(
            input_size=options.get_input_size(),
            hidden_size=saved_config.hidden_size,
            output_size=options.get_movie_size(),
            n_layers=4,
            device=config.device
        ).to(config.device)

        crits = nn.NLLLoss()
        trainer = Trainer(model, crits, config, options)
    trainer.test(test)







