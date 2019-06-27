from argparse import ArgumentParser
import torch
import torch.nn as nn
from datahandler.data_loader import btvDataLoader
import logging
from tools import LOGFILE_LEVEL, CONSOLE_LEVEL
from tools import OPTIONS, optionStruct

def build_config():
    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", metavar="MODE", default="train",
                        help="Choose Mode : train, evaluate, generate")
    parser.add_argument("--movie_path", dest="movie_path", default='data/NEW_MOVIES.csv')
    parser.add_argument("--view_path", dest="view_path", default='data/SKB_DLP_VIEWS.csv')
    parser.add_argument("--question_path", dest="question_path", default='data/SKB_DLP_QUESTION.csv')
    parser.add_argument("--batch_size", dest="batch_size", default="32")
    parser.add_argument("--window_size", dest="window_size", default="10")
    parser.add_argument("--device", dest="device", default="gpu")
    parser.add_argument("--test_portion", dest="test_portion", default="0.05")
    parser.add_argument("--model", dest="model", default="SeqModel3")
    parser.add_argument("--hidden_size", dest="hidden_size", default=256)
    parser.add_argument("--word_vec_dim", dest="word_vec_dim", default=256)
    parser.add_argument("--lr", dest="lr", default=1.0)
    parser.add_argument("--n_epochs", dest="n_epochs", default=18)
    parser.add_argument("--early_stop", dest="early_stop", default=-1)
    parser.add_argument("--target", dest="target", default="target")

    config = parser.parse_args()
    return config



if __name__ == '__main__':
    config = build_config()

    ## gpu사용할 것인지 cpu사용할 것인지 확인.
    config.device = torch.device("cuda" if torch.cuda.is_available() and (config.device == 'gpu' or config.device == 'cuda') else "cpu")

    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    _print_config(config)

    if not logging.getLogger() == None:
        for handler in logging.getLogger().handlers[:]:  # make a copy of the list
            logging.getLogger().removeHandler(handler)

    logging.basicConfig(filename='train_log', level=LOGFILE_LEVEL)  # logging의 config 변경
    console = logging.StreamHandler()  # logging을 콘솔화면에 출력
    console.setLevel(CONSOLE_LEVEL)  # log level 설정
    logging.getLogger().addHandler(console)  # logger 인스턴스에 콘솔창의 결과를 핸들러에 추가한다.

    options = optionStruct(**OPTIONS)

    # Load training and validation data set.
    loader = btvDataLoader(movie_path=config.movie_path,
                           view_path=config.view_path,
                           question_path=config.question_path,
                           batch_size=config.batch_size,
                           window_size=config.window_size,
                           device=config.device,
                           test_portion=config.test_portion,
                           options = options
                           )
    train, valid = loader.load_train_data()

    options.set_movie_size(loader.get_movie_size())
    options.set_nation_size(loader.get_nation_size())
    options.set_genre_size(loader.get_genre_size())

    if config.model == 'SeqModel':
        from model.model import SeqModel
        from trainer import Trainer

        model = SeqModel(
            input_size=options.get_input_size(),
            hidden_size=config.hidden_size,
            output_size=options.get_movie_size(),
            n_layers=4,
            device=config.device
        ).to(config.device)

        crits = nn.NLLLoss()
        trainer = Trainer(model, crits, config, options)
    elif config.model == 'SeqModel3':
        from model.model import SeqModel3
        from trainer.sequence_trainer import seq_Trainer
        model = SeqModel3(
            input_size=options.get_movie_size(),
            word_vec_dim=config.word_vec_dim,
            hidden_size=config.hidden_size,
            output_size=options.get_movie_size(),
            device='cpu'
        ).to(config.device)

        crits = nn.NLLLoss()
        trainer = seq_Trainer(model, crits, config, options)

    trainer.train(train, valid)














