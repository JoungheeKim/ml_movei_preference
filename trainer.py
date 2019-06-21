import os
import re
import torch
from tqdm import tqdm
import torch.optim as optim
import logging
from torch import nn

class Trainer():
    def __init__(self, model, crit, config, options):
        self.model = model
        self.crit = crit
        self.config = config
        self.device = config.device
        self.options = options

        super().__init__()

        self.n_epochs = config.n_epochs
        self.lower_is_better = True
        self.best = {'epoch': 0,
                     'config': config,
                     'options': options
                     }
        logging.info("###init Trainer")

        path = os.path.join(os.getcwd(), 'model')
        if not os.path.isdir(path):
            os.mkdir(path)
        self.model_path = os.path.join(path, "model")

        ##options
        if hasattr(options, 'movie_size'):
            size = options.get_movie_size()
            emb = nn.Embedding(size, size)
            emb.weight.data = torch.eye(size)
            emb.weight.requires_grad = False
            self.movie_emb = emb

        if hasattr(options, 'nation_size'):
            size = options.get_nation_size()
            emb = nn.Embedding(size, size)
            emb.weight.data = torch.eye(size)
            emb.weight.requires_grad = False
            self.nation_emb = emb

        if hasattr(options, 'genre_size'):
            size = options.get_genre_size()
            emb = nn.Embedding(size, size)
            emb.weight.data = torch.eye(size)
            emb.weight.requires_grad = False
            self.genre_emb = emb

    def get_best_model(self):
        self.model.load_state_dict(self.best['model'])
        return self.model

    def save_training(self, path):
        torch.save(self.best, path)

    def _get_loss(self, y_hat, y, crit=None):
        # |y_hat| = (batch_size, length, output_size)
        # |y| = (batch_size, length)
        crit = self.crit if crit is None else crit    ##NLL loos를 사용했을꺼셈.....
        loss = crit(y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1)
                    )

        return loss

    def train(self, train, valid):
        '''
        Train with given train and valid iterator until n_epochs.
        If early_stop is set,
        early stopping will be executed if the requirement is satisfied.
        '''

        logging.info("run train")
        best_loss = float('Inf') * (1 if self.lower_is_better else -1)
        lowest_after = 0

        progress_bar = tqdm(range(self.best['epoch'], self.n_epochs),
                            desc='Training: ',
                            unit='epoch'
                            )

        optimizer = optim.Adam(self.model.parameters())
        for idx in progress_bar:  # Iterate from 1 to n_epochs
            avg_train_loss = self.train_epoch(train, optimizer)
            avg_valid_loss = self.validate(valid)
            progress_bar.set_postfix_str('train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (avg_train_loss,
                                                                                                  avg_valid_loss,
                                                                                                  best_loss))
            logging.debug('train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (avg_train_loss,
                                                                                   avg_valid_loss,
                                                                                   best_loss))

            if (self.lower_is_better and avg_valid_loss < best_loss) or \
               (not self.lower_is_better and avg_valid_loss > best_loss):
                # Update if there is an improvement.
                best_loss = avg_valid_loss
                lowest_after = 0

                self.best['model'] = self.model.state_dict()
                self.best['epoch'] = idx + 1

                # Set a filename for model of last epoch.
                # We need to put every information to filename, as much as possible.

                self.save_training(self.model_path + str(self.best['epoch']) + '.pwf')
            else:
                lowest_after += 1

                if lowest_after >= self.config.early_stop and \
                   self.config.early_stop > 0:
                    logging.debug("early stop")
                    break
        progress_bar.close()

    def convert_data(self, options, batch):
        names = options.get_name()
        temp_tensor = []
        for idx, name in enumerate(names):
            temp = None
            if name == 'MOVIE_ID':
                temp = batch[:, :, idx].type(torch.long)
                temp = self.movie_emb(temp).type(torch.float)
            elif name == 'nation':
                temp = batch[:, :, idx].type(torch.long)
                temp = self.nation_emb(temp).type(torch.float)
            elif name == 'genre':
                temp = batch[:, :, idx].type(torch.long)
                temp = self.genre_emb(temp).type(torch.float)
            elif name in ['WATCH_SEQ', 'score', 'participa', 'showtime', 'exist']:
                temp = batch[:, :, idx].unsqueeze(2).type(torch.float)

            temp_tensor.append(temp)

        temp_tensor = torch.cat(temp_tensor, dim=2).to(self.device)
        return temp_tensor

    def get_movie(self, options, labels):
        names = options.get_name()
        for idx, name in enumerate(names):
            if name == 'MOVIE_ID':
                return labels[:, idx].type(torch.long).to(self.device)

        return labels[:, 0].type(torch.long).to(self.device)



    def train_epoch(self, train, optimizer):
        '''
        Train an epoch with given train iterator and optimizer.
        '''
        total_loss, total_word_count = 0, 0
        total_correct = 0
        avg_loss = 0
        avg_correct = 0


        progress_bar = tqdm(train,
                            desc='Training: ',
                            unit='batch'
                            )
        # Iterate whole train-set.
        for idx, (batch, labels) in enumerate(progress_bar):

            x = self.convert_data(self.options, batch)

            optimizer.zero_grad()

            y_hat = self.model(x)
            ## |y_hat| = (batch_size, length, output_size)

            # Calcuate loss and gradients with back-propagation.
            y = self.get_movie(self.options, labels)
            loss = self.crit(y_hat, y)
            loss.backward()

            # Simple math to show stats.
            # Don't forget to detach final variables.
            total_loss += float(loss)
            total_word_count += int(batch.size(1))
            avg_loss = total_loss / total_word_count

            ps = torch.exp(y_hat)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y.view(*top_class.shape)
            total_correct += torch.sum(equals)

            avg_correct = total_correct / total_word_count
            progress_bar.set_postfix_str('avg_loss=%.4e  correct=%.4e' % (avg_loss, avg_correct))

            # Take a step of gradient descent.
            optimizer.step()

        progress_bar.close()

        return avg_loss

    def validate(self, valid):
        total_loss, total_word_count = 0, 0
        total_correct = 0
        avg_loss = 0
        avg_correct = 0

        self.model.eval()
        with torch.no_grad():
            total_loss, total_word_count = 0, 0
            progress_bar = tqdm(valid, desc='Validation: ', unit='batch')
            # Iterate for whole valid-set.
            for idx, (batch, labels) in enumerate(progress_bar):
                x = self.convert_data(self.options, batch)

                y_hat = self.model(x)

                y = self.get_movie(self.options, labels)
                loss = self.crit(y_hat, y)

                total_loss += float(loss)
                total_word_count += int(batch.size(1))
                avg_loss = total_loss / total_word_count

                ps = torch.exp(y_hat)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == y.view(*top_class.shape)
                total_correct += torch.sum(equals)

                avg_correct = total_correct / total_word_count
                progress_bar.set_postfix_str('avg_loss=%.4e  correct=%.4e' % (avg_loss, avg_correct))

            progress_bar.close()
        self.model.train()
        return avg_loss

    def test(self, test):
        total_loss, total_word_count = 0, 0
        avg_loss = 0

        result = []

        with torch.no_grad():
            total_loss, total_word_count = 0, 0

            progress_bar = tqdm(test, desc='test: ', unit='batch')

            self.model.eval()
            # Iterate for whole valid-set.
            for idx, batch in enumerate(progress_bar):
                movie = batch[:, :, 0].type(torch.long).to(self.device)
                nation = batch[:, :, 3].type(torch.long).to(self.device)
                genre = batch[:, :, 4].type(torch.long).to(self.device)

                seq = batch[:, :, 1].unsqueeze(2).to(self.device)
                score = batch[:, :, 2].unsqueeze(2).to(self.device)
                extra = torch.cat([seq, score], dim=2).to(self.device)

                y_hat = self.model(movie, nation, genre, extra)
                result.append(y_hat)

            progress_bar.close()

        return result
