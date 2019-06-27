from trainer.trainer import Trainer
import torch
from tqdm import tqdm

class step_Trainer(Trainer):
    def __init__(self, model, crit, config, options):
        super().__init__(model, crit, config, options)

    def get_movie(self, options, labels):
        names = options.get_name()
        for idx, name in enumerate(names):
            if name == 'MOVIE_ID':
                return labels[:, :, idx].type(torch.long).to(self.device)

        return labels[:, :, 0].type(torch.long).to(self.device)

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

            x = self.get_movie(self.options, batch)

            optimizer.zero_grad()

            y_hat = self.model.search(x)
            ## |y_hat| = (batch_size, length, output_size)

            # Calcuate loss and gradients with back-propagation.
            y = self.get_movie(self.options, labels)[:, -1]

            loss = self.crit(y_hat, y)
            loss.backward()

            # Simple math to show stats.
            # Don't forget to detach final variables.

            total_loss += float(loss)
            total_word_count += int(batch.size(1))
            avg_loss = total_loss / total_word_count
            pred = y_hat.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(y.view_as(pred)).sum().item()

            avg_correct = total_correct / total_word_count
            progress_bar.set_postfix_str('avg_loss=%.4e  correct=%.4f' % (avg_loss, avg_correct))

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
                x = self.get_movie(self.options, batch)

                y_hat = self.model.search(x)

                # Calcuate loss and gradients with back-propagation.
                y = self.get_movie(self.options, labels)[:, -1]

                loss = self.crit(y_hat, y)

                total_loss += float(loss)
                total_word_count += int(batch.size(1))
                avg_loss = total_loss / total_word_count
                pred = y_hat.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(y.view_as(pred)).sum().item()

                avg_correct = total_correct / total_word_count
                progress_bar.set_postfix_str('avg_loss=%.4e  correct=%.4f' % (avg_loss, avg_correct))

            progress_bar.close()
        self.model.train()
        return avg_loss, avg_correct

