from torch import nn
import logging
import json
from datetime import datetime

class pytorch_loss:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger("Loss Info")
        self.logger.setLevel(logging.INFO)
        self.fh = logging.FileHandler("/log/train.log")
        self.fh.setLevel(logging.INFO)
        self.logger.addHandler(self.fh)

        # block user access to the pytorch network itself
        # network defined as private variable,
        # so Logging & History will be handled by this class, not the one user defined
        self.__IEOSPQMS = IEOSPQMS(**kwargs)

        self.n_epoch = 1
        self.n_batch = 1
        self.mode = "Train"

    def train(self):
        self.mode = "Train"

    def eval(self):
        self.mode = "Eval"

    def __call__(self, *args):
        loss = self.__IEOSPQMS(*args)
        self.logger.info(
			json.dumps({
				"type": self.mode.lower(),
				"epoch": self.n_epoch,
				"batch": self.n_batch,
				"@timestamp": datetime.utcnow().isoformat()[:-3]+'Z',
				"loss": loss.item()}
			)
        )
        return loss

    def backward(self):
        return self.__IEOSPQMS.backward()

    def n_epoch_increment(self):
        self.n_epoch += 1

    def n_batch_increment(self):
        self.n_batch += 1


IEOSPQMS = nn.MSELoss
