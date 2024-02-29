from torch.utils.tensorboard import SummaryWriter

class TensorBoardWriter:
    _instance = None

    @classmethod
    def get_writer(cls, log_dir):
        if cls._instance is None:
            cls._instance = SummaryWriter(log_dir=log_dir)
        return cls._instance