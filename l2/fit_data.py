class FitData:
    def __init__(self):
        self.validation_losses = []
        self.train_losses = []
        self.best_train_loss = (-1, -10000)
        self.best_validation_loss = (-1, -10000)

    def add_validation_loss(self, loss):
        self.validation_losses.append(loss)
        #if (loss > )

    def add_train_loss(self, loss):
        pass
