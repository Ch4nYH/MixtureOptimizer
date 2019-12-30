class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None, USE_CUDA=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.iterations = 0
        self.USE_CUDA = USE_CUDA

        self.stats = {}
        
    def run(self, epochs=1):
        for i in range(1, epochs + 1):
            self.train()
            
    def train(self):
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input, batch_target = data
            input_var = batch_input
            target_var = batch_target
            if self.USE_CUDA:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            def closure():
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                loss.backward()
                return loss
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
        self.iterations += i