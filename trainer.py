from utils import accuracy, AverageMeter

class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None, USE_CUDA=True, \
        unroll_length = 20, meta = False, val_dataset = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.iterations = 0
        self.USE_CUDA = USE_CUDA
        self.unroll_length = unroll_length
        self.stats = {}
        self.meta = meta
        self.val_dataset = val_dataset
        self.train_acc = AverageMeter()
        self.val_acc   = AverageMeter()
        
        self.reward_baseline = 0
        
    def run(self, epochs=1):
        for i in range(1, epochs + 1):
            self.train_acc.reset()
            self.val_acc.reset()
            self.train()
            self.val()
    
    def reward(self):
        self.baseline = 0.9 * self.baseline + 0.1 * self.val_acc.avg
        return self.val_acc.avg - self.baseline
    
    def train(self):
        self.model.train()
        
        for i, data in enumerate(self.dataset):
            batch_input, batch_target = data
            input_var = batch_input
            target_var = batch_target
            if self.USE_CUDA:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            batch_output = self.model(input_var)
            loss = self.criterion(batch_output, target_var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.meta and i % self.unroll_length == 0:
                self.val_acc.reset()
                reward = self.reward()
                self.optimizer.meta_update(reward)
        
        if self.meta:
            self.val_acc.reset()
            reward = self.reward()
            self.optimizer.meta_update(reward)
            
        self.iterations += i
    
    def val(self):
        self.model.val()
        with torch.no_grad():
            for i, data in enumerate(self.val_dataset):
                batch_input, batch_target = data
                input_var = batch_input
                target_var = batch_target
                if self.USE_CUDA:
                    input_var = input_var.cuda()
                    target_var = target_var.cuda()
                    
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                acc = accuracy(batch_output, target_var)
                self.val_acc.update(acc)
            