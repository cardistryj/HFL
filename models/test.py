import torch.nn.functional as F
from torch.utils.data import DataLoader

class Tester:
    def __init__(self, args) -> None:
        self.args = args
    
    def set_model(self, model):
        self.model = model
    
    def run_test(self, dataset):
        self.model.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(dataset, batch_size=self.args.test_bs, num_workers=self.args.num_dataset_workers)
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != '-1':
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = correct / len(data_loader.dataset)
        return accuracy.item(), test_loss

