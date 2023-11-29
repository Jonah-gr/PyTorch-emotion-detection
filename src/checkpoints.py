import torch
from device import DEVICE
import json
from model import Network
from trainloop import Trainer
from label_and_dir import label_and_dir
from data_generator import data_loader

class CheckpointTrainer(Trainer):
    def __init__(self, network, loss_function, chkpt_path):
      super().__init__(network, loss_function)

      self.ep = 0
      self.chkpt_path = chkpt_path
      self.best_val_acc = 0
      try:
          chkpt = torch.load(self.chkpt_path)
          self.network.load_state_dict(chkpt["net_state_dict"])
          self.optim.load_state_dict(chkpt["optim_state_dict"])
          self.scheduler.load_state_dict(chkpt["scheduler_state_dict"])
          self.best_val_acc = chkpt["best_val_acc"]
          self.ep = chkpt["epoch"]
      except:
          print("Could not find checkpoint, starting from scratch")



    def train(self, loader_train, loader_val):
      while True:
        train_loss, train_acc = self.epoch(loader_train, True, self.ep)
        if loader_val is not None:
          val_loss, val_acc = self.epoch(loader_val, False, self.ep)
        else:
           val_loss, val_acc = train_loss, train_acc
        self.scheduler.step()
        
        self.ep += 1

        self.logger( {
           "epoch": self.ep,
           "training": { "loss": train_loss, "accuracy": train_acc },
           "validation": { "loss": val_loss, "accuracy": val_acc }
          }
        )

        self.best_val_acc = val_acc
        torch.save({
            "net_state_dict": self.network.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "epoch": self.ep
        }, self.chkpt_path)

    def logger(self, statistics):
       print(json.dumps(statistics, indent=3))



if __name__ == "__main__":
    # Assuming you have set up your data loaders and DEVICE
    train_dir, valid_dir, test_dir, train_label, valid_label, test_label = label_and_dir()
    # Assuming you have set up your data loaders and DEVICE
    train_loader, valid_loader, test_loader = data_loader(train_dir, valid_dir, test_dir, train_label, valid_label, test_label)

    net = Network().to(DEVICE)  # Set the appropriate number of classes
    loss = torch.nn.CrossEntropyLoss(reduction="none")

    trainer = Trainer(net, loss)
    trainer.epoch(train_loader, net, True)

    trainer = CheckpointTrainer(net, loss, "model.pt")
    trainer.train(train_loader, valid_loader)