import torch
from tqdm import tqdm
from device import DEVICE
from label_and_dir import label_and_dir
from data_generator import data_loader
from models import Network


class Trainer:
  def __init__(self, network, loss_function):
    self.network = network
    self.loss_function = loss_function
    
    self.init_optimizer()
    self.init_scheduler()

  def init_optimizer(self):
    self.optim = torch.optim.Adam(self.network.parameters(), lr=0.0001)
    

  def init_scheduler(self):
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.95) # torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

  def report_top_loss(self, top_loss_samples, epoch):
    return None

  def epoch(self, dataloader, training, epoch=0):
    # We want a dedicated TQDM bar, so we can set the description after each step
    bar = tqdm(dataloader)
    
    # Tell the network whether we are training or evaluating (to disable DropOut)
    if training:
      self.network.train()
      name="train"
    else:
      self.network.eval()
      name="val"

    # This epoch starts
    total_loss = 0
    correct = 0
    cnt = 0

    top_loss_values = None
    top_loss_samples = None

    # Iterate over the whole epoch
    for batch, labels in bar:
      # If we are training, zero out the gradients in the network
      if training:
        self.optim.zero_grad()

      # Do one forward pass
      res = self.network(batch)

      # Reshape labels for processing
      labels = labels.reshape(-1)

      # Calculcate the (BCE)-Loss
      loss = self.loss_function(res, labels)
      if top_loss_values is None:
        top_loss_values = loss.detach()
        top_loss_samples = batch.detach()
      else:
        top_loss_values = torch.cat([top_loss_values, loss.detach()])
        top_loss_samples = torch.cat([top_loss_samples, batch.detach()])
        
        top_loss_values, indices = torch.sort(top_loss_values, descending=True)
        top_loss_values = top_loss_values[:batch.shape[0]]
        top_loss_samples = top_loss_samples[indices,:,:]

      loss = torch.mean(loss)
      # Sum the total loss
      total_loss += loss.item()

      # Count how many correct predictions we have (for accuracy)
      correct += torch.sum(torch.argmax(res, dim=1) == labels).item()

      # Count total samples processed
      cnt += batch.shape[0]

      # Update bar description

      bar.set_description(f"ep: {epoch:.0f} ({name}), loss: {1000.0*total_loss / cnt:.3f}, acc: {100.0*correct/cnt:.2f}%")

      # If we are training, do backward pass 
      if training:
          # Calculcate backward gradients
          loss.backward()

          # Step the optimizer
          self.optim.step()

    self.report_top_loss(top_loss_samples, epoch)

    return 1000.0 * total_loss / cnt, 100.0*correct/cnt

if __name__ == "__main__":
    # Assuming you have set up your data loaders and DEVICE
    train_dir, valid_dir, test_dir, train_label, valid_label, test_label = label_and_dir()
    # Assuming you have set up your data loaders and DEVICE
    train_loader, valid_loader, test_loader = data_loader(train_dir, valid_dir, test_dir, train_label, valid_label, test_label)

    net = Network().to(DEVICE)  # Set the appropriate number of classes
    loss = torch.nn.CrossEntropyLoss() # reduction="none")

    trainer = Trainer(net, loss)
    trainer.epoch(train_loader, net, True)