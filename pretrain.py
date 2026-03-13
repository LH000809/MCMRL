import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.nt_xent import NTXentLoss
from dataset.dataset import MoleculeDatasetWrapper


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    # print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class MolCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, data_3d, datafp, n_iter):
        # get the representations and the projections
        zis, zjs = model(data, data_3d, datafp)  # [N,C]

        # get the representations and the projections
        # rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        if self.config['model_type'] == 'gin':
            from models.gin_pretrain import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        else:
            raise ValueError('Undefined GNN model.')
        print(model)
        
        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=eval(self.config['weight_decay'])
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'], 
            eta_min=0, last_epoch=-1
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, (data, data_3d, datafp) in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                data_3d = data_3d.to(self.device)
                datafp = datafp.to(self.device)
                
                loss = self._step(model, data, data_3d, datafp, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(epoch_counter, bn, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
            
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        if self.config['load_model'] is None:
            print("No checkpoint specified. Training from scratch.")
            return model
            
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            checkpoint_path = os.path.join(checkpoints_folder, 'model.pth')
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load state dict with map_location to handle device differences
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
            # Load state dict with strict=False to allow partial loading if needed
            model.load_state_dict(state_dict, strict=False)
            
            print(f"Loaded pre-trained model from: {checkpoint_path}")
            print("Loaded pre-trained model with success.")
        except FileNotFoundError as e:
            print(f"Pre-trained weights not found: {e}")
            print("Training from scratch.")
        except Exception as e:
            print(f"Error loading pre-trained weights: {e}")
            print("Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (data, data_3d, datafp) in valid_loader:
                data = data.to(self.device)
                data_3d = data_3d.to(self.device)
                datafp = datafp.to(self.device)

                loss = self._step(model, data, data_3d, datafp, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    molclr = MolCLR(dataset, config)
    molclr.train()


if __name__ == "__main__":
    main()
