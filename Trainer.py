import pytorch_lightning as pl
import torch


class PL_Trainer(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
#         # freeze layers
#         for p in self.model.parameters():
#             p.requires_grad = False
#         # unfreeze N decoder layers
#         for p in self.model.model.decoder.layers[:].parameters():
#             p.requires_grad = True

    
    def forward(self, batch): 
        ids = batch['input_ids'].to(dtype = torch.long)#.to(model.device)
        mask = batch['attention_mask'].to(dtype = torch.long)#.to(model.device)
        labels = batch['labels'].to(dtype = torch.long)#.to(model.device)
        loss, tr_logits = self.model(input_ids=ids, attention_mask=mask, labels=labels,
                               return_dict=False)
        return loss, tr_logits        
    
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": 'val_loss',
        }
    
    def training_step(self, batch, batch_idx):
        ids = batch['input_ids'].to(dtype = torch.long)
        mask = batch['attention_mask'].to(dtype = torch.long)
        labels = batch['labels'].to(dtype = torch.long)
        output = self.model(input_ids=ids, attention_mask=mask, labels=labels,
                                 return_dict=False)
        loss, y_hat = output[0], output[1]
        self.log("train_loss", loss)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx): 
        ids = batch['input_ids'].to(dtype = torch.long)
        mask = batch['attention_mask'].to(dtype = torch.long)
        labels = batch['labels'].to(dtype = torch.long)
        output = self.model(input_ids=ids, attention_mask=mask, labels=labels,
                                 return_dict=False)
        loss, y_hat = output[0], output[1]
        self.log("val_loss", loss)
        return {"val_loss": loss}