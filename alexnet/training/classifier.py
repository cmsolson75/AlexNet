import lightning as L
import torch
import torchmetrics


class ClassifierTrainingWrapper(L.LightningModule):
    def __init__(self, classifier, loss_fn, lr):
        super().__init__()
        self.classifier = classifier
        self.loss_fn = loss_fn
        self.lr = lr
        # Set classes from config
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.classifier(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.classifier(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Make Optimizer Fully Configurable
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 30], gamma=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
