import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig


class ClassifierTrainingWrapper(L.LightningModule):
    def __init__(self, classifier, loss_fn, cfg: DictConfig):
        super().__init__()
        self.classifier = classifier
        self.loss_fn = loss_fn
        self.optimizer_cfg = cfg.optimizer

        # Set classes from config
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.out_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.out_classes)

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
        opt_cfg = self.optimizer_cfg
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
        )

        sch_cfg = opt_cfg.get("scheduler", None)
        if sch_cfg is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=sch_cfg.milestones, gamma=sch_cfg.gamma
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
