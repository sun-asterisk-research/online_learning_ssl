#  Import pytorch lightning
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch


class OnlineSegmentationModel(pl.LightningModule):

    def __init__(self, arch='', encoder_name='', in_channels=3, out_classes=1,
                 checkpoint_path='', mm_checkpoint_path='', mm_teacher_checkpoint_path='', labeled_dataloader=None,
                 unlabeled_dataloader=None, momentum=0.99, momentum_teacher=0.99, use_momentum=False, is_semi=False,
                 is_online=False, teacher=None, use_soft_label=True, **kwargs):
        super().__init__()

        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # max iou of origin model and momentum model
        self.m_max_iou = 0
        self.mm_max_iou = 0
        self.mm_teacher_max_iou = 0
        self.checkpoint_path = checkpoint_path
        self.mm_checkpoint_path = mm_checkpoint_path
        self.mm_teacher_checkpoint_path = mm_teacher_checkpoint_path

        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        self.momentum = momentum
        self.momentum_teacher = momentum_teacher
        self.teacher = teacher
        self.is_semi = is_semi
        self.is_online = is_online
        self.use_momentum = use_momentum
        self.use_soft_label = use_soft_label
        # Beta for unlabeled loss
        self.beta = 0.3

        if self.use_momentum:
            self.momentum_model = smp.create_model(
                arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
            )
            for param in self.momentum_model.parameters():
                param.requires_grad = False
        else:
            self.momentum_model = None
        # Non back-propagate into teacher
        if self.is_semi and self.teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False

    def forward(self, image):
        return self.model(image)

    def mm_forward(self, image):
        # normalize image here
        mask = self.momentum_model.cuda()(image)
        return mask

    def mm_teacher_forward(self, image):
        # normalize image here
        mask = self.teacher.cuda()(image)
        return mask

    def make_pseudo_label(self, image):
        logits = self.teacher.cuda()(image)
        prob_mask = logits.sigmoid()
        # Generate hard label
        pred_mask = (prob_mask > 0.5).float()
        if self.use_soft_label:
            return prob_mask
        return pred_mask

    @torch.no_grad()
    def _update_momentum_network(self):
        """Momentum update of the teacher model with student weight"""
        if self.use_momentum and self.momentum_model:
            for param_model, param_momentum_model in zip(self.model.parameters(), self.momentum_model.parameters()):
                param_momentum_model.data = param_momentum_model.data * self.momentum + param_model.data * (
                        1.0 - self.momentum)

    @torch.no_grad()
    def _update_teacher_network(self):
        """Momentum update of the teacher model with student weight"""
        if self.is_semi and self.is_online:
            for param_student, param_teacher in zip(self.model.parameters(), self.teacher.parameters()):
                param_teacher.data = param_teacher.data * self.momentum_teacher + param_student.data * (
                        1.0 - self.momentum)

    def compute_loss(self, image, mask):
        assert image.ndim == 4
        assert mask.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        result = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "loss": loss
        }
        return result

    def shared_step(self, batch, stage):
        # If not have unlabeled data, the batch is only have labeled data
        batch_labeled = batch

        if stage == 'train':
            batch_labeled = batch['labeled']

        # Batch of labeled
        l_image = batch_labeled["image"]
        l_mask = batch_labeled["mask"]

        l_result = self.compute_loss(l_image, l_mask)
        l_loss = l_result['loss']
        # Predefine for unlabeled loss
        u_result = l_result
        u_result['loss'] = 0
        u_loss = u_result['loss']

        # Batch of unlabeled
        if self.is_semi and stage == 'train':
            batch_unlabeled = batch['unlabeled']
            u_image = batch_unlabeled["image"]
            # Compute mask from unlabled image with teacher model in online learning
            u_mask = self.make_pseudo_label(batch_unlabeled["image"])
            u_result = self.compute_loss(u_image, u_mask)
            u_loss = u_result['loss']

        # Compute total loss
        loss = l_loss + self.beta * u_loss

        self.log("{}/labeled_loss".format(stage), l_loss)

        self.log("{}/unlabeled_loss".format(stage), u_loss)

        self.log("{}/loss".format(stage), loss)

        # Append loss to result
        result = l_result
        result['loss'] = loss

        if self.use_momentum:
            logits_mm_mask = self.mm_forward(l_image)

            prob_mm_mask = logits_mm_mask.sigmoid()
            pred_mm_mask = (prob_mm_mask > 0.5).float()
            mm_tp, mm_fp, mm_fn, mm_tn = smp.metrics.get_stats(pred_mm_mask.long(), l_mask.long(), mode="binary")

            result['mm_tp'] = mm_tp
            result['mm_fp'] = mm_fp
            result['mm_fn'] = mm_fn
            result['mm_tn'] = mm_tn

        if self.is_online and self.is_semi:
            logits_mm_mask = self.mm_teacher_forward(l_image)

            prob_mm_mask = logits_mm_mask.sigmoid()
            pred_mm_mask = (prob_mm_mask > 0.5).float()
            mm_tp, mm_fp, mm_fn, mm_tn = smp.metrics.get_stats(pred_mm_mask.long(), l_mask.long(), mode="binary")

            result['mmt_tp'] = mm_tp
            result['mmt_fp'] = mm_fp
            result['mmt_fn'] = mm_fn
            result['mmt_tn'] = mm_tn

        return result

    def shared_epoch_end(self, outputs, stage):
        if stage == 'train' and self.is_online and self.is_semi:
            # Update teacher network
            self._update_teacher_network()

        if stage == 'train' and self.use_momentum:
            self._update_momentum_network()

        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        per_image_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        mm_per_image_iou = 0
        mm_teacher_max_iou = 0

        if self.use_momentum:
            # aggregate step metics
            tp = torch.cat([x["mm_tp"] for x in outputs])
            fp = torch.cat([x["mm_fp"] for x in outputs])
            fn = torch.cat([x["mm_fn"] for x in outputs])
            tn = torch.cat([x["mm_tn"] for x in outputs])

            mm_per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        if self.is_online and self.is_semi:
            # aggregate step metics
            tp = torch.cat([x["mmt_tp"] for x in outputs])
            fp = torch.cat([x["mmt_fp"] for x in outputs])
            fn = torch.cat([x["mmt_fn"] for x in outputs])
            tn = torch.cat([x["mmt_tn"] for x in outputs])

            mm_teacher_max_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        if stage == 'valid':
            # Save best checkpoint
            if per_image_iou > self.m_max_iou:
                print('\nSave origin model with checkpoint loss = {}'.format(per_image_iou))
                torch.save(self.model, self.checkpoint_path)
                self.m_max_iou = per_image_iou

            if self.use_momentum:
                if mm_per_image_iou > self.mm_max_iou:
                    print('\nSave momentum model with checkpoint loss = {}'.format(mm_per_image_iou))
                    torch.save(self.momentum_model, self.mm_checkpoint_path)
                    self.mm_max_iou = mm_per_image_iou

            if self.is_online and self.is_semi:
                if mm_teacher_max_iou > self.mm_teacher_max_iou:
                    print('\nSave momentum teacher model with checkpoint loss = {}'.format(mm_teacher_max_iou))
                    torch.save(self.teacher, self.mm_teacher_checkpoint_path)
                    self.mm_teacher_max_iou = mm_teacher_max_iou

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_per_image_dice": per_image_dice,
            f"{stage}_mm_per_image_iou": mm_per_image_iou,
            f"{stage}_mm_teacher_max_iou": mm_teacher_max_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def train_dataloader(self):
        if self.is_semi:
            loaders = {"labeled": self.labeled_dataloader, "unlabeled": self.unlabeled_dataloader}
        else:
            loaders = {"labeled": self.labeled_dataloader}

        return loaders

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
