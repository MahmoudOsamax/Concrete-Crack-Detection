import torch
import torch.nn as nn
import numpy as np
from .base_model import BaseModel
from .deepcrack_networks import define_deepcrack, BinaryFocalLoss
import logging
logger = logging.getLogger(__name__)
class DeepCrackModel(BaseModel):
    """Implements the DeepCrack model."""
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--display_sides', type=bool, default=False, help='Whether to display side outputs')
        parser.add_argument('--lambda_side', type=float, default=1.0, help='Weight for side output loss')
        parser.add_argument('--lambda_fused', type=float, default=1.0, help='Weight for fused loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['side', 'fused', 'total']
        self.display_sides = getattr(opt, 'display_sides', False)
        self.visual_names = ['image', 'label_viz', 'fused']
        if self.display_sides:
            self.visual_names += ['side1', 'side2', 'side3', 'side4', 'side5']
        self.model_names = ['G']

        # Ensure num_classes is defined
        num_classes = getattr(opt, 'num_classes', 1)  # Default to 1 for binary classification
        try:
            self.netG = define_deepcrack(opt.input_nc, 
                                       num_classes, 
                                       opt.ngf, 
                                       opt.norm,
                                       opt.init_type, 
                                       opt.init_gain, 
                                       self.gpu_ids)
            self.netG.to(self.device)
            logger.info("DeepCrack network initialized")
        except Exception as e:
            logger.error(f"Error initializing network: {str(e)}")
            raise
        self.softmax = torch.nn.Softmax(dim=1)

        if self.isTrain:
            if opt.loss_mode == 'focal':
                self.criterionSeg = BinaryFocalLoss()
            else:
                self.criterionSeg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0/3e-2).to(self.device))
            self.weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]
            self.optimizer = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=2e-4)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data."""
        try:
            self.image = input['image'].to(self.device)
            self.label = input['label'].to(self.device)
            self.image_paths = input['A_paths']
        except Exception as e:
            logger.error(f"Error in set_input: {str(e)}")
            raise

    def forward(self):
        """Run forward pass."""
        try:
            self.outputs = self.netG(self.image)
            self.label_viz = (self.label.float() - 0.5) / 0.5
            self.fused = (torch.sigmoid(self.outputs[-1]) - 0.5) / 0.5
            if self.display_sides:
                self.side1 = (torch.sigmoid(self.outputs[0]) - 0.5) / 0.5
                self.side2 = (torch.sigmoid(self.outputs[1]) - 0.5) / 0.5
                self.side3 = (torch.sigmoid(self.outputs[2]) - 0.5) / 0.5
                self.side4 = (torch.sigmoid(self.outputs[3]) - 0.5) / 0.5
                self.side5 = (torch.sigmoid(self.outputs[4]) - 0.5) / 0.5
        except Exception as e:
            logger.error(f"Error in forward: {str(e)}")
            raise

    def backward(self):
        """Calculate the loss."""
        try:
            lambda_side = self.opt.lambda_side
            lambda_fused = self.opt.lambda_fused
            self.loss_side = 0.0
            for out, w in zip(self.outputs[:-1], self.weight_side):
                self.loss_side += self.criterionSeg(out, self.label) * w
            self.loss_fused = self.criterionSeg(self.outputs[-1], self.label)
            self.loss_total = self.loss_side * lambda_side + self.loss_fused * lambda_fused
            self.loss_total.backward()
        except Exception as e:
            logger.error(f"Error in backward: {str(e)}")
            raise

    def optimize_parameters(self, epoch=None):
        """Update network weights."""
        try:
            self.forward()
            self.optimizer.zero_grad()
            self.backward()
            self.optimizer.step()
        except Exception as e:
            logger.error(f"Error in optimize_parameters: {str(e)}")
            raise