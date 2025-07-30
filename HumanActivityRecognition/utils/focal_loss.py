import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


#combina due perdite con peso epsilon dove x  è perdita da etichette smussate mentre y è la perdita della classica cross entropy
def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

#PER definire come combinare le perdite
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1] #numero di classi
        log_preds = F.log_softmax(preds, dim=-1)  # trasformazione in log-probabilità Serve per avere le probabilità logaritmiche necessarie per la CE
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction) #somma le log-prob di tutte le classi per ogni esempio, col - trasformo in perdita, e con reduce_loss applico la riduzione scelta
        nll = F.nll_loss(log_preds, target, reduction=self.reduction) 
        return linear_combination(loss / n, nll, self.epsilon)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        """
        Weighted Cross Entropy Loss con pesi personalizzati
        
        Args:
            class_weights: tensor con pesi per ogni classe
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights)


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.7, ce_weight=0.3, gamma=2.0, class_weights=None, num_classes=None):
        """
        Combinazione di Focal Loss e Weighted Cross Entropy
        
        Args:
            focal_weight: peso per la focal loss
            ce_weight: peso per la cross entropy
            gamma: parametro gamma per focal loss
            class_weights: pesi delle classi
            num_classes: numero di classi
        """
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        
        #CREO LE DUE LOSS FUNCTION PER LA COMBINAZIONE
        self.focal_loss = FocalLoss(
            gamma=gamma, 
            alpha=class_weights, 
            task_type='multi-class', 
            num_classes=num_classes
        )
        self.ce_loss = WeightedCrossEntropyLoss(class_weights)
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets) # calcolo la focal loss
        ce = self.ce_loss(inputs, targets) # calcolo la cross entropy weighted
        return self.focal_weight * focal + self.ce_weight * ce #combino le due perdite con i pesi specificati
    
