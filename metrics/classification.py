import numpy as np
from numpy import ndarray

from registry import METRICS
from .metrics import Metric


@METRICS.register_class
class AccuracyMeter(Metric):

    def __init__(self, name=None, get_logits=False, target_fields=None):
        super().__init__('accuracy' if name is None else name, target_fields=target_fields)
        self.get_logits = get_logits

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        if self.get_logits and prediction.ndim == target.ndim:
            prediction = prediction > 0

        if prediction.ndim == target.ndim + 1:
            prediction = prediction.argmax(1)
        return (target == prediction).mean()


@METRICS.register_class
class FbetaMeter(Metric):
    MODES = ['macro', 'binary']

    def __init__(self, num_classes, name=None, get_logits=False, average='macro', beta=1, target_fields=None):
        if name is None:
            name = f'F_beta={beta}_{average}'

        super().__init__(name, target_fields=target_fields)

        if average not in self.MODES:
            raise ValueError(f"Invalid average setting. "
                             f"Please choose one of {self.MODES}.")

        if average == 'binary' and num_classes != 2:
            raise ValueError(f"Target is multiclass but average='binary'. "
                             f"Please choose another average setting, "
                             f"one of {[mode for mode in self.MODES if mode != 'binary']}.")

        self.beta_sq = beta ** 2
        self.num_classes = num_classes
        self.mode = average
        self.get_logits = get_logits
        self.true_pos = np.zeros(self.num_classes)
        self.false_pos = np.zeros(self.num_classes)
        self.false_neg = np.zeros(self.num_classes)

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        """ Usage on mini-batch is deprecated """
        if prediction.ndim == target.ndim + 1:
            prediction = prediction.argmax(1)

        f1_scores = np.zeros(self.num_classes)
        for n in range(self.num_classes):
            true_n = target == n
            pred_n = prediction == n
            tp = (pred_n & true_n).sum()
            fp = (pred_n & ~true_n).sum()
            fn = (true_n & ~pred_n).sum()
            tp_rate = (1 + self.beta_sq) * tp
            denum = tp_rate + self.beta_sq * fn + fp
            if denum == 0.0:
                f1_scores[n] = 0
            else:
                f1_scores[n] = tp_rate / denum

        if self.mode == 'binary':
            return f1_scores[1]

        elif self.mode == 'macro':
            return f1_scores.mean()

    def update(self, target, prediction, *args, **kwargs):
        if self.get_logits and prediction.ndim == target.ndim:
            prediction = prediction > 0

        if prediction.ndim == target.ndim + 1:
            prediction = prediction.argmax(1)
        for n in range(self.num_classes):
            true_n = target == n
            pred_n = prediction == n
            self.true_pos[n] += (pred_n & true_n).sum()
            self.false_pos[n] += (pred_n & ~true_n).sum()
            self.false_neg[n] += (true_n & ~pred_n).sum()

    def reset(self):
        self.true_pos = np.zeros(self.num_classes)
        self.false_pos = np.zeros(self.num_classes)
        self.false_neg = np.zeros(self.num_classes)

    def on_epoch_end(self):
        if self.mode == 'binary':
            tp = self.true_pos[1]
            fp = self.false_pos[1]
            fn = self.false_neg[1]
        elif self.mode == 'macro':
            tp = self.true_pos.sum()
            fp = self.false_pos.sum()
            fn = self.false_neg.sum()
        tp_rate = (1 + self.beta_sq) * tp
        denum = tp_rate + self.beta_sq * fn + fp
        if denum == 0.0:
            output = 0
        else:
            output = tp_rate / denum
        self.reset()
        return output


@METRICS.register_class
class F1Meter(FbetaMeter):

    def __init__(self, num_classes, name=None, get_logits=False, average='macro', target_fields=None):
        if name is None:
            name = f'F1_{average}'

        super().__init__(num_classes, name=name, get_logits=get_logits,
                         average=average, beta=1, target_fields=target_fields)


@METRICS.register_class
class MultiLabelFbetaMeter(Metric):
    MODES = ['macro', 'binary']

    def __init__(self, num_classes, name=None, get_logits=False,
                 target_class=None, beta=1, target_fields=None):
        if name is None:
            if target_class is None:
                name = f'MultiLabel_F_beta={beta}'
            else:
                name = f'F_beta={beta}_class={target_class}'

        super().__init__(name, target_fields=target_fields)

        self.beta_sq = beta ** 2
        self.num_classes = num_classes
        self.target_class = target_class
        self.get_logits = get_logits
        if target_class is None:
            self.true_pos = np.zeros(self.num_classes)
            self.false_pos = np.zeros(self.num_classes)
            self.false_neg = np.zeros(self.num_classes)
        else:
            self.true_pos = 0
            self.false_pos = 0
            self.false_neg = 0

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        """ Usage on mini-batch is deprecated """
        if self.get_logits:
            prediction = prediction > 0

        if self.target_class is not None:
            prediction = prediction[:, self.target_class]
            target = target[:, self.target_class]
        tp = (prediction & target).sum(0)
        fp = (prediction & ~target).sum(0)
        fn = (target & ~prediction).sum(0)

        tp_rate = (1 + self.beta_sq) * tp
        denum = tp_rate + self.beta_sq * fn + fp
        if isinstance(denum, int):
            f1 = 1 if denum == 0.0 else tp_rate / denum
        else:
            f1 = np.where(denum == 0.0, 1, tp_rate / denum)

        if self.target_class is None:
            f1 = f1.mean()
        return f1

    def update(self, target, prediction, *args, **kwargs):
        if self.get_logits:
            prediction = prediction > 0

        if self.target_class is not None:
            prediction = prediction[:, self.target_class]
            target = target[:, self.target_class]
        self.true_pos += (prediction & target).sum(0)
        self.false_pos += (prediction & ~target).sum(0)
        self.false_neg += (target & ~prediction).sum(0)

    def reset(self):
        if self.target_class is None:
            self.true_pos = np.zeros(self.num_classes)
            self.false_pos = np.zeros(self.num_classes)
            self.false_neg = np.zeros(self.num_classes)
        else:
            self.true_pos = 0
            self.false_pos = 0
            self.false_neg = 0

    def on_epoch_end(self):
        tp_rate = (1 + self.beta_sq) * self.true_pos
        denum = tp_rate + self.beta_sq * self.false_neg + self.false_pos
        if isinstance(denum, int):
            f1 = 1 if denum == 0.0 else tp_rate / denum
        else:
            f1 = np.where(denum == 0.0, 1, tp_rate / denum)

        if self.target_class is None:
            f1 = f1.mean()
        self.reset()
        return f1


@METRICS.register_class
class MultiLabelF1Meter(MultiLabelFbetaMeter):

    def __init__(self, num_classes, name=None, get_logits=False,
                 target_class=None, target_fields=None):
        if name is None:
            if target_class is None:
                name = f'MultiLabel_F1'
            else:
                name = f'F1_class={target_class}'

        super().__init__(num_classes, name=name, target_class=target_class,
                         get_logits=get_logits, beta=1, target_fields=target_fields)
