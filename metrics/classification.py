import numpy as np
from numpy import ndarray

from registry import METRICS
from .metrics import Metric


@METRICS.register_class
class AccuracyMeter(Metric):

    def __init__(self, name=None, binary_mod=False, target_fields=None):
        super().__init__('accuracy' if name is None else name, target_fields=target_fields)
        self.binary_mod = binary_mod

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        if self.binary_mod:
            tdim = target.ndim
            pdim = prediction.ndim
            if tdim == pdim:
                prediction = prediction > 0
            else:
                raise ValueError(f'Dimension sizes for target and prediction do not match {tdim} != {pdim}')
        else:
            if prediction.ndim == target.ndim + 1:
                prediction = prediction.argmax(1)
        return (target == prediction).mean()


@METRICS.register_class
class FbetaMeter(Metric):

    def __init__(self, beta, num_classes=None, target_class=None, binary_mod=False, name=None,
                 weighted=False, ignore_index=-100, reduce=True, target_fields=None):
        if num_classes is None and target_class is None and not binary_mod:
            raise TypeError('You must specify either `num_classes` or `target_class` or `binary_mod`')
        if target_class is not None and binary_mod:
            raise ValueError('`target_class` is not compatible with `binary_mod`')
        if (target_class is not None or binary_mod) and weighted:
            raise ValueError('`weighted` is not compatible with `binary_mod` and `target_class`')
        if name is None:
            if target_class is None:
                name = f'F_beta={beta}'
            else:
                name = f'F_beta={beta}_class={target_class}'

        super().__init__(name, target_fields=target_fields)

        self._num_classes = num_classes
        self._target_class = target_class
        self._binary_mod = binary_mod
        self._weighted = weighted
        self._ignore_index = ignore_index
        self._reduce = reduce
        self._beta_sq = beta ** 2
        if self._binary_mod or self._target_class is not None:
            self.true_pos = 0
            self.false_pos = 0
            self.false_neg = 0
        else:
            self._classes_idx = np.arange(self._num_classes)[:, None]
            self.true_pos = np.zeros(self._num_classes)
            self.false_pos = np.zeros(self._num_classes)
            self.false_neg = np.zeros(self._num_classes)

    def reset(self):
        if self._binary_mod or self._target_class is not None:
            self.true_pos = 0
            self.false_pos = 0
            self.false_neg = 0
        else:
            self.true_pos = np.zeros(self._num_classes)
            self.false_pos = np.zeros(self._num_classes)
            self.false_neg = np.zeros(self._num_classes)

    def _unify_shapes(self, target, prediction):
        if self._binary_mod:
            if prediction.shape != target.shape:
                raise ValueError('shapes of target and prediction do not match',
                                 target.shape, prediction.shape)
            prediction = prediction > 0
        else:
            # Dimensions check
            if prediction.shape[0] != target.shape[0]:
                raise ValueError('Batch size of target and prediction do not match',
                                 target.shape[0], prediction.shape[0])
            if prediction.ndim == target.ndim + 1:
                prediction = prediction.argmax(1)

            # Dimensions check
            if prediction.shape[1:] != target.shape[1:]:
                raise ValueError('Spatial shapes of target and prediction do not match',
                                 target.shape[1:], prediction.shape[1:])

            if self._target_class is not None:
                target = target == self._target_class
                prediction = prediction == self._target_class

        target = target.reshape(-1)
        prediction = prediction.reshape(-1)
        prediction = prediction[target != self._ignore_index]
        target = target[target != self._ignore_index]
        return prediction, target

    def calculate(self, target: ndarray, prediction: ndarray) -> ndarray:
        target, prediction = self._unify_shapes(target, prediction)

        if self._binary_mod or self._target_class is not None:
            pred_n = prediction
            true_n = target
        else:
            true_n: np.ndarray = target == self._classes_idx
            pred_n: np.ndarray = prediction == self._classes_idx
        tp = (pred_n & true_n).sum(-1)
        fp = (pred_n & ~true_n).sum(-1)
        fn = (~pred_n & true_n).sum(-1)
        tp_rate = (1 + self._beta_sq) * tp
        denum = tp_rate + self._beta_sq * fn + fp
        np.seterr(divide='ignore')
        f1_scores = np.where(denum != 0.0, tp_rate / denum, 0)

        if self._reduce:
            if self._weighted:
                weights = (tp + fn) / target.shape[0]
                f1_scores = weights @ f1_scores
            else:
                f1_scores = np.mean(f1_scores)
        return f1_scores

    def update(self, target, prediction, *args, **kwargs):
        target, prediction = self._unify_shapes(target, prediction)

        if self._binary_mod or self._target_class is not None:
            pred_n = prediction
            true_n = target
        else:
            true_n: np.ndarray = target == self._classes_idx
            pred_n: np.ndarray = prediction == self._classes_idx
        self.true_pos += (pred_n & true_n).sum(-1)
        self.false_pos += (pred_n & ~true_n).sum(-1)
        self.false_neg += (~pred_n & true_n).sum(-1)

    def on_epoch_end(self, do_reset=True):
        tp = self.true_pos
        fp = self.false_pos
        fn = self.false_neg

        tp_rate = (1 + self._beta_sq) * tp
        denum = tp_rate + self._beta_sq * fn + fp
        np.seterr(divide='ignore')
        f1_scores = np.where(denum != 0.0, tp_rate / denum, 0)

        if self._reduce:
            if self._weighted:
                weights = (tp + fn) / (tp + fn).sum()
                f1_scores = weights @ f1_scores
            else:
                f1_scores = np.mean(f1_scores)
        if do_reset:
            self.reset()
        return f1_scores


@METRICS.register_class
class F1Meter(FbetaMeter):

    def __init__(self, num_classes=None, target_class=None, binary_mod=False, name=None, weighted=False,
                 ignore_index=-100, reduce=True, target_fields=None):
        if name is None:
            if target_class is None:
                name = f'F1'
            else:
                name = f'F1_class={target_class}'

        super().__init__(1, num_classes, target_class, binary_mod, name, weighted,
                         ignore_index, reduce, target_fields)


@METRICS.register_class
class MultiLabelFbetaMeter(Metric):
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
