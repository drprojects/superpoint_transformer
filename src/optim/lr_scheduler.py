from torch.optim.lr_scheduler import _LRScheduler, StepLR, MultiStepLR, \
    ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import math
import warnings


__all__ = [
    'CosinePowerAnnealingLR', 'StepLRWithWarmup', 'MultiStepLRWithWarmup',
    'ExponentialLRWithWarmup', 'CosineAnnealingLRWithWarmup',
    'CosinePowerAnnealingLRWithWarmup', 'ReduceLROnPlateauWithWarmup']


class _WarmupLR(_LRScheduler):
    """Wrapper adding a warmup phase to a Pytorch Scheduler.

    This class is not intended to be directly instantiated. One should
    instead create child classes with the desired `_SCHEDULER_CLASS`.

    Credit: https://github.com/lehduong/torch-warmup-lr

    :param init_lr: float
        Learning rate value to start the warmup from. All your
        optimizer's parameter groups will be warmed up from
        `init_lr` to their initial value as set in the optimizer
    :param num_warmup: int
        Number of scheduler steps (ie epochs, most of the time)
        dedicated to warming up
    :param warmup_strategy: str
        Warmup strategy, among ['linear', 'cos', 'constant']
    """
    _SCHEDULER_CLASS = None

    def __init__(
            self, *args, warmup_init_lr=1e-6, num_warmup=1,
            warmup_strategy='cos', **kwargs):

        assert warmup_strategy in ['linear', 'cos', 'constant'], \
            f"Expect warmup_strategy to be one of ['linear', 'cos', " \
            f"'constant'] but got {warmup_strategy}"

        self._scheduler = self._SCHEDULER_CLASS(*args, **kwargs)
        self._init_lr = warmup_init_lr
        self._num_warmup = num_warmup
        self._step_count = 0

        # Define the strategy to warm up learning rate
        self._warmup_strategy = warmup_strategy
        if warmup_strategy == 'cos':
            self._warmup_func = self._warmup_cos
        elif warmup_strategy == 'linear':
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const

        # Dave initial learning rate of each param group. only useful
        # when each param groups having different learning rate
        self._format_param()

        # A first step is needed to initialize the LR
        self.step()

    def __getattr__(self, name):
        if name == '_scheduler':
            if name in self.__dict__.keys():
                return self._scheduler
            else:
                return None
        return getattr(self._scheduler, name)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if (key != 'optimizer' and key != '_scheduler')}

        wrapped_state_dict = {
            key: value
            for key, value in self._scheduler.__dict__.items()
            if key != 'optimizer'}

        return {'wrapped': wrapped_state_dict, 'wrapper': wrapper_state_dict}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        :param state_dict: dict
            Scheduler state. Should be an object returned from a call
            to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict['wrapper'])
        self._scheduler.__dict__.update(state_dict['wrapped'])

    def _format_param(self):
        """Set the first and last learning rates for the warmup phase,
        for each parameter group. All parameter groups will start the
        warmup at the same value `self._init_lr`.
        """
        for group in self._scheduler.optimizer.param_groups:
            group['warmup_max_lr'] = group['lr']
            group['warmup_initial_lr'] = min(self._init_lr, group['lr'])

    def _warmup_cos(self, start, end, pct):
        """Cosine warmup scheme.
        """
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _warmup_const(self, start, end, pct):
        """Constant warmup scheme.
        """
        return start if pct < 0.9999 else end

    def _warmup_linear(self, start, end, pct):
        """Linear warmup scheme.
        """
        return (end - start) * pct + start

    def get_lr(self):
        lrs = []
        step_num = self._step_count

        # warm up learning rate
        if step_num <= self._num_warmup:
            for group in self._scheduler.optimizer.param_groups:
                computed_lr = self._warmup_func(
                    group['warmup_initial_lr'], group['warmup_max_lr'],
                    step_num / self._num_warmup)
                lrs.append(computed_lr)
        else:
            lrs = self._scheduler.get_lr()
        return lrs

    def step(self, *args, **kwargs):
        if self._step_count <= self._num_warmup:
            values = self.get_lr()
            for param_group, lr in zip(
                    self._scheduler.optimizer.param_groups, values):
                param_group['lr'] = lr
            self._step_count += 1
        else:
            self._scheduler.step(*args, **kwargs)


class CosinePowerAnnealingLR(CosineAnnealingLR):
    """Same as CosineAnnealingLR, but with an additional `power`
    parameter, to mitigate the annealing time spent on large learning
    rates (ie `power < 1`) or small learning rates (ie `power > 1`).
    """

    def __init__(
            self, optimizer, T_max, eta_min=0, power=2, last_epoch=-1,
            verbose=False):
        super().__init__(
            optimizer, T_max, eta_min=eta_min, last_epoch=last_epoch,
            verbose=verbose)
        self.power = power

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min + (base_lr - self.eta_min) *
                ((1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2) ** self.power
                for base_lr, group in
                zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) *
                ((1 - math.cos(math.pi / self.T_max)) / 2) ** self.power
                for base_lr, group in
                zip(self.base_lrs, self.optimizer.param_groups)]
        return [
            ((1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
             (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))) ** self.power *
            (group['lr'] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            ((1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2) ** self.power
            for base_lr in self.base_lrs]


class StepLRWithWarmup(_WarmupLR):
    """StepLRWithWarmup with warmup.
    """
    _SCHEDULER_CLASS = StepLR


class MultiStepLRWithWarmup(_WarmupLR):
    """MultiStepLR with warmup.
    """
    _SCHEDULER_CLASS = MultiStepLR


class ExponentialLRWithWarmup(_WarmupLR):
    """ExponentialLR with warmup.
    """
    _SCHEDULER_CLASS = ExponentialLR


class CosineAnnealingLRWithWarmup(_WarmupLR):
    """CosineAnnealingLR with warmup.
    """
    _SCHEDULER_CLASS = CosineAnnealingLR


class CosinePowerAnnealingLRWithWarmup(_WarmupLR):
    """CosinePowerAnnealingLR with warmup.
    """
    _SCHEDULER_CLASS = CosinePowerAnnealingLR


class ReduceLROnPlateauWithWarmup(_WarmupLR):
    """ReduceLROnPlateau with warmup.
    """
    _SCHEDULER_CLASS = ReduceLROnPlateau


ON_PLATEAU_SCHEDULERS = (ReduceLROnPlateau, ReduceLROnPlateauWithWarmup)
