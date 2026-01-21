import torch
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display
from src.utils.configs import get_config_structure


__all__ = ['ConfigUI']


class ConfigUI:
    def __init__(self, tasks=None):
        """A little dashboard for selecting configuration options from
        a graphical interface.

        Disclaimer: this is not an extensive interface covering all
        possible configurations in the project but rather a minimal one
        for the main options needed in a notebook. It would be possible
        to extend this further, with more options and internal logic.


        :param tasks: str or List[str]
            Tasks to use from the `configs/experiment/` directory. If
            not provided, all tasks found in `configs/experiment/`
            will be available.
        """
        if tasks is None:
            self._tasks = []
        elif isinstance(tasks, str):
            self._tasks = [tasks]
        else:
            self._tasks =  list(tasks)
        self._make_experiment_widgets()
        self._make_device_widget()
        self._make_checkpoint_widgets()
        self._make_split_widget()
        self._make_mini_widget()
        self._update_ckpt_visibility()
        self._display()

    def _make_experiment_widgets(self):
        """Generate two co-dependent widgets for selecting the task and
        experiment from a predefined set of experiment configs.
        """
        experiment_configs = {
            k: sorted(v[1])
            for k, v in get_config_structure()[0]['experiment'][0].items()
        }

        tasks = list(experiment_configs.keys()) if len(self._tasks) == 0 \
            else self._tasks

        default_task = 'semantic' if 'semantic' in tasks else tasks[0]
        default_expe = experiment_configs[default_task][0]

        self.task = widgets.ToggleButtons(
            options=tasks,
            value=default_task,
            description="👉 Task",
        )

        self.experiment = widgets.ToggleButtons(
            options=experiment_configs[default_task],
            value=default_expe,
            description="👉 Experiment",
        )

        def update_experiments(change):
            self.experiment.options = experiment_configs[self.task.value]
            self.experiment.value = self.experiment.options[0]
            self._update_ckpt_visibility()

        self.task.observe(update_experiments, names='value')

        self.experiment.observe(self._on_experiment_change, names='value')

    def _on_experiment_change(self, change):
        self._update_ckpt_visibility()

    def _update_ckpt_visibility(self):
        if 'ezsp' in self.experiment.value:
            self.ckpt_ezsp_partition_box.layout.display = 'block'
        else:
            self.ckpt_ezsp_partition_box.layout.display = 'none'
            self.ckpt_ezsp_partition.reset(filename='')

    def _make_device_widget(self):
        """Generate a widget for selecting the device on which to work
        """
        devices = [torch.device('cpu')] + [
            torch.device('cuda', i) for i in range(torch.cuda.device_count())]

        self.device = widgets.ToggleButtons(
            options=devices,
            value=devices[1] if len(devices) > 1 else devices[0],
            description="👉 Device",
            disabled=False,
            button_style='')

    def _make_split_widget(self):
        """Generate a widget for selecting the data split on which to
        work
        """
        self.split = widgets.ToggleButtons(
            options=['train', 'val', 'test'],
            value='val',
            description="👉 Data split",
            disabled=False,
            button_style='')

    def _make_mini_widget(self):
        """Generate a widget for selecting the data size on which to
        work
        """
        self.mini = widgets.ToggleButtons(
            options=[
                ('Full', False),
                ('Mini', True),
            ],
            value=False,
            description="👉 Dataset size",
        )

    def _make_checkpoint_widgets(self):
        self.ckpt = self._make_ckpt_chooser(
            title=f"👉 Checkpoint for the model (i.e. *.ckpt file)"
        )

        self.ckpt_ezsp_partition = self._make_ckpt_chooser(
            title=f"👉 REQUIRED checkpoint for the EZ-SP partition model (i.e. *.ckpt file)"
        )

        # Wrap second ckpt in a container so we can hide/show it
        self.ckpt_ezsp_partition_box = widgets.VBox([self.ckpt_ezsp_partition])
        self.ckpt_ezsp_partition_box.layout.display = 'none'

    def _make_ckpt_chooser(self, title):
        w = FileChooser('', layout=widgets.Layout(width='75%'))
        w.reset(path='..', filename='')
        w.sandbox_path = '/'
        w.show_hidden = False
        w.filter_pattern = '*.ckpt'
        w.title = title
        return w

    def _display(self):
        expe_box = widgets.VBox(
            [self.task, self.experiment],
            layout=widgets.Layout(padding='10px')
        )

        data_box = widgets.VBox(
            [self.split, self.mini],
            layout=widgets.Layout(padding='10px')
        )

        ckpt_box = widgets.VBox(
            [self.ckpt, self.ckpt_ezsp_partition_box],
            layout=widgets.Layout(padding='10px')
        )

        run_box = widgets.VBox(
            [self.device],
            layout=widgets.Layout(padding='10px')
        )

        tabs = widgets.Tab(
            children=[expe_box, data_box, ckpt_box, run_box]
        )

        tabs.set_title(0, '🧪 Experiment')
        tabs.set_title(1, '📦 Data')
        tabs.set_title(2, '💾 Checkpoints')
        tabs.set_title(3, '⚙️ Compute')

        tabs.selected_index = 0  # show Experiment first

        display(tabs)
    #
    # def _display(self):
    #     expe_box = widgets.VBox([
    #         self.task,
    #         self.experiment,
    #     ])
    #
    #     data_box = widgets.VBox([
    #         self.split,
    #         self.mini,
    #     ])
    #
    #     ckpt_box = widgets.VBox([
    #         self.ckpt,
    #         self.ckpt_ezsp_partition_box,
    #     ])
    #
    #     run_box = widgets.VBox([
    #         self.device,
    #     ])
    #
    #     accordion = widgets.Accordion(
    #         children=[expe_box, data_box, ckpt_box,  run_box],
    #     )
    #
    #     accordion.set_title(0, '🧪 Experiment')
    #     accordion.set_title(1, '📦 Data')
    #     accordion.set_title(2, '💾 Checkpoints')
    #     accordion.set_title(3, '⚙️ Compute')
    #
    #     accordion.selected_index = 0  # open first section
    #
    #     display(accordion)

    @property
    def values(self):
        """Clean access to all widget values"""
        return {
            'task': self.task.value,
            'experiment': self.experiment.value,
            'ckpt': self.ckpt.selected,
            'ckpt_ezsp_partition': self.ckpt_ezsp_partition.selected,
            'split': self.split.value,
            'mini': self.mini.value,
            'device': self.device.value,
        }
