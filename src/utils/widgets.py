import torch
import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display
from src.utils.configs import get_config_structure


__all__ = [
    'make_experiment_widgets', 'make_device_widget', 'make_split_widget',
    'make_checkpoint_file_search_widget']


def make_experiment_widgets():
    """
    Generate two co-dependent ipywidgets for selecting the task and 
    experiment from a predefined set of experiment configs.
    """
    # Parse list of experiment configs
    experiment_configs = {
        k: v[1] for k, v in get_config_structure()[0]['experiment'][0].items()}
    default_task = list(experiment_configs.keys())[0]
    default_expe = experiment_configs[default_task][0]
    
    w_task = widgets.ToggleButtons(
        options=experiment_configs.keys(),
        value=default_task,
        description="👉 Choose a segmentation task:",
        disabled=False,
        button_style='')

    w_expe = widgets.ToggleButtons(
        options=experiment_configs[default_task],
        value=default_expe,
        description="👉 Choose an experiment:",
        disabled=False,
        button_style='')

    # Define a function that updates the content of one widget based on 
    # what we selected for the other
    def update(*args):
        print(f"selected : {w_task.value}")
        w_expe.options = experiment_configs[w_task.value]
        
    w_task.observe(update)

    display(w_task)
    display(w_expe)
    
    return w_task, w_expe


def make_device_widget():
    """
    Generate an ipywidget for selecting the device on which to work
    """
    devices = [torch.device('cpu')] + [
        torch.device('cuda', i) for i in range(torch.cuda.device_count())]
    
    w = widgets.ToggleButtons(
        options=devices,
        value=devices[0],
        description="👉 Choose a device:",
        disabled=False,
        button_style='')
    
    display(w)
    
    return w


def make_split_widget():
    """
    Generate an ipywidget for selecting the data split on which to work
    """    
    w = widgets.ToggleButtons(
        options=['train', 'val', 'test'],
        value='val',
        description="👉 Choose a data split:",
        disabled=False,
        button_style='')
    
    display(w)
    
    return w


def make_checkpoint_file_search_widget():
    """
    Generate an ipywidget for locally browsing a checkpoint file
    """
    # Create and display a FileChooser widget
    w = FileChooser('', layout = widgets.Layout(width='80%'))
    display(w)
    
    # Change defaults and reset the dialog
    w.default_path = '..'
    w.default_filename = ''
    w.reset()
    
    # Shorthand reset
    w.reset(path='..', filename='')
    
    # Restrict navigation to /Users
    w.sandbox_path = '/'
    
    # Change hidden files
    w.show_hidden = False
    
    # Customize dir icon
    w.dir_icon = '/'
    w.dir_icon_append = True
    
    # Switch to folder-only mode
    w.show_only_dirs = False
    
    # Set a file filter pattern (uses https://docs.python.org/3/library/fnmatch.html)
    # w.filter_pattern = '*.txt'
    w.filter_pattern = '*.ckpt'
    
    # Set multiple file filter patterns (uses https://docs.python.org/3/library/fnmatch.html)
    # w.filter_pattern = ['*.jpg', '*.png']
    
    # Change the title (use '' to hide)
    w.title = "👉 Choose a checkpoint file *.ckpt relevant to your experiment (eg use our or your own pretrained models for this):"
    
    # Sample callback function
    def change_title(chooser):
        chooser.title = 'Selected checkpoint:'
    
    # Register callback function
    w.register_callback(change_title)

    return w
