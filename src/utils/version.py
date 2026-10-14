import git
import logging

log = logging.getLogger(__name__)

__all__ = [
    'get_commit_hash', 'VersionHolder', 'MIN_CHECKPOINT_VERSION',
    'parse_version', 'check_checkpoint_version']



def get_commit_hash():
    """Get the current git commit hash of the current directory."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except Exception:
        log.warning("⚠️ No git commit hash found in the current directory.")
        return 'unknown'
    

class VersionHolder:
    """A simple container to hold a version string that can be shared
    across multiple modules.

    This allows for the version to be set once and accessed by all
    modules that hold a reference to the same VersionHolder instance,
    enabling efficient version management across the network.
    
    
    Where is it used ?
    - In the `TransformerBlock` to determine the definition of the
      residual connection added after the FFN. This is implemented to
      ensure compatibility with the official weights of SPT and SPC released
      before this commit fixing the residual connection.
      (https://github.com/drprojects/superpoint_transformer/commit/a0f753b35b86e06d426113bdeac9b0123b220aa3)
      
      If the version is greater than or equal to 3.0.0, the residual 
      connection is the input value of the FFN.
      Otherwise, it is the input value of the self-attention block.
    """
    def __init__(self, 
                 version = None,
                 commit_hash: str = get_commit_hash()):
        """
        :param version: str
            Version string in MAJOR.MINOR.PATCH format.
        :param commit_hash: str
        """
        self.value = version
        self.commit_hash = commit_hash
       
    @property
    def parsed(self):
        """
        Parse the version string into a dictionary containing the major,
        minor, and patch version numbers.
        
        :return: dict
            Dictionary containing the major, minor, and patch version numbers.
        """
        assert self.value is not None, "Version string is not set"
        parts = self.value.split('.')
        return {
            'major': int(parts[0]),
            'minor': int(parts[1]), 
            'patch': int(parts[2])}
        
    @property
    def major(self):
        return self.parsed['major']
    
    @property
    def minor(self):
        return self.parsed['minor']
    
    @property
    def patch(self):
        return self.parsed['patch']


# Oldest code version whose checkpoints are compatible with the current
# code. Fixes introduced in 3.1.0 change the features and graphs that
# the models see, so weights trained with earlier versions do not match
MIN_CHECKPOINT_VERSION = '3.1.0'


def parse_version(version: str):
    """Parse a 'MAJOR.MINOR.PATCH' version string into a tuple of ints,
    for comparison.
    """
    return tuple(int(x) for x in version.split('.')[:3])


def check_checkpoint_version(
        version: str,
        allow_old_checkpoint: bool = False,
        source: str = 'checkpoint'):
    """Raise an error if `version`, the code version a checkpoint was
    saved with, is older than `MIN_CHECKPOINT_VERSION`.

    :param version: str
        Version string in MAJOR.MINOR.PATCH format
    :param allow_old_checkpoint: bool
        For development only. If True, only a warning is logged and the
        checkpoint can be loaded anyway
    :param source: str
        Name of the checkpoint, for the error message
    """
    if parse_version(version) >= parse_version(MIN_CHECKPOINT_VERSION):
        return

    msg = (
        f"The {source} was saved with version {version} of the code, but "
        f"version {MIN_CHECKPOINT_VERSION} or later is required. The fixes "
        f"introduced in {MIN_CHECKPOINT_VERSION} change the features and "
        f"graphs the models are trained on (see CHANGELOG.md), so weights "
        f"trained with an earlier version are not compatible with this "
        f"code. Please either retrain your model with this version, or use "
        f"the code release matching your checkpoint (e.g. "
        f"`git checkout v3.0.0`).")

    if not allow_old_checkpoint:
        raise ValueError(
            f"{msg}\nFor development purposes only, you may load it anyway "
            f"with `allow_old_checkpoint=True`.")

    log.warning(
        f"⚠️ {msg}\n`allow_old_checkpoint=True`: loading it anyway. "
        f"Results may differ from those obtained with version {version}.")
