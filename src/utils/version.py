import git
import logging

log = logging.getLogger(__name__)

__all__ = ['get_commit_hash', 'VersionHolder']



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