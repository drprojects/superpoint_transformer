import os
import os.path as osp
from six.moves import urllib
import ssl
import subprocess


def download_url(url, folder, log=True):
    """Download the content of an URL to a specific folder.

    :param url: string
    :param folder: string
    :param log: bool
        If `False`, will not print anything to the console.
    :return:
    """
    filename = url.rpartition("/")[2]
    path = osp.join(folder, filename)
    if osp.exists(path):  # pragma: no cover
        if log:
            print("Using exist file", filename)
        return path
    if log:
        print("Downloading", url)
    try:
        os.makedirs(folder)
    except:
        pass
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with open(path, "wb") as f:
        f.write(data.read())
    return path


def run_command(cmd):
    """Run a command-line process from Python and print its outputs in
    an online fashion.

    Credit: https://www.endpointdev.com/blog/2015/01/getting-realtime-output-using-python/
    """
    # Create the process
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    # p = subprocess.run(cmd, shell=True)

    # Poll process.stdout to show stdout live
    while True:
        output = p.stdout.readline()
        if p.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = p.poll()
    print('Done')
    print('')

    return rc
