import pickle
import bz2
import numpy as np
import sys
def save(filename, myobj):
    """
    save object to file using pickle

    @param filename: name of destination file
    @type filename: str
    @param myobj: object to save (has to be pickleable)
    @type myobj: obj
    """

    if isinstance(myobj,np.ndarray):
        if not filename.endswith('.npy.bz2'):
            filename = filename.split('.')[0]+'.npy.bz2'
    else:
        if not filename.endswith('.pkl.bz2'):
            filename = filename.split('.')[0]+'.pkl.bz2'

    try:
        f = bz2.BZ2File(filename, 'wb')
    except IOError as details:
        sys.stderr.write('File ' + filename + ' cannot be written\n')
        sys.stderr.write(details)
        return
    if isinstance(myobj,np.ndarray):
        np.save(file=f, arr=myobj)
    else:
        pickle.dump(myobj, f, protocol=2)
    f.close()


def load(filename):
    """
    Load from filename using pickle

    @param filename: name of file to load from
    @type filename: str
    """
    try:
        f = bz2.BZ2File(filename, 'rb')
    except IOError as details:
        sys.stderr.write('File ' + filename + ' cannot be read\n')
        sys.stderr.write(details)
        return
    if filename.endswith('.npy.bz2'):
        myobj = np.load(f)
    elif filename.endswith('.pkl.bz2'):
        myobj = pickle.load(f)
    else:
        raise ValueError('Does not recognize extension')
    f.close()
    return myobj
