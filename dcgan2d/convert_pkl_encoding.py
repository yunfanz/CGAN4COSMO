import numpy as np, pickle as pkl
import os, fnmatch
from scipy import interpolate
def save_obj(name, obj):
    with open(name, 'wb') as f:
        pkl.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pkl.load(f)

def find_files(directory, pattern='*.pkl', sortby="auto"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    files = np.sort(files)
    return files

def convert(dataset):
    print "converting", dataset

    output = str(dataset.split('.')[0])
        
    dic = load_obj(dataset)
    meta = dic['metadata']
    data = dic['data']
    grid = dic['grid'].astype(np.float32)

    pspec = data[:,meta=='ps'].reshape(data.shape[0], 44,13)
    #pspec /= np.std(pspec)

    redshift_in = np.arange(5.5, 27.01, 0.5)
    redshift_out = np.linspace(5.5, 27.0, 32)
    k_in = np.array([0.019712,0.033116,0.046741,0.067654,0.101422,0.152922,
        0.229226,0.343984,0.516296,0.774576,1.161992,1.677987,2.192064])
    k_out = np.exp(np.linspace(np.log(k_in[0]), np.log(k_in[-1]),16))

    pspec_interp = np.zeros((pspec.shape[0], 32, 16), dtype=np.float32)

    for i, ps in enumerate(pspec):
        f = interpolate.interp2d(k_in, redshift_in, ps, kind='cubic')
        pspec_interp[i] = f(k_out, redshift_out)

    #gridmean = np.mean(grid, axis=0, keepdims=True)
    #gridstd = np.std(grid, axis=0, keepdims=True)

    #grid = (grid - gridmean)/gridstd

    #import IPython; IPython.embed()
    np.savez(output+'.npz', data=pspec_interp, grid=grid, k=k_out, Z=redshift_out)
    
    return

if __name__ == "__main__":
    datadir = "/data1/21cmFast/pspec_emulator/"
    files = find_files(datadir, pattern='*.pkl')
    print files
    for file in files:
        convert(file)

