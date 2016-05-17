# requries convert, montage tools in the OS
import os
def _get_ifile(ifile_raw, ivec=None):
    '''get the actual ifile and ofile that can be used in the image manipulating functions'''
    if isinstance(ifile_raw, str):
        if '*.' in ifile_raw:
            assert ivec is not None, 'Please provide ivec.'
            ifile = ' '.join([ifile_raw.replace('*.', str(i)+'.')
                for i in ivec])
        else:
            ifile = ifile_raw
    else:
        # ifile_input is a list
        ifile = ' '.join(ifile_raw)
    return ifile
def _run_shell(cmd):
    '''run system shell command'''
    print(cmd)
    status = os.system(cmd)
    if status == 0:
        print('\t** Succeeded:) **')
    else:
        print('\t** Failed! **')
def make_gif(ifile_raw, ivec=None, ofile=None, delay=100, loop=0):
    '''make gif images from ifiles'''
    # deal with input parameters
    ifile = _get_ifile(ifile_raw=ifile_raw, ivec=ivec)
    if ofile is None:
        assert isinstance(ifile_raw, str) and '*.' in ifile_raw
        i = ifile_raw.index('*.')
        ofile = ifile_raw[:i] + 'i.gif'

    cmd = ' '.join(['convert', '-delay', str(delay), '-loop', str(loop),
        ifile, ofile])
    _run_shell(cmd)
def merge_imgs(ifile_raw, ivec=None, ofile=None, n_col=1):
    '''Merge multiple images into a single one.'''
    ifile = _get_ifile(ifile_raw, ivec=ivec)
    if ofile is None:
        assert isinstance(ifile_raw, str) and '*.' in ifile_raw
        ofile = ifile_raw.replace('*.', 'i.')
    # system montage command
    cmd = ' '.join(['montage', '-mode concatenate', '-tile',
        '{}x'.format(n_col), ifile, ofile])
    _run_shell(cmd)
