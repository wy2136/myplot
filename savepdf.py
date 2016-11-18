import os, tempfile
import matplotlib.pyplot as plt

print('[Attention]: system command "epstopdf" needed for myplot.savepdf.savefig.')

def savefig(fname, **kw):
    '''Modified pyplot.savefig:
    Save to eps first, then convert to pdf using 'epstopdf' system command. '''
    if fname.endswith('.pdf'):
        fname = fname[0:-4]
    with tempfile.NamedTemporaryFile(suffix='.eps') as tmp:
        plt.savefig(tmp.name, format='eps', **kw)
        print('[OK]: {} has been created for temporary use.'.format(tmp.name))
        cmd = ' '.join(['epstopdf', tmp.name, '--outfile='+fname+'.pdf'])
        status = os.system(cmd)
        if status == 0:
            print('[OK]: ' + cmd)
        else:
            print('\t[Error]: ' + cmd)
