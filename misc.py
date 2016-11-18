import matplotlib.pyplot as plt
import datetime, calendar

def xticks2dayofyear():
    '''adjust xticks when it represents day of year.'''
    xticks = [datetime.datetime(2000, month, 1).timetuple().tm_yday
        for month in range(1, 13)]
    months = [datetime.datetime(2000, month, 1).strftime('%b')
        for month in range(1, 13)]
    plt.xticks(xticks, months, ha='left')
    plt.xlim(1, 366)

def _num2lat(n):
    if n > 0:
        lat = '{}$^{{\circ}}$N'.format(n)
    elif n <0:
        lat = '{}$^{{\circ}}$S'.format(-n)
    else:
        lat = '{}$^{{\circ}}$'.format(n)
    return lat
def _num2lon(n):
    n = n % 360
    if n < 180:
        lon = '{}$^{{\circ}}$E'.format(n)
    elif n > 180:
        lon = '{}$^{{\circ}}$W'.format(360-n)
    else:
        lon = '{}$^{{\circ}}$'.format(n)
    return lon
def _num2mon(n):
    return calendar.month_name[n][0:3]
def _set_new_tick_labels(axis, ticks, ticklabels, **kw):
    if axis == 'x':
        plt.xticks(ticks, ticklabels, **kw)
    elif axis == 'y':
        plt.yticks(ticks, ticklabels, **kw)
    else:
        raise ValueError('The axis parameter is either "x" or "y"!')
def change_ticklabels(axis, kind, ticks=None):
    '''
    axis: 'x' or 'y'.
    kind: 'lat', 'lon', 'month', or 'dayofyear'.'''

    # get ticks
    if ticks is None:
        if axis == 'x':
            ticks = plt.gca().get_xticks()
            xlim = plt.xlim()
            ticks = [tick for tick in ticks if xlim[0]<=tick<=xlim[1]]
        elif axis == 'y':
            ticks = plt.gca().get_yticks()
            ylim = plt.ylim()
            ticks = [tick for tick in ticks if ylim[0]<=tick<=ylim[1]]
        else:
            raise ValueError('axis is either "x" or "y".')

    # change tick labels
    if kind == 'lat':
        ticks = [tick for tick in ticks if -90<=tick<=90 ]
        ticklabels = [_num2lat(tick) for tick in ticks]
        _set_new_tick_labels(axis, ticks, ticklabels)
    elif kind == 'lon':
        ticklabels = [_num2lon(tick) for tick in ticks]
        _set_new_tick_labels(axis, ticks, ticklabels)
    elif kind == 'month':
        ticks = np.arange(1, 13)
        ticklabels = [_num2mon(tick) for tick in ticks]
        _set_new_tick_labels(axis, ticks, ticklabels)
        if axis == 'x':
            plt.xlim(0.5, 12.5)
        else:
            plt.ylim(0.5, 12.5)
            plt.gca().invert_yaxis()
    elif kind == 'dayofyear':
        ticks = [datetime.datetime(2000, month, 1).timetuple().tm_yday
            for month in range(1, 13)]
        ticklabels = [calendar.month_name[i][0:3] for i in range(1,13)]
        if axis == 'x':
            _set_new_tick_labels(axis, ticks, ticklabels, ha='left')
            plt.xlim(1, 366)
        else:
            _set_new_tick_labels(axis, ticks, ticklabels, va='top')
            plt.ylim(1, 366)
            plt.gca().invert_yaxis()
