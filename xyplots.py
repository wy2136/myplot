# -*- coding: utf-8 -*-
"""
@author: Wenchang Yang (yang.wenchang@uci.edu)
"""
# from .mypyplot import vcolorbar, hcolorbar

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

#
# load 2-d test data
def load_test_data(Nx=180, Ny=90, Nmode_x=2, Nmode_y=1):
    x = np.linspace(0, 360, Nx+1)[:-1]
    x = x[np.newaxis, :] # reshape(1, Nx)
    y = np.linspace(-90, 90, Ny+1)
    y = (y[:-1] + y[1:])/2.0
    y = y[np.newaxis,:] # reshape(1, Ny)
    data = np.cos(y/180.0 * np.pi * Nmode_y).T * np.sin(x/360.0 * 2 * np.pi * Nmode_x)
    return data, np.squeeze(x), np.squeeze(y)
#
def _get_grid_edges(x,y):
    '''Get grid edges from grid centers.'''
    if len(x)==2:
        # when x has only two elements: treat them as edges
        x_edge = x
    else:
        x_ = np.hstack((
            x[0]*2 - x[1],
            x,
            x[-1]*2 - x[-2]))
        x_edge = ( x_[:-1] + x_[1:] )/2.0
    if len(y)==2:
        y_edge = y
    else:
        y_ = np.hstack((
            y[0]*2 - y[1],
            y,
            y[-1]*2 - y[-2]))
        y_edge = ( y_[:-1] + y_[1:] )/2.0
    return x_edge, y_edge
def _load_coast():
    this_dir, this_filename = os.path.split(__file__)
    lon_path = os.path.join(this_dir, 'data', 'lon.npy')
    lat_path = os.path.join(this_dir, 'data', 'lat.npy')
    coast_lon = np.load(lon_path)
    coast_lat = np.load(lat_path)
    return coast_lon, coast_lat
def colorbar(**kw):
    '''Customized colorbar.
    parameters:
        cbar_type: str, 'vertical' ('v') or 'horizontal' ('h').
        size: str, default is '2.5%' for 'vertical', '5%' for 'horizontal'.
        pad: number, default is 0.1 for 'vertical', 0.4 for 'horizontal'.
        extend: str, default is 'neither'.
        units: str or None (default). '''
    ax_current = plt.gca()
    cbar_type = kw.pop('cbar_type', 'vertical')
    cbar_extend = kw.pop('extend', 'neither')
    units = kw.pop('units', None)
    mappable = kw.pop('mappable', None)
    if cbar_type in ('v', 'vertical'):
        cbar_size = kw.pop('size', '2.5%')
        cbar_pad = kw.pop('pad', 0.1)
        cbar_position = 'right'
        cbar_orientation = 'vertical'
    elif cbar_type in ('h', 'horizontal'):
        cbar_size = kw.pop('size', '5%')
        cbar_pad = kw.pop('pad', 0.4)
        cbar_position = 'bottom'
        cbar_orientation = 'horizontal'
    divider = make_axes_locatable(ax_current)
    cax = divider.append_axes(cbar_position, size=cbar_size, pad=cbar_pad)
    cbar = plt.colorbar(mappable=mappable, cax=cax, extend=cbar_extend,
        orientation=cbar_orientation, **kw)
    # units position
    if units is not None:
        if cbar_type in ('v', 'vertical'):
            # put the units on the top of the vertical colorbar
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.set_xlabel(units)
        else:
            cbar.ax.yaxis.set_label_position('right')
            cbar.ax.set_ylabel(units, rotation=0, ha='left', va='center')
    # set back the main axes as the current axes
    plt.sca(ax_current)
def figcolorbar(mappable=None, **kw):
    '''Colorbar for multiple axes in the same figure.
    kw parameters:
        cbar_type: str, default 'vertical'.
        center: tuple, (x, y).
        width: numeric.
        height: numeric.
        unist: str.
        extend: str, default is 'neither'. '''
    cbar_type = kw.pop('cbar_type', 'vertical')
    extend = kw.pop('extend', 'neither')
    units = kw.pop('units', None)
    if cbar_type in ('vertical', 'v'):
        orientation = 'vertical'
        right = plt.rcParams['figure.subplot.right']
        center = kw.pop('center', (0.95, 0.5))
        width = kw.pop('width', 0.01)
        height = kw.pop('height', 0.5)
        # plt.subplots_adjust(right=right)
    else:
        orientation = 'horizontal'
        right = plt.rcParams['figure.subplot.right']
        center = kw.pop('center', (0.5, 0.05))
        width = kw.pop('width', 0.5)
        height = kw.pop('height', 0.015)
        # plt.subplots_adjust(bottom=bottom)
    pos = [center[0]-width/2.0, center[1]-height/2.0, width, height]
    cax = plt.gcf().add_axes(pos)
    cbar = plt.colorbar(mappable=mappable,
        cax=cax, extend=extend, orientation=orientation, **kw)
    # units
    if units is not None:
        if cbar_type in ('vertical', 'v'):
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.set_xlabel(units)
        else:
            cbar.ax.yaxis.set_label_position('right')
            cbar.ax.set_ylabel(units, rotation=0, ha='left', va='center')

def text(*args, **kw):
    transform = kw.pop('transform', plt.gca().transAxes)
    plt.text(*args, transform=transform, **kw)
def xticks2lon(new_xticks=None):
    '''Convert xticks to longitudes. '''
    if new_xticks is not None:
        plt.gca().set_xticks(new_xticks)
    current_xticks = plt.gca().get_xticks()
    current_xticklabels = plt.gca().get_xticklabels()
    new_xticklabels = current_xticklabels
    for i, x in enumerate(current_xticks):
        x = np.mod(x,360)
        if x>0 and x<180:
            new_xticklabels[i] = str(int(x)) + '$^{\circ}$E'
        elif x>180 and x<360:
            new_xticklabels[i] = str(int(360-x))+'$^{\circ}$W'
        elif x==0 or x==180:
            current_xticklabels[i] = str(int(x)) + '$^{\circ}$'
    plt.gca().set_xticklabels(new_xticklabels)
def xticks2month(show_initials=False):
    '''Convert xticks to months. '''
    months = ['Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec']
    if show_initials:
        months = [mon[0] for mon in months]
    plt.xlim(.5,12.5)
    plt.xticks(range(1,13),months)
def yticks2lat(new_yticks=None):
    '''Convert yticks to latitudes. '''
    if new_yticks is not None:
        plt.gca().set_yticks(new_yticks)
    current_yticks = plt.gca().get_yticks()
    current_yticklabels = plt.gca().get_yticklabels()
    new_yticklabels = current_yticklabels
    for i, y in enumerate(current_yticks):
        if y>0:
            new_yticklabels[i] = str(int(y)) + '$^{\circ}$N'
        elif y<0 :
            new_yticklabels[i] = str(int(-y))+'$^{\circ}$S'
        else:
            new_yticklabels[i] = str(int(y)) + '$^{\circ}$'
    plt.gca().set_yticklabels(new_yticklabels)
#
# ######## plot basemap
def mapplot(lon=None, lat=None, **kw):
    '''Plot the basemap using coast data from Matlab.

    #### Input
        lon: vector-like longitude range.
        lat: vector-like latitude range.
        kw: optional parameters

    ------
    Optional parameters:
        lonlatbox: [lon_start, lon_end, lat_start, lat_end] like array.
        lonlatbox_color: lonlat box color.
        lonlatbox_kw: dict parameters used in plotting lonlatbox.

        fill_continents: boolean, default is False.
        continents_color: color of continents, default is 0.5.
        coastlines_color: color of coastlines, default is 0.66.
        coastlines_width: line width of coastlines, default is 1.

    '''
    if lon is None:
        if plt.get_fignums():# figures already exist
            lon = plt.gca().get_xlim()
        else:
            lon = np.arange(0,360,2)
    if lat is None:
        if plt.get_fignums(): # figures already exist
            lat = plt.gca().get_ylim()
        else:
            lat = np.arange(-89, 90, 2)
    lon = np.squeeze(lon)
    lat = np.squeeze(lat)

    # get grid edges
    lon_edge, lat_edge = _get_grid_edges(lon, lat)
    lat_edge[lat_edge < -90] = -90
    lat_edge[lat_edge >  90] = 90

    # plot lonlatbox
    lonlatbox = kw.pop('lonlatbox', None)
    if lonlatbox is not None:
        lon0, lon1, lat0, lat1 = lonlatbox # unpack lonlatbox
        lon_ = np.array([
            np.linspace(lon0, lon1, 100),
            lon1*np.ones(100),
            np.linspace(lon1, lon0, 100),
            lon0*np.ones(100)
            ]).ravel()
        lat_ = np.array([
            lat0*np.ones(100),
            np.linspace(lat0, lat1, 100),
            lat1*np.ones(100),
            np.linspace(lat1, lat0, 100)
            ]).ravel()
        lonlatbox_kw = kw.pop('lonlatbox_kw', {})
        lonlatbox_color = kw.pop('lonlatbox_color', 'k')
        lonlatbox_color = lonlatbox_kw.pop('color', lonlatbox_color)
        plt.plot(lon_, lat_, color=lonlatbox_color, **lonlatbox_kw)

    # plot coast lines
    # load coast data
    lonlon, latlat = _load_coast()
    # fill continents or plot coast lines
    fill_continents = kw.pop('fill_continents', False)
    xticks = kw.pop('xticks', np.arange(0, 360, 60))
    yticks = kw.pop('yticks', np.arange(-90, 91, 30))
    if fill_continents:
        # indices of western and eastern boundaries
        ivec = np.arange(lonlon.size)[latlat==-83.83]
        i0,i1,i2,i3,i4,i5 = ivec
        # correct the Antarctic map by adding coast points at the south pole
        lonlon = np.hstack((
            lonlon[:i0],
            lonlon[i0], lonlon[i0:i1+1], lonlon[i1],
            lonlon[i1+1:i2],
            lonlon[i2], lonlon[i2:i3+1], lonlon[i3],
            lonlon[i3+1:i4],
            lonlon[i4], lonlon[i4:i5+1], lonlon[i5],
            lonlon[i5+1:]))
        latlat = np.hstack((
            latlat[:i0],
            -90, latlat[i0:i1+1], -90,
            latlat[i1+1:i2],
            -90, latlat[i2:i3+1], -90,
            latlat[i3+1:i4],
            -90, latlat[i4:i5+1], -90,
            latlat[i5+1:]))
        continents_color = kw.pop('continents_color', '0.5')
        plt.fill(lonlon, latlat,
            color=continents_color, edgecolor='none',
            **kw)
    else:
        # plot coastlines
        coastlines_color = kw.pop('coastlines_color', '0.66')
        coastlines_width = kw.pop('linewidth', 1)
        coastlines_width = kw.pop('coastlines_width', coastlines_width)
        plt.plot(lonlon, latlat,
            color=coastlines_color, linewidth=coastlines_width,
            **kw)

    xticks2lon(xticks)
    yticks2lat(yticks)
    plt.xlim(min(lon_edge), max(lon_edge))
    plt.ylim(min(lat_edge), max(lat_edge))
    return
#
# ######## plot data on basemap
def xyplot(data, x=None, y=None, **kw):
    '''Show 2D data in a x-y plane, which can also be a lon-lat plane.

    #### Input
        data: 2D array.
        x: vector.
        y: vector.
        kw: optional parameters.

    ------
    Optional parameters:
        add_basemap: boolean; default is False.
        basemap_kw: parameters in plotting the basemap.
        fill_continents: boolean; default is False.
        lonlatbox: length-4 array, or None (default).

        plot_type: 'pcolor', 'pcolormesh', 'imshow', 'conturf', 'contour',
            'contourf+', 'quiver', or 'hatch'.
        cmap: color map.
        units: units.
        clim: color limit.
        cbar_type: 'vertical', 'v', 'horizontal', or 'h'.
        cbar_kw: dict parameters in plotting color bar.
        cbar_extend: boolean.
        cbar_size: '2.5%' for vertial cbar; '5%' for horizontal cbar.
        cbar_pad: 0.1 for vertical cbar; 0.4 for horizontal cbar.
        hide_cbar: boolean; whether to show cbar.

    ------
    imshow parameters:
        origin:
        extent:
        interpolation: 'nearest'(default) or 'bilnear'.

    ------
    contour parameters:
        colors: 'gray'(default) or other colors.
        levels:
        label_contour: boolean.

    ------
    quiver parameters:
        stride:
        stride_lon:
        stride_lat:
        quiver_color:
        scale:
        hide_qkey:
        qkey_kw:
        qkey_X:
        qkey_Y:
        qkey_U:
        qkey_label:
        qkey_labelpos:

    ------
    hatch plot
        hatches: default is ['///']; can be ['/'], ['//'], ['////'], etc.


        '''
    # data prepare
    input_data_have_two_components = ( isinstance(data, tuple)
        or isinstance(data, list) )
    if input_data_have_two_components:
        # input data is (u,v) or [u, v] where u, v are ndarray and two components of a vector
        assert len(data) == 2,'quiver data must contain only two componets u and v'
        u = data[0].squeeze()
        v = data[1].squeeze()
        assert u.ndim == 2, 'u component data must be two dimensional'
        assert v.ndim == 2, 'v component data must be two dimensional'
        data = np.sqrt( u**2 + v**2 ) # calculate wind speed
    else:# input data is a ndarray
        data = data.squeeze()
        assert data.ndim == 2, 'Input data must be two dimensional!'
    Ny, Nx = data.shape

    # x, y and basemap
    add_basemap = kw.pop('add_basemap', False)
    if add_basemap:
        if x is None:
            x = np.linspace(0, 360, Nx+1)[0:-1]
        if y is None:
            y = ( np.linspace(-90, 90, Ny+1)[:-1]
                + np.linspace(-90, 90, Ny+1)[1:] )/2.0
        basemap_kw = kw.pop('basemap_kw', {})
        fill_continents = kw.pop('fill_continents', False)
        fill_continents = basemap_kw.pop('fill_continents', fill_continents)
        lonlatbox = kw.pop('lonlatbox', None)
        lolatbox = basemap_kw.pop('lonlatbox', lonlatbox)
        mapplot(x, y, fill_continents=fill_continents,
            lonlatbox=lonlatbox, **basemap_kw)
    else:
        if x is None:
            x = np.arange(Nx)
        if y is None:
            y = np.arange(Ny)
    X, Y = np.meshgrid(x, y)
    x_edge, y_edge = _get_grid_edges(x, y)
    if add_basemap:
        y_edge[y_edge < -90] = -90
        y_edge[y_edge >  90] =  90



    #
    # ###### plot parameters
    # plot_type
    plot_type = kw.pop('plot_type', None)
    if plot_type is None:
        if input_data_have_two_components:
            plot_type = 'quiver'
        else:
            plot_type = 'pcolormesh'
            print('plot_type **** pcolormesh **** is used.')

    # cmap
    cmap = kw.pop('cmap', None)
    if cmap is None:
         zz_max = data.max()
         zz_min = data.min()
         if zz_min >=0:
             try:
                 cmap = plt.get_cmap('viridis')
             except:
                 cmap = plt.get_cmap('OrRd')
         elif zz_max<=0:
             try:
                 cmap = plt.get_cmap('viridis')
             except:
                 cmap = plt.get_cmap('Blues_r')
         else:
             cmap = plt.get_cmap('RdBu_r')

    # units
    units = kw.pop('units', '')
    # clim
    clim = kw.pop('clim', None)
    # colorbar
    if plot_type in ('pcolor', 'pcolormesh', 'contourf', 'imshow', 'contourf+'):
        cbar_type = kw.pop('cbar_type', 'vertical')
        cbar_kw = kw.pop('cbar_kw', {})
        cbar_extend = kw.pop('cbar_extend', 'neither')
        cbar_extend = cbar_kw.pop('extend', cbar_extend)

        if cbar_type in ('v', 'vertical'):
            cbar_size = kw.pop('cbar_size', '2.5%')
            cbar_size = cbar_kw.pop('size', cbar_size)
            cbar_pad = kw.pop('pad', 0.1)
            cbar_pad = cbar_kw.pop('pad', cbar_pad)
            cbar_position = 'right'
            cbar_orientation = 'vertical'
        elif cbar_type in ('h', 'horizontal'):
            # cbar = hcolorbar(units=units)
            cbar_size = kw.pop('cbar_size', '5%')
            cbar_size = cbar_kw.pop('size', cbar_size)
            cbar_pad = kw.pop('cbar_pad', 0.4)
            cbar_pad = cbar_kw.pop('pad', cbar_pad)
            cbar_position = 'bottom'
            cbar_orientation = 'horizontal'
        hide_cbar = kw.pop('hide_cbar', False)

    # xticks and yticks
    if add_basemap:
        xticks = kw.pop('xticks', np.arange(0, 360, 60))
        yticks = kw.pop('yticks', np.arange(-90, 91, 30))
    #
    # ###### start plot
    # pcolor
    if plot_type in ('pcolor',):
        plot_obj = plt.pcolor(x_edge, y_edge, data, cmap=cmap, **kw)
    # pcolormesh
    elif plot_type in ('pcolormesh',):
        plot_obj = plt.pcolormesh(x_edge, y_edge, data, cmap=cmap, **kw)
    # imshow
    elif plot_type in ('imshow',):
        if y_edge[-1] > y_edge[0]:
            origin = kw.pop('origin', 'lower')
        else:
            origin = kw.pop('origin', 'upper')
        extent = kw.pop('extent', [x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]])
        interpolation = kw.pop('interpolation', 'nearest')
        plot_obj = plt.imshow(data, origin=origin, cmap=cmap, extent=extent,
            interpolation=interpolation, **kw)
    # contourf
    elif plot_type in ('contourf',):
        extend = kw.pop('extend', 'both')
        plot_obj = plt.contourf(x, y, data, extend=extend, cmap=cmap, **kw)
    # contour
    elif plot_type in ('contour',):
        colors = kw.pop('colors', 'gray')
        if colors is not None:
            cmap = None
        plot_obj = plt.contour(x, y, data, cmap=cmap, colors=colors, **kw)
        label_contour = kw.pop('label_contour', False)
        if label_contour:
            plt.clabel(plot_obj,plot_obj.levels[::2],fmt='%.2G')
    # contourf+: contourf + contour
    elif plot_type in ('contourf+',):
        colors = kw.pop('colors', 'gray')
        extend = kw.pop('extend', 'both')
        plot_obj = plt.contourf(x, y, data, extend=extend, cmap=cmap, **kw)
        if colors is not None:
            cmap = None
        plt.contour(x, y, data, cmap=cmap, colors=colors, **kw)
    elif plot_type in ('quiver',):
        stride = kw.pop('stride', 1)
        stride_lon = kw.pop('stride_lon', stride)
        stride_lat = kw.pop('stride_lat', stride)
        x_ = x[::stride_lon] # subset of lon
        y_ = y[::stride_lat]
        u_ = u[::stride_lat, ::stride_lon]
        v_ = v[::stride_lat, ::stride_lon]
        quiver_color = kw.pop('quiver_color', 'g')
        quiver_scale = kw.pop('scale', None)
        hide_qkey = kw.pop('hide_qkey', False)
        if not hide_qkey:
            qkey_kw = kw.pop('qkey_kw', {})
            qkey_X = kw.pop('qkey_X', 0.85)
            qkey_X = qkey_kw.pop('X', qkey_X)
            qkey_Y = kw.pop('qkey_Y', 1.02)
            qkey_Y = qkey_kw.pop('Y', qkey_Y)
            qkey_U = kw.pop('qkey_U', 2)
            qkey_U = qkey_kw.pop('U', qkey_U)
            qkey_label = kw.pop('qkey_label', '{:g} '.format(qkey_U) + units)
            qkey_label = qkey_kw.pop('label', qkey_label)
            qkey_labelpos = kw.pop('qkey_labelpos', 'W')
            labelpos = qkey_kw.pop('labelpos', qkey_labelpos)
        # quiverplot
        plot_obj = plt.quiver(x_, y_, u_, v_, color=quiver_color,
            scale=quiver_scale, **kw)
        if not hide_qkey:
            # quiverkey plot
            plt.quiverkey(plot_obj, qkey_X, qkey_Y, qkey_U,
                label=qkey_label, labelpos=qkey_labelpos, **qkey_kw)

    # hatch plot
    elif plot_type in ('hatch', 'hatches'):
        hatches = kw.pop('hatches', ['///'])
        plot_obj = plt.contourf(x, y, data, colors='none', hatches=hatches,
            extend='both', **kw)
    else:
        print('Please choose a right plot_type from ("pcolor", "contourf", "contour")!')

    # clim
    if plot_type in ('pcolor', 'pcolormesh', 'imshow'):
        if clim is None:
            if isinstance(data,np.ma.core.MaskedArray):
                data1d = data.compressed()
            else:
                data1d = data.ravel()
            notNaNs = np.logical_not(np.isnan(data1d))
            data1d = data1d[notNaNs]
            a = np.percentile(data1d,2)
            b = np.percentile(data1d,98)
            if a * b < 0:
                b = max(abs(a), abs(b))
                a = -b
            clim = a, b
        else:
            pass
        plt.clim(clim)

    # colorbar
    if plot_type in ('pcolor', 'pcolormesh', 'contourf', 'imshow', 'contourf+'):
        colorbar(mappable=plot_obj, cbar_type=cbar_type, size=cbar_size,
            pad=cbar_pad, extend=cbar_extend, units=units, **cbar_kw)
        # remove the colorbar to avoid repeated colorbars
        if hide_cbar:
            cbar.remove()

    # xticks and yticks
    if add_basemap:
        xticks2lon(xticks)
        yticks2lat(yticks)

    # xlim and ylim
    # if plot_type in ('pcolormesh', 'pcolor', 'imshow'):
    #     plt.xlim(min(x_edge), max(x_edge))
    #     plt.ylim(min(y_edge), max(y_edge))
    # else:
    #     plt.xlim(min(x), max(x))
    #     plt.ylim(min(y), max(y))
    plt.xlim(min(x_edge), max(x_edge))
    plt.ylim(min(y_edge), max(y_edge))

    return plot_obj
