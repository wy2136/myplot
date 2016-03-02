# -*- coding: utf-8 -*-
"""
@author: Wenchang Yang (yang.wenchang@uci.edu)
"""
# from .mypyplot import vcolorbar, hcolorbar

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import addcyclic, shiftgrid
# from copy import deepcopy

# ###### universal 2-D plot function on the lon-lat plane
def geoplot(data=None, lon=None, lat=None, **kw):
    '''Show 2D data in a lon-lat plane.
    
    **** Input ****:
        data: array of shape (n_lat, n_lon), or [u_array, v_array]-like for (u,v) 
            data or None(default) when only plotting the basemap.
        lon: n_lon length vector or None(default).
        lat: n_lat length vector or None(default).
        kw: dict parameters related to basemap or plot functions.
    
    --------
    Basemap related parameters:
        proj or projection: map projection name ('cyl' default).
        lon_0: map center longitude (None as default).
        lat_0: map center latitude (None as default).
        boundinglat: latitude at the map out boundary (None as default).
        basemap_kw: dict parameter in the initialization of a Basemap.
        
        fill_continents: bool value (False as default).
        continents_color: color of continents ('0.66' as default).
        continents_kw: dict parameter used in the Basemap.fillcontinents method.
        coastlines_color: color of coast lines ('0.66' as default).
        coastlines_kw: dict parameter used in the Basemap.drawcoastlines method.
        parallels: parallels to be drawn (np.arange(-60, 61, 30) as default).
        parallels_color: color of parallels ('0.66' as default).
        parallels_kw: dict parameters used in the Basemap.drawparallels method.
        meridians: meridians to be drawn (np.arange(-180, 360, 60) as default).
        meridians_color: color of meridians ('0.66' as default).
        meridians_kw: dict parameters used in the Basemap.drawmeridians method.
    
    --------    
    Plot related parameters:
        plot_type: a string of plot type from ('pcolor', 'pcolormesh', 'imshow', 'contourf', 'contour', 'quiver', 'scatter') or None(default).
        cmap: pyplot colormap.
        clim: a tuple of colormap limit.
        cbar_type: colorbar type from options of ('v', 'vertical', 'h', 'horizontal').
        hide_cbar: bool value. 
        units: data units to be shown in the colorbar.
        plot_kw: dict parameters used in the plot functions.
    
    --------
    Quiver plot related parameters:
        scale: quiver scale.
        quiver_color: quiver color.
        quiver_kw: dict parameters used in the plt.quiver function.
        
        hide_qkey: bool value, whether to show the quiverkey plot.
        qkey_X: X parameter in the plt.quiverkey function.
        qkey_Y: Y parameter in the plt.quiverkey function.
        qkey_U: U parameter in the plt.quiverkey function.
        qkey_label: label parameter in the plt.quiverkey function.
        qkey_labelpos: labelpos parameter in the plt.quiverkey function.
        qkey_kw: dict parameters used in the plt.quiverkey function.
    
    --------
    Colorbar related parameters:
        hide_cbar: bool value, whether to show the colorbar.
        cbar_type: 'vertical'(shorten as 'v') or 'horizontal' (shorten as 'h').
        cbar_extend: extend parameter in the plt.colorbar function. 
            'neither' as default here.
        cbar_size: default '2.5%' for vertical colorbar, 
            '5%' for horizontal colorbar.
        cbar_pad: default 0.1 for vertical colorbar, 
            0.4 for horizontal colorbar.
        cbar_kw: dict parameters used in the plt.colorbar function.
        
    **** Returns ****
        basemap object if only basemap is plotted.
        plot object if data is shown.
        '''
        
    # #### basemap parameters
    # basemap kw parameter
    basemap_kw = kw.pop('basemap_kw', {})
    # projection
    proj = kw.pop('proj', 'cyl')
    proj = kw.pop('projection', proj) # projection overrides the proj parameter
    # short names for nplaea and splaea projections
    if proj in ('npolar', 'polar', 'np'):
        proj = 'nplaea'
    elif proj in ('spolar', 'sp'):
        proj = 'splaea'
        
    # lon_0
    lon_0 = kw.pop('lon_0', None)
    if lon_0 is None:
        if lon is not None:
            if np.isclose( np.abs(lon[-1] - lon[0]), 360 ):
                lon_0 = (lon[0] + lon[-2])/2.0
            else:
                lon_0 = (lon[0] + lon[-1])/2.0
        else:
            # dummy = np.linspace(0, 360, data.shape[1]+1)[0:-1]
            # lon_0 = ( dummy[0] + dummy[-1] )/2.0
            lon_0 = 180
        # elif proj in ('moll', 'cyl', 'hammer', 'robin'):
        #     lon_0 = 180
        # elif proj in ('ortho','npstere', 'nplaea', 'npaeqd', 'spstere', 'splaea', 'spaeqd'):
        #     lon_0 = 0 
    else: # lon_0 is specified
        if lon is not None and proj in ('moll', 'cyl'):
            # correct the lon_0 so that it is at the edge of a grid box
            lon_0_data = (lon[0] + lon[-1])/2.0
            d_lon = lon[1] - lon[0]
            d_lon_0 = lon_0 - lon_0_data
            lon_0 = float(int(d_lon_0 / d_lon)) * d_lon + lon_0_data
    
    # lat_0
    lat_0 = kw.pop('lat_0', None)
    if lat_0 is None:
        if lat is not None:
            lat_0 = ( lat[0] + lat[-1] )/2.0
        elif proj in ('ortho',):
            lat_0 = 45
    
    # lonlatcorner = (llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat)
    lonlatcorner = kw.pop('lonlatcorner', None)
    if lonlatcorner is not None:
        llcrnrlon = lonlatcorner[0]
        urcrnrlon = lonlatcorner[1]
        llcrnrlat = lonlatcorner[2]
        urcrnrlat = lonlatcorner[3]
    else:
        llcrnrlon = None
        urcrnrlon = None
        llcrnrlat = None
        urcrnrlat = None
    llcrnrlon = basemap_kw.pop('llcrnrlon', llcrnrlon)
    urcrnrlon = basemap_kw.pop('urcrnrlon', urcrnrlon)
    llcrnrlat = basemap_kw.pop('llcrnrlat', llcrnrlat)
    urcrnrlat = basemap_kw.pop('urcrnrlat', urcrnrlat)
    if llcrnrlon is None and urcrnrlon is None and llcrnrlat is None and urcrnrlat is None:
        lonlatcorner = None
    else:
        lonlatcorner = (llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat)  
           
    # boundinglat
    boundinglat = kw.pop('boundinglat', None)
    if boundinglat is None:
        if proj in ('npstere', 'nplaea', 'npaeqd'):
            boundinglat = 30
        elif proj in ('spstere', 'splaea', 'spaeqd'):
            boundinglat = -30
    
    # basemap round: True or False
    if proj in ('npstere', 'nplaea', 'npaeqd', 'spstere', 'splaea', 'spaeqd'):
        basemap_round = kw.pop('basemap_round', True)
    else:
        basemap_round = kw.pop('basemap_round', False)
    basemap_round = kw.pop('round', basemap_round)
    
    # base map
    proj = basemap_kw.pop('projection', proj)
    lon_0 = basemap_kw.pop('lon_0', lon_0)
    lat_0 = basemap_kw.pop('lat_0', lat_0)
    boundinglat = basemap_kw.pop('boundinglat', boundinglat)
    basemap_round = basemap_kw.pop('round', basemap_round)
    m = Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0, boundinglat=boundinglat,
        round=basemap_round, 
        llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon,
        llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
        **basemap_kw)
    
    # fill continents or plot coast lines
    fill_continents = kw.pop('fill_continents', False)
    if fill_continents:
        # use Basemap.fillcontinents method
        continents_kw = kw.pop('continents_kw', {})
        continents_color = kw.pop('continents_color', '0.5')
        continents_color = continents_kw.pop('color', continents_color)
        lake_color = kw.pop('lake_color', continents_color)
        lake_color = continents_kw.pop('lake_color', lake_color)
        m.fillcontinents(color=continents_color, lake_color=lake_color,
            **continents_kw)
    else:
        # use Basemap.drawcoastlines method
        coastlines_kw = kw.pop('coastlines_kw', {})
        coastlines_color = kw.pop('coastlines_color', '0.66')
        coastlines_color = coastlines_kw.pop('color', coastlines_color)
        m.drawcoastlines(color=coastlines_color)
    
    # parallels
    parallels_kw = kw.pop('parallels_kw', {}) 
    parallels = kw.pop('parallels', np.arange(-90,91,30))
    parallels = parallels_kw.pop('parallels', parallels)
    parallels_color = kw.pop('parallels_color', '0.75')
    parallels_color = parallels_kw.pop('color', parallels_color)
    if proj in ('cyl',):
        parallels_labels = kw.pop('parallels_labels', [1,0,0,0])
    else:
        parallels_labels = kw.pop('parallels_labels', [0,0,0,0])
    parallels_labels = parallels_kw.pop('labels', parallels_labels)
    m.drawparallels(parallels, color=parallels_color, labels=parallels_labels,
        **parallels_kw)
    
    # meridians
    meridians_kw = kw.pop('meridians_kw', {})
    meridians = kw.pop('meridians', np.arange(-180, 360, 60))
    meridians = meridians_kw.pop('meridians', meridians)
    meridians_color = kw.pop('meridians_color', parallels_color)
    meridians_color = meridians_kw.pop('color', meridians_color)
    if proj in ('cyl',):
        meridians_labels = kw.pop('meridians_labels', [0,0,0,1])
    # elif proj in ('npstere', 'nplaea', 'npaeqd', 'spstere', 'splaea', 'spaeqd'):
    #     meridians_labels = kw.pop('meridians_labels', [1,0,0,0])
    else:
        meridians_labels = kw.pop('meridians_labels', [0,0,0,0])
    meridians_labels = meridians_kw.pop('labels', meridians_labels)
    m.drawmeridians(meridians, color=meridians_color, labels=meridians_labels,
        **meridians_kw)
    
    # lonlatbox
    lonlatbox = kw.pop('lonlatbox', None)
    if lonlatbox is not None:
        lonlon = np.array([
            np.linspace(lonlatbox[0], lonlatbox[1], 100),
            lonlatbox[1]*np.ones(100),
            np.linspace(lonlatbox[1], lonlatbox[0], 100),
            lonlatbox[0]*np.ones(100) 
            ]).ravel()
        latlat = np.array([
            lonlatbox[2]*np.ones(100),
            np.linspace(lonlatbox[2], lonlatbox[3], 100),
            lonlatbox[3]*np.ones(100),
            np.linspace(lonlatbox[3], lonlatbox[2], 100) 
            ]).ravel()
        lonlatbox_color = kw.pop('lonlatbox_color', 'k')
        lonlatbox_kw = kw.pop('lonlatbox_kw', {})
        m.plot(lonlon, latlat, latlon=True, color=lonlatbox_color, **lonlatbox_kw)
    
    # #### stop here and return the map object if data is None
    if data is None:
        return m
    
    # data prepare
    input_data_have_two_components = isinstance(data, tuple) or isinstance(data, list)
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
    
    # lon
    if lon is None:
        # lon = np.linspace(0, 360, data.shape[1]+1)[0:-1]
        # lon_edge = np.hstack((
        #     lon[0]*2 - lon[1],
        #     lon,
        #     lon[-1]*2 - lon[-2]))
        # lon_edge = ( lon_edge[:-1] + lon_edge[1:] )/2.0
        lon_edge = np.linspace(lon_0-180, lon_0+180, data.shape[1]+1)
        lon = (lon_edge[:-1] + lon_edge[1:])/2.0
    else:# lon is specified
        if np.isclose( np.abs(lon[-1] - lon[0]), 360 ):
            # first and last longitude point to the same location: remove the last longitude
            lon = lon[:-1]
            data = data[:, :-1]
            if input_data_have_two_components:
                u = u[:, :-1]
                v = v[:, :-1]
        if (not np.isclose(lon_0, (lon[0] + lon[-1])/2.0) 
            and proj in ('moll', 'cyl', 'hammer', 'robin')
            ): 
            # lon_0 not at the center of lon, need to shift grid
            lon_west_end = lon_0 - 180 + (lon[1] - lon[0])/2.0 # longitude of west end
            # make sure the longitude of west end within the lon
            if lon_west_end < lon[0]:
                lon_west_end += 360
            elif lon_west_end > lon[-1]:
                lon_west_end -= 360
            data, lon_shift = shiftgrid(lon_west_end, data, lon, start=True)
            if input_data_have_two_components:
                u, lon_shift = shiftgrid(lon_west_end, u, lon, start=True)
                v, lon_shift = shiftgrid(lon_west_end, v, lon, start=True)
            lon = lon_shift
            if lon[0]<-180:
                lon += 360
            elif lon[-1]>=540:
                lon -= 360
        lon_hstack = np.hstack((2*lon[0] - lon[1], lon, 2*lon[-1] - lon[-2]))
        lon_edge = (lon_hstack[:-1] + lon_hstack[1:])/2.0
    
    # lat
    if lat is None:
        lat_edge = np.linspace(-90, 90, data.shape[0]+1)
        lat = (lat_edge[:-1] + lat_edge[1:])/2.0
    else:
        lat_hstack = np.hstack((2*lat[0] - lat[1], lat, 2*lat[-1] - lat[-2]))
        lat_edge = (lat_hstack[:-1] + lat_hstack[1:])/2.0
        lat_edge[lat_edge>90] = 90
        lat_edge[lat_edge<-90] = -90
    Lon, Lat = np.meshgrid(lon, lat)
    X, Y = m(Lon, Lat)
    Lon_edge, Lat_edge = np.meshgrid(lon_edge, lat_edge)
    X_edge, Y_edge = m(Lon_edge, Lat_edge)
    
    
    # ###### plot parameters
    # plot_type
    plot_type = kw.pop('plot_type', None)
    if plot_type is None:
        if input_data_have_two_components:
            plot_type = 'quiver'
        # elif ( proj in ('cyl',)
        #     and lonlatcorner is None
        #     ):
        #     plot_type = 'imshow'
        #     print('plot_type **** imshow **** is used.')
        elif proj in ('nplaea', 'splaea', 'ortho'):
            # pcolormesh has a problem for these projections
            plot_type = 'pcolor'
            print('plot_type **** pcolor **** is used.')
        else:
            plot_type = 'pcolormesh'
            print ('plot_type **** pcolormesh **** is used.')
    
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
    units = kw.pop('units', None)
    if units is None:
        units = ''
    
    # colorbar parameters
    if plot_type in ('pcolor', 'pcolormesh', 'contourf', 'contourf+', 'imshow'):
        cbar_type = kw.pop('cbar_type', 'vertical')
        cbar_kw = kw.pop('cbar_kw', {})
        cbar_extend = kw.pop('cbar_extend', 'neither')
        cbar_extend = cbar_kw.pop('extend', cbar_extend)
        
        if cbar_type in ('v', 'vertical'):
            cbar_size = kw.pop('cbar_size', '2.5%')
            cbar_size = cbar_kw.pop('size', cbar_size)
            cbar_pad = kw.pop('cbar_pad', 0.1)
            cbar_pad = cbar_kw.pop('pad', cbar_pad)
            cbar_position = 'right'
            cbar_orientation = 'vertical'
        elif cbar_type in ('h', 'horizontal'):
            # cbar = hcolorbar(units=units)
            cbar_size = kw.pop('cbar_size', '5%')
            cbar_size = cbar_kw.pop('size', cbar_size)
            cbar_pad = kw.pop('cbar_pad', 0.4)
            cbar_pad = cbar_kw.pop('pad', cbar_pad)
            cbar_position = 'bottome'
            cbar_orientation = 'horizontal'
        hide_cbar = kw.pop('hide_cbar', False)
    
    # ###### plot
    # pcolor
    if plot_type in ('pcolor',):
        plot_obj = m.pcolor(X_edge, Y_edge, data, cmap=cmap, **kw) 
    # pcolormesh
    elif plot_type in ('pcolormesh',):
        plot_obj = m.pcolormesh(X_edge, Y_edge, data, cmap=cmap, **kw)
    # imshow
    elif plot_type in ('imshow',):
        if Y_edge[-1,0] > Y_edge[0,0]:
            origin = kw.pop('origin', 'lower')
        else:
            origin = kw.pop('origin', 'upper')
        extent = kw.pop('extent', [X_edge[0,0], X_edge[0,-1], Y_edge[0,0], Y_edge[-1,0]])
        interpolation = kw.pop('interpolation', 'nearest')
        plot_obj = m.imshow(data, origin=origin, cmap=cmap, extent=extent,
            interpolation=interpolation, **kw)
    # contourf
    elif plot_type in ('contourf',):
        if proj in ('ortho','npstere', 'nplaea', 'npaeqd', 'spstere', 'splaea', 'spaeqd'):
            data, lon_ = addcyclic(data, lon)
            Lon, Lat = np.meshgrid(lon,lat)
            X, Y = m(Lon, Lat)
        extend = kw.pop('extend', 'both')
        # levels = kw.pop('levels', None)
        plot_obj = m.contourf(X, Y, data, extend=extend, cmap=cmap, **kw)
    # contour
    elif plot_type in ('contour',):
        if proj in ('ortho','npstere', 'nplaea', 'npaeqd', 'spstere', 'splaea', 'spaeqd'):
            data, lon = addcyclic(data, lon)
            Lon, Lat = np.meshgrid(lon,lat)
            X, Y = m(Lon, Lat)
        colors = kw.pop('colors', 'gray')
        if colors is not None:
            cmap = None
        plot_obj = m.contour(X, Y, data, cmap=cmap, colors=colors, **kw)
    elif plot_type in ('contourf+',):
        if proj in ('ortho','npstere', 'nplaea', 'npaeqd', 'spstere', 'splaea', 'spaeqd'):
            data, lon = addcyclic(data, lon)
            Lon, Lat = np.meshgrid(lon,lat)
            X, Y = m(Lon, Lat)
        extend = kw.pop('extend', 'both')
        plot_obj = m.contourf(X, Y, data, extend=extend, cmap=cmap, **kw)
        colors = kw.pop('colors', 'gray')
        if colors is not None:
            cmap = None
        m.contour(X, Y, data, cmap=cmap, colors=colors, **kw)
    elif plot_type in ('quiver',):
        stride = kw.pop('stride', 1)
        stride_lon = kw.pop('stride_lon', stride)
        stride_lat = kw.pop('stride_lat', stride)
        lon_ = lon[::stride_lon] # subset of lon
        lat_ = lat[::stride_lat]
        u_ = u[::stride_lat, ::stride_lon]
        v_ = v[::stride_lat, ::stride_lon]
        Lon_, Lat_ = np.meshgrid(lon_, lat_)
        u_rot, v_rot, X_, Y_ = m.rotate_vector(
            u_, v_, Lon_, Lat_, returnxy=True
        )
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
        plot_obj = m.quiver(X_, Y_, u_rot, v_rot, color=quiver_color, 
            scale=quiver_scale, **kw)
        if not hide_qkey:
            # quiverkey plot
            plt.quiverkey(plot_obj, qkey_X, qkey_Y, qkey_U, 
                label=qkey_label, labelpos=qkey_labelpos, **qkey_kw)
    # scatter
    elif plot_type in ('scatter',):
        L = data.astype('bool')
        marker_color = kw.pop('marker_color', 'k')
        plot_obj = m.scatter(X[L], Y[L], color=marker_color, **kw)
    else:
        print('Please choose a right plot_type from ("pcolor", "contourf", "contour")!')
    
    # clim
    clim = kw.pop('clim', None)
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
    if plot_type in ('pcolor', 'pcolormesh', 'contourf', 'contourf+', 'imshow'):
        ax_current = plt.gca()
        divider = make_axes_locatable(ax_current)
        cax = divider.append_axes(cbar_position, size=cbar_size, pad=cbar_pad)
        cbar = plt.colorbar(plot_obj, cax=cax, extend=cbar_extend,
            orientation=cbar_orientation, **cbar_kw)
        # units position
        if units is not None and units != '':
            if cbar_type in ('v', 'vertical'):
                # put the units on the top of the vertical colorbar
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.set_xlabel(units)
            else:
                cbar.ax.yaxis.set_label_position('right')
                cbar.ax.set_ylabel(units, rotation=0, ha='left', va='center')
        # remove the colorbar to avoid repeated colorbars
        if hide_cbar:
            cbar.remove()
        # set back the main axes as the current axes
        plt.sca(ax_current)
    

    return plot_obj
