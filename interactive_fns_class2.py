import glob
import numpy as np
import pykrige
import xarray as xr
import rioxarray
import scipy.ndimage as ndimage
import geopandas
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from tqdm import tqdm

import ipywidgets as widgets
from ipywidgets import interactive_output, Layout
from IPython.display import clear_output, display
import datesConverter
import warnings


class InteractiveRainfallBlending():

    def __init__(self, path_to_latlon, path_to_stations,
                 path_to_geojson, chirp_floc,
                 chirps_floc, blend_floc):

        # Adding paths to class variables.
        self.path_to_latlon = path_to_latlon
        self.path_to_stations = path_to_stations
        self.path_to_geojson = path_to_geojson
        self.chirp_floc = chirp_floc
        self.chirps_floc = chirps_floc
        self.blend_floc = blend_floc

        # Reading files in
        # Latlon DataFrame
        self.latlon = pd.read_csv(path_to_latlon, index_col=0)
        # Station data DataFrame
        self.stations_1 = pd.read_csv(path_to_stations, index_col=0,
                                      parse_dates=['Date'], dayfirst=True)
        # World's country borders geojson/DataFrame
        self.world_shp = geopandas.read_file(self.path_to_geojson)

        # Initilising variable that will be defined later
        self.chirps_path = ''
        self.chirp_df = pd.DataFrame()
        self.stations = pd.DataFrame()
        self.chirp_df_U = pd.DataFrame()
        self.ratios = pd.DataFrame()
        self.res = pd.DataFrame()

    def check_date_in_data(self, chirp_fpath):
        '''
        Function that takes the CHIRP file path, extracts the date
        and checks if that date is within the date-range of the
        station data.
        '''

        dekad = chirp_fpath[-12:-4]
        dt = datesConverter.str_to_dek(dekad)
        dt = pd.Timestamp(dt)

        if self.stations_1.index[-1] < dt:
            raise IndexError('CHIRP dekad is not contained in station data')

        if self.stations_1.index[0] > dt:
            raise IndexError('CHIRP dekad is not contained in station data')


    def check_date_range(self, start_date, end_date):
        if start_date > end_date:
            raise ValueError('End date must be after start date.')

        elif self.stations_1.index[-1] < start_date:
            raise IndexError('Start date is not contained in station data. '
                                +'Choose a date before {}'.format(self.stations_1.index[-1].date()))

        elif self.stations_1.index[0] > end_date:
            raise IndexError('End_date is not contained in station data. '
                                +'Choose a date after {}'.format(self.stations_1.index[0].date()))
        else:
            pass

        if (self.stations_1.index[0] > start_date) or (self.stations_1.index[-1] < end_date):
            warnings.warn('\nWarning: date range exceeds station data coverage '
                            +'({0} to {1}), all data being Krigged.'.format(self.stations_1.index[0].date(),
                                                                            self.stations_1.index[-1].date()))
        else:
            pass


    def get_chirps(self, chirp_fpath):
        '''
        Function takes CHIRP filepath, and returns fpath of matching CHIRPS.
        '''
        self.chirps_path = glob.glob(self.chirps_floc+'*'+chirp_fpath.split('\\')[1][6:])[0]
        return self.chirps_path


    def preprocess_data(self, chirp_fpath, outlier_threshold):
        '''
        Function that prepares the data for Kriging.
        '''

        #region Step 1: Getting CHIRPS data at station locations.
        dekad = chirp_fpath[-12:-4]
        dt = datesConverter.str_to_dek(dekad)
        dt = pd.Timestamp(dt)
        chirp_df = pd.DataFrame(index=[dt], columns=self.latlon.index)

        with rio.open(chirp_fpath) as img:
            raster = img.read()[0]

            vals = []
            for l in self.latlon.index:

                # Getting coordinates
                coord = rio.transform.rowcol(img.transform, self.latlon.loc[l].Longitude,
                                             self.latlon.loc[l].Latitude)
                vals.append(raster[coord[0], coord[1]])

        chirp_df.loc[dt] = vals
        chirp_df.index = chirp_df.index.rename('Date')

        intersection = self.stations_1.index.intersection(chirp_df.index)
        intersection = intersection.sort_values()

        self.stations = self.stations_1.loc[intersection]
        self.chirp_df = chirp_df.loc[intersection]
        #endregion

        #region Step 2: Removing outliers.
        threshold = outlier_threshold

        diff = np.abs(self.stations - self.chirp_df)
        self.stations[diff > threshold] = np.nan
        #endregion

        #region Step 3: Unbiasing
        self.chirp_df_U = self.chirp_df.copy()
        ratio = []

        for dt in self.chirp_df.index:
            ch = self.chirp_df.loc[dt]
            st = self.stations.loc[dt]

            if np.std(st) > 0:
                ch_ = ch[np.abs((self.stations.loc[dt] - np.mean(st))/np.std(st)) < 3]
                st_ = st[np.abs((self.stations.loc[dt] - np.mean(st))/np.std(st)) < 3]
            else:
                ch_ = ch
                st_ = st

            c = 1
            r = (np.mean(st_) + c) / (np.mean(ch_) + c)
            ratio.append(round(r, 3))

            self.chirp_df_U.loc[dt] = ch * r

        self.ratios = pd.DataFrame(data={'ratio': ratio}, index=self.chirp_df.index)
        #endregion

        #region Step 4: Residuals
        self.res = self.chirp_df_U - self.stations
        #endregion


    def do_blending(self, chirp_fpath, var_p1, var_p2, var_p3, var_model,
                    n_points, sigma):
        '''
        Function that performs Kriging.
        '''

        dekad = chirp_fpath[-12:-4]
        dt = datesConverter.str_to_dek(dekad)
        dt = pd.Timestamp(dt)

        z = self.res.loc[dt].dropna()
        x = self.latlon.loc[z.index]['Longitude'].values
        y = self.latlon.loc[z.index]['Latitude'].values
        z = z.values

        krig = pykrige.ok.OrdinaryKriging(x, y, z,
                                          variogram_model=var_model,
                                          variogram_parameters=[var_p1, var_p2, var_p3],
                                          coordinates_type='geographic')

        # Getting raster grid
        da = xr.open_rasterio(chirp_fpath)
        xp, yp = np.meshgrid(da['x'], da['y'])
        xp, yp = xp[0], np.array([yp[i][0] for i in range(len(yp))])

        # Kriging
        if n_points is False:
            kriged_result = krig.execute(style='grid', xpoints=xp, ypoints=yp)[0].data
        else:
            try:
                kriged_result = krig.execute(style='grid', xpoints=xp, ypoints=yp,
                                             backend='C', n_closest_points=n_points)[0].data
            except: # If less than 10 points
                kriged_result = krig.execute(style='grid', xpoints=xp, ypoints=yp)[0].data


        # Smoothing residuals
        if sigma is not False:
            kriged_result = ndimage.gaussian_filter(kriged_result, sigma=sigma)

        # Unbiasing
        da = da * self.ratios.loc[dt]['ratio']

        # Producing blended raster: removing kriged residuals from CHIRP raster
        blended_CHIRP = da - kriged_result

        # Converting to int and setting all values<0 to 0
        bCH = blended_CHIRP.values
        bCH[bCH < 0] = 0
        blended_CHIRP.values = bCH.astype(int)


        # Save rasters as tiff
        tif_name = (self.blend_floc + chirp_fpath.split('\\')[1][:3]
                    + 'rfb' + datesConverter.dek_to_str(dt) + '.tif')
        blended_CHIRP.rio.to_raster(tif_name)

        return tif_name


    def overlay(self, blended,
                chirp, chirps, do_chirp,
                mask_country,
                textsize, plotsize):
        '''
        Function that loads the relevant rasters and plots them.
        '''

        if mask_country is False:
            raise ValueError('Please select a country name')
        country_geometry = self.world_shp[self.world_shp.CNTRY_NAME == mask_country].geometry
        # Loading tifs

        with rio.open(blended) as img:
            if mask_country is not False:
                out_image, _ = rio.mask.mask(img, country_geometry,
                                             nodata=-9999, crop=True)
                raster_b = out_image[0]
            else:
                raster_b = img.read()[0]

        with rio.open(chirp) as img:
            if mask_country is not False:
                out_image, _ = rio.mask.mask(img, country_geometry,
                                             nodata=-9999, crop=True)
                raster = out_image[0]
            else:
                raster = img.read()[0]

        with rio.open(chirps) as img:
            if mask_country is not False:
                out_image, _ = rio.mask.mask(img, country_geometry,
                                             nodata=-9999, crop=True)
                chirps_raster = out_image[0]
            else:
                chirps_raster = img.read()[0]

        # Masking data outside of country of interest.
        if mask_country:
            raster_b = np.ma.masked_where(chirps_raster == -9999, raster_b)
            raster = np.ma.masked_where(chirps_raster == -9999, raster)
            chirps_raster = np.ma.masked_where(chirps_raster == -9999, chirps_raster)

        # Getting date of file as string (YYYY-MM-DD).
        date_str = blended[-12:-8]+'-'+blended[-8:-6]+'-'
        if blended[-6:-4] == 'd1':
            date_str = date_str + '10'
        elif blended[-6:-4] == 'd2':
            date_str = date_str + '20'
        elif blended[-6:-4] == 'd3':
            date_str = date_str + '28'
        else:
            print('Something is wrong.')

        # Extracting station data at specific time
        lats, lons, vals = [], [], []
        for i in self.stations.columns:
            lat = self.latlon.loc[i].Latitude
            lon = self.latlon.loc[i].Longitude
            value = self.stations[i].loc[date_str]

            lats.append(lat)
            lons.append(lon)
            vals.append(value)

        # Defining colourmap
        cmap_rgba = [[255, 255, 255, 255],
                     [250, 243, 210, 255],
                     [218, 227, 161, 255],
                     [160, 199, 135, 255],
                     [104, 171, 121, 255],
                     [155, 234, 250, 255],
                     [0, 177, 222, 255],
                     [0, 90, 230, 255],
                     [0, 0, 200, 255],
                     [160, 0, 250, 255],
                     [250, 120, 250, 255],
                     [255, 196, 238, 255]]
        cmap = mpl.colors.ListedColormap(np.array(cmap_rgba)/255)

        bounds = [1, 3, 10, 20, 30, 40, 60, 80, 100, 120, 150, 200, 1000]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # Making plot
        # Ensuring that the plot is the right size
        corners = country_geometry.bounds.values[0]
        corners = corners[[0, 2, 1, 3]] # Rearranging to be [minlon, maxlon, minlat, maxlat]

        fig = plt.figure(figsize=(plotsize, plotsize), dpi=150)
        axs = [plt.subplot(3, 1, i+1) for i in range(3)]

        axs[0].imshow(raster, extent=corners,
                      cmap=cmap, norm=norm)

        blended_im = axs[1].imshow(raster_b, extent=corners,
                                   cmap=cmap, norm=norm)

        axs[2].imshow(chirps_raster, extent=corners,
                      cmap=cmap, norm=norm)

        if do_chirp:
            axs[0].scatter(lons, lats, c='k', s=2, marker=',')
            [axs[0].text(lon, lat, str(val), fontsize=textsize)
             for lat, lon, val in zip(lats, lons, vals)]

        axs[2].set_aspect('equal')

        [ax.set_title(title) for ax, title in zip(axs, ['CHIRP', 'Blended', 'CHIRPS'])]
        [ax.set_ylabel('Latitude') for ax in axs]
        axs[2].set_xlabel('Longitude')

        plt.tight_layout()

        cbax = fig.add_axes((1, 0.15, 0.02, 0.7))
        plt.colorbar(blended_im, cax=cbax,
                     extend='max', ticks=bounds[:-1])

        plt.savefig('chirp_blended_chirps.pdf')
        plt.show()


    def interactive_plot(self, chirp_fpath,
                         var_p1, var_p2, var_p3, var_model,
                         n_points, sigma,
                         do_chirp,
                         mask_country,
                         textsize, plotsize, outlier_threshold,
                         krig_all_flag, start_date, end_date):
        '''
        Function that call the relevant data processing and plotting
        functions so that the widgets can affect the results.
        '''

        clear_output()

        self.check_date_in_data(chirp_fpath)

        self.preprocess_data(chirp_fpath, outlier_threshold)

        # Find CHIRPS file for CHIRP date
        chirps_fpath = self.get_chirps(chirp_fpath)

        # Interactive selection/specification of variogram model type/parameters
        # Run Kriging with above
        blended_fpath = self.do_blending(chirp_fpath,
                                         var_p1, var_p2, var_p3, var_model,
                                         n_points, sigma)
        # Plot CHIRP/BLENDED/CHIRPS data with interative options
            # Which panel/s to display station data
        self.overlay(blended=blended_fpath,
                     chirp=chirp_fpath, chirps=chirps_fpath, do_chirp=do_chirp,
                     mask_country=mask_country,
                     textsize=textsize, plotsize=plotsize)

        if krig_all_flag is True:

            # if start_date > end_date:
            #     raise ValueError('End date must be after start date.')

            # elif self.stations_1.index[-1] < start_date:
            #     raise IndexError('Start date is not contained in station data. '
            #                      +'Choose a date before {}'.format(self.stations_1.index[-1].date()))

            # elif self.stations_1.index[0] > end_date:
            #     raise IndexError('End_date is not contained in station data. '
            #                      +'Choose a date after {}'.format(self.stations_1.index[0].date()))
            # else:
            #     pass

            # if (self.stations_1.index[0] > start_date) or (self.stations_1.index[-1] < end_date):
            #     warnings.warn('\nWarning: date range exceeds station data coverage '
            #                   +'({0} to {1}), all data being Krigged.'.format(self.stations_1.index[0].date(),
            #                                                                  self.stations_1.index[-1].date()))
            # else:
            #     pass
            self.check_date_range(start_date, end_date)

            self.krig_all(outlier_threshold, start_date, end_date,
                          var_p1, var_p2, var_p3, var_model, n_points, sigma)


    def do_interact(self):
        '''
        Function that defines all of the widgets and call the
        function that handles the data processing and plotting.
        '''
        fname_dropdown = widgets.Dropdown(options=glob.glob(self.chirp_floc+'*.tif'),
                                          disabled=False,
                                          layout=Layout(width='250px'))

        cname_dropdown = widgets.Dropdown(options=[False] + list(geopandas.read_file(self.path_to_geojson).CNTRY_NAME.values),
                                          disabled=False,
                                          layout=Layout(width='100px'))

        var_sill_w = widgets.FloatText(value=1, disabled=False,
                                       layout=Layout(width='100px'))
        var_range_w = widgets.FloatText(value=1, disabled=False,
                                        style={'description_width': 'initial'},
                                        layout=Layout(width='100px'))
        var_nugg_w = widgets.FloatText(value=1, disabled=False,
                                       style={'description_width': 'initial'},
                                       layout=Layout(width='100px'))
        var_model_type = widgets.Dropdown(options=['spherical', 'exponential'], disabled=False,
                                          layout=Layout(width='100px'))
        n_points_w = widgets.IntText(value=10, disabled=False,
                                     layout=Layout(width='100px'))
        sigma_w = widgets.IntText(value=2, disabled=False,
                                  layout=Layout(width='100px'))


        stat_on_chirp = widgets.Checkbox(value=True, disabled=False, indent=False,
                                         layout=Layout(width='100px'))

        textsize_w = widgets.IntText(value=5, disabled=False,
                                     layout=Layout(width='100px'))
        plotsize_w = widgets.IntText(value=6, disabled=False,
                                     layout=Layout(width='100px'))

        outlier_thresh_w = widgets.FloatText(value=400, disabled=False,
                                             layout=Layout(width='100px'))

        krig_all_w = widgets.Checkbox(value=False, disabled=False, indent=False,
                                      layout=Layout(width='100px'))

        dates_4_w = [pd.Timestamp(datesConverter.str_to_dek(dekad[-12:-4])).date()
                     for dekad in np.sort(glob.glob(self.chirp_floc+'*.tif'))]

        start_date_w = widgets.Dropdown(options=dates_4_w,
                                        disabled=False,
                                        layout=Layout(width='150px'))

        end_date_w = widgets.Dropdown(options=dates_4_w,
                                      disabled=False,
                                      layout=Layout(width='150px'))

        grid_ui = widgets.GridspecLayout(5, 6, height='200px', width='1000px')

        grid_ui[0, 0] = widgets.Label('File name:')
        grid_ui[0, 1] = fname_dropdown

        grid_ui[0, 2] = widgets.Label('Overlay stations:')
        grid_ui[0, 3] = stat_on_chirp

        grid_ui[0, 4] = widgets.Label('Country Name:')
        grid_ui[0, 5] = cname_dropdown

        grid_ui[1, 0] = widgets.Label('Variogram sill:')
        grid_ui[1, 2] = widgets.Label('Variogram range:')
        grid_ui[1, 4] = widgets.Label('Variogram nugget:')

        grid_ui[1, 1] = var_sill_w
        grid_ui[1, 3] = var_range_w
        grid_ui[1, 5] = var_nugg_w

        grid_ui[2, 0] = widgets.Label('Model type:')
        grid_ui[2, 2] = widgets.Label('N points:')
        grid_ui[2, 4] = widgets.Label('Gauss. Sigma:')

        grid_ui[2, 1] = var_model_type
        grid_ui[2, 3] = n_points_w
        grid_ui[2, 5] = sigma_w

        grid_ui[3, 0] = widgets.Label('Text size:')
        grid_ui[3, 2] = widgets.Label('Plot size:')
        grid_ui[3, 4] = widgets.Label('Outlier threshold:')

        grid_ui[3, 1] = textsize_w
        grid_ui[3, 3] = plotsize_w
        grid_ui[3, 5] = outlier_thresh_w

        grid_ui[4, 0] = widgets.Label('Start date:')
        grid_ui[4, 2] = widgets.Label('End_date:')
        grid_ui[4, 4] = widgets.Label('Perform Krig:')

        grid_ui[4, 1] = start_date_w
        grid_ui[4, 3] = end_date_w
        grid_ui[4, 5] = krig_all_w

        out = interactive_output(self.interactive_plot,
                                 {'chirp_fpath': fname_dropdown,
                                  'var_p1': var_sill_w,
                                  'var_p2': var_range_w,
                                  'var_p3': var_nugg_w,
                                  'var_model': var_model_type,
                                  'n_points': n_points_w,
                                  'sigma': sigma_w,
                                  'do_chirp': stat_on_chirp,
                                  'mask_country': cname_dropdown,
                                  'textsize': textsize_w,
                                  'plotsize': plotsize_w,
                                  'outlier_threshold': outlier_thresh_w,
                                  'krig_all_flag': krig_all_w,
                                  'start_date': start_date_w,
                                  'end_date': end_date_w})

        display(grid_ui, out)


    def preprocess_all_data(self, outlier_threshold, start_date, end_date):
        '''
        Function that prepares the data for Kriging.
        '''

        # Limit CHIRP files to dates for which there is station data
        chirp_files = np.sort(glob.glob(self.chirp_floc+'*.tif'))
        file_dates = [pd.Timestamp(datesConverter.str_to_dek(dekad[-12:-4]))
                      for dekad in chirp_files]
        file_dates = np.array(file_dates)

        # chirp_files = chirp_files[(file_dates >= self.stations_1.index[0])
        #                           & (file_dates <= self.stations_1.index[-1])]
        # file_dates = file_dates[(file_dates >= self.stations_1.index[0])
        #                         & (file_dates <= self.stations_1.index[-1])]

        chirp_files = chirp_files[(file_dates >= start_date)
                                  & (file_dates <= end_date)]
        file_dates = file_dates[(file_dates >= start_date)
                                & (file_dates <= end_date)]
        self.dates_all = file_dates
        self.chirp_files_all = chirp_files

        #########################################################

        #region Step 1: Getting CHIRPS data at station locations.
        chirp_df = pd.DataFrame(index=file_dates, columns=self.latlon.index)

        for chirp_fpath, dt in tqdm(zip(chirp_files, file_dates)):
            with rio.open(chirp_fpath) as img:
                raster = img.read()[0]

                vals = []
                for l in self.latlon.index:

                    # Getting coordinates
                    coord = rio.transform.rowcol(img.transform, self.latlon.loc[l].Longitude,
                                                self.latlon.loc[l].Latitude)
                    vals.append(raster[coord[0], coord[1]])

            # print(np.shape(chirp_df.loc[dt].values), np.shape(vals))
            chirp_df.loc[dt] = vals
        chirp_df.index = chirp_df.index.rename('Date')

        intersection = self.stations_1.index.intersection(chirp_df.index)
        intersection = intersection.sort_values()

        self.stations_all = self.stations_1.loc[intersection]
        self.chirp_df_all = chirp_df.loc[intersection]
        #endregion

        #region Step 2: Removing outliers.
        threshold = outlier_threshold

        diff = np.abs(self.stations_all - self.chirp_df_all)
        self.stations_all[diff > threshold] = np.nan
        #endregion

        #region Step 3: Unbiasing
        self.chirp_df_U_all = self.chirp_df_all.copy()
        ratio = []

        for dt in self.chirp_df_all.index:
            ch = self.chirp_df_all.loc[dt]
            st = self.stations_all.loc[dt]

            if np.std(st) > 0:
                ch_ = ch[np.abs((self.stations_all.loc[dt] - np.mean(st))/np.std(st)) < 3]
                st_ = st[np.abs((self.stations_all.loc[dt] - np.mean(st))/np.std(st)) < 3]
            else:
                ch_ = ch
                st_ = st

            c = 1
            r = (np.mean(st_) + c) / (np.mean(ch_) + c)
            ratio.append(round(r, 3))

            self.chirp_df_U_all.loc[dt] = ch * r

        self.ratios_all = pd.DataFrame(data={'ratio': ratio}, index=self.chirp_df_all.index)
        #endregion

        #region Step 4: Residuals
        self.res_all = self.chirp_df_U_all - self.stations_all
        #endregion

    def do_blending_all(self, var_p1, var_p2, var_p3, var_model,
                    n_points, sigma):
        '''
        Function that performs Kriging.
        '''
        for chirp_fpath, dt in tqdm(zip(self.chirp_files_all, self.dates_all)):

            z = self.res_all.loc[dt].dropna()
            x = self.latlon.loc[z.index]['Longitude'].values
            y = self.latlon.loc[z.index]['Latitude'].values
            z = z.values

            krig = pykrige.ok.OrdinaryKriging(x, y, z,
                                            variogram_model=var_model,
                                            variogram_parameters=[var_p1, var_p2, var_p3],
                                            coordinates_type='geographic')

            # Getting raster grid
            da = xr.open_rasterio(chirp_fpath)
            xp, yp = np.meshgrid(da['x'], da['y'])
            xp, yp = xp[0], np.array([yp[i][0] for i in range(len(yp))])

            # Kriging
            if n_points is False:
                kriged_result = krig.execute(style='grid', xpoints=xp, ypoints=yp)[0].data
            else:
                try:
                    kriged_result = krig.execute(style='grid', xpoints=xp, ypoints=yp,
                                                backend='C', n_closest_points=n_points)[0].data
                except: # If less than 10 points
                    kriged_result = krig.execute(style='grid', xpoints=xp, ypoints=yp)[0].data


            # Smoothing residuals
            if sigma is not False:
                kriged_result = ndimage.gaussian_filter(kriged_result, sigma=sigma)

            # Unbiasing
            da = da * self.ratios_all.loc[dt]['ratio']

            # Producing blended raster: removing kriged residuals from CHIRP raster
            blended_CHIRP = da - kriged_result

            # Converting to int and setting all values<0 to 0
            bCH = blended_CHIRP.values
            bCH[bCH < 0] = 0
            blended_CHIRP.values = bCH.astype(int)


            # Save rasters as tiff
            tif_name = (self.blend_floc + chirp_fpath.split('\\')[1][:3]
                        + 'rfb' + datesConverter.dek_to_str(dt) + '.tif')
            blended_CHIRP.rio.to_raster(tif_name)


    def krig_all(self, outlier_threshold, start_date, end_date,
                var_p1, var_p2, var_p3, var_model, n_points, sigma):

        self.preprocess_all_data(outlier_threshold, start_date, end_date)

        self.do_blending_all(var_p1, var_p2, var_p3, var_model, n_points, sigma)
