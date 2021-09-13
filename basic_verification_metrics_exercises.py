'''
Calculate and plot metrics for SWIFT verification training workshop

On Jasmin use jaspy environment (module load jaspy) to ensure all required
python libraries are available.
'''


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import datetime
import cartopy.crs as ccrs
import cartopy.feature as feature
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from os.path import isfile
from scipy.stats import percentileofscore


IMERG_dir = '/gws/nopw/j04/swift/public/requests/IMERG_for_WP2/' # Change as appropriate.
temperature_dir = '/gws/nopw/j04/swift/public/requests/KMD_TminTMax_for_WP2/' # Change as appropriate
rainfall_dir = '/gws/nopw/j04/swift/public/requests/GMet_Rainfall_for_WP2/' # Change as appropriate
UM_dir_global = '/gws/nopw/j04/swift/public/requests/SWIFT_TB3_WP2/GLOBAL_00Z_0p1deg/' # Change as appropriate.
UM_dir_ta = '/gws/nopw/j04/swift/public/requests/SWIFT_TB3_WP2/TAM_18Z_0p1deg/' # Change as appropriate.
plot_dir = '/home/users/phill/images/EvaluationWorkshop/' # Change as appropriate
lon_west = 1.
lon_east = 16. 
lat_south = 1. 
lat_north = 15.

varname_dict = { 'RAIN' : 'stratiform_rainfall_amount', # matches variable name used in model filename to that used in the model data.
                 'TMIN' : 'air_temperature',
                 'TMAX' : 'air_temperature'
               }


def continuous_example():
    '''
    Calculate and plot some continuous metrics for rainfall forecasts.

    Demonstrates how to read model rainfall forecasts and IMERG data, 
    produce plots comparing one model to IMERG, calculate some continuous 
    verification metrics, and plot these verification metrics.
   
    '''
    datelist = generate_datelist(datetime.datetime(2019,9,1,0,0,0), datetime.datetime(2019,10,31,0,0,0))
    global_day1 = read_um_data_global(datelist, 'RAIN', lon_west, lon_east, lat_south, lat_north, lead_time=0) 
    global_day2 = read_um_data_global(datelist, 'RAIN', lon_west, lon_east, lat_south, lat_north, lead_time=1) 
    ta_day1 = read_um_data_ta(datelist, 'RAIN', lon_west, lon_east, lat_south, lat_north, lead_time=0) 
    ta_day2 = read_um_data_ta(datelist, 'RAIN', lon_west, lon_east, lat_south, lat_north, lead_time=1)
    imerg = read_imerg_data(datelist, lon_west, lon_east, lat_south, lat_north)
    print('global_day1.mean(axis=0).shape=', global_day1.mean(axis=0).shape)
    map_plot(global_day1.mean(axis=0), 'Global Day 1 (mm)', lon_west, lon_east, lat_south, lat_north, savename='rain_map_plot_SeptOct2019_global_day1.png') # Plot map of time mean global model day 1 forecast
    map_plot(imerg.mean(axis=0), 'IMERG (mm)', lon_west, lon_east, lat_south, lat_north, savename='rain_map_plot_SeptOct2019_imerg.png') # Plot map of time mean IMERG observations
    map_plot(global_day1.mean(axis=0)-imerg.mean(axis=0), 'Global Day 1 - IMERG (mm)', lon_west, lon_east, lat_south, lat_north, levels=[-25,-15,-10,-5,0,5,10,15,20,25], cmapname='bwr', savename='rain_map_plot_SeptOct2019_global_day1_minus_imerg.png') # Plot map of time mean global model day 1 forecast, IMERG observations and difference
    timeseries_plot([global_day1.mean(axis=(1,2)), global_day2.mean(axis=(1,2)), ta_day1.mean(axis=(1,2)), ta_day2.mean(axis=(1,2))], ['Global Day 1', 'Global Day 2', 'Trop Afr Day 1', 'Trop Afr Day 2'], datelist, lon_west, lon_east, lat_south, lat_north, 'Sept-Oct 2019', 'Daily rainfall (mm)', savename='rain_timeseries_plot_SeptOct2019.png', obs=imerg.mean(axis=(1,2)), obs_label='IMERG') # Plot timeseries of spatial mean rainfall from each model and IMERG
    global_day1_bias, global_day1_mse, global_day1_rmse, global_day1_mae, global_day1_correlation = continuous_metrics(global_day1, imerg, axis=None) # Metrics for whole domain and time period
    global_day1_bias_timeseries, global_day1_mse_timeseries, global_day1_rmse_timeseries, global_day1_mae_timeseries, global_day1_correlation_timeseries = continuous_metrics(global_day1, imerg, axis=(1,2)) # Metrics for each day
    timeseries_plot([global_day1_mse_timeseries], ['Global Day 1'], datelist, lon_west, lon_east, lat_south, lat_north, 'Sept-Oct 2019', 'Daily rainfall MSE (mm^2)', savename='rain_GlobalDay1_mse_timeseries_plot_SeptOct2019.png') # Plot timeseries of spatial mean rainfall from each model and IMERG
    global_day1_bias_map, global_day1_mse_map, global_day1_rmse_map, global_day1_mae_map, global_day1_correlation_map = continuous_metrics(global_day1, imerg, axis=0) # Metrics for each lat-lon point.
    map_plot(global_day1_correlation_map, 'Global Day and IMERG 1 correlation', lon_west, lon_east, lat_south, lat_north, savename='rain_correlation_map_plot_SeptOct2019_global_day1_imerg.png') # Plot map of time mean global model day 1 forecast, IMERG observations and difference
    persistence = read_imerg_data([d - datetime.timedelta(days=1) for d in datelist], lon_west, lon_east, lat_south, lat_north) #  persistence using previous days IMERG data.
    persistence_bias, persistence_mse, persistence_rmse, persistence_mae, persistence_correlation = continuous_metrics(persistence, imerg, axis=None) # Metrics for whole domain and time period
    global_day1_rmse_skill = continuous_generalised_skill_score(global_day1_rmse, persistence_rmse, 0) # Calculate skill score
    print('RMSE Skill score for day 1 forecast from global model (versus persistence)=', global_day1_rmse_skill)
    percentile_all = percentileofscore(imerg.flatten(), 10.0) # example of using percentile of score to calculate the percentile (of all data) for a rainfall of 10 mm
    percentile_rain_events = percentileofscore(imerg[imerg > 0], 10.0) # example of using percentile of score to calculate the percentile (of all data) for a rainfall of 10 mm
    print('For IMERG 10 mm = '+str(percentile_all)+'-th percentile of all points and '+str(percentile_rain_events)+'-th percentile of rainy points')


def binary_example():
    '''
    Calculate and plot some binary metrics for max temperature forecasts.

    Demonstrates how to read model maximum temperature forecasts and station
    data, produce plots comparing one model to the observations, calculate
    some binary verification metrics, and plot these verification metrics.
   
    '''
    threshold = 30. # Arbitrary temperature threshold used to convert to binary 
    datelist = generate_datelist(datetime.datetime(2020,1,1,0,0,0), datetime.datetime(2020,11,30,0,0,0))
    global_day1 = read_um_data_global(datelist, 'TMAX', lon_west, lon_east, lat_south, lat_north, lead_time=0) 
    global_day2 = read_um_data_global(datelist, 'TMAX', lon_west, lon_east, lat_south, lat_north, lead_time=1) 
    ta_day1 = read_um_data_ta(datelist, 'TMAX', lon_west, lon_east, lat_south, lat_north, lead_time=0) 
    ta_day2 = read_um_data_ta(datelist, 'TMAX', lon_west, lon_east, lat_south, lat_north, lead_time=1)
    station_names, station_lons, station_lats, max_data_obs, min_data_obs = read_station_temperature_data(datelist) # Read station tempeature data
    ta_day1_station = match_model_to_station(station_lons, station_lats, ta_day1, lon_west, lon_east, lat_south, lat_north)-273.15 # Get model data at station locations and convert to Celsius to match obs.
    ta_day2_station = match_model_to_station(station_lons, station_lats, ta_day2, lon_west, lon_east, lat_south, lat_north)-273.15 # Get model data at station locations and convert to Celsius to match obs.
    global_day1_station = match_model_to_station(station_lons, station_lats, global_day1, lon_west, lon_east, lat_south, lat_north)-273.15 # Get model data at station locations and convert to Celsius to match obs.
    global_day2_station = match_model_to_station(station_lons, station_lats, global_day2, lon_west, lon_east, lat_south, lat_north)-273.15 # Get model data at station locations and convert to Celsius to match obs.
    timeseries_plot([ta_day1_station[:,0], ta_day2_station[:,0], global_day1_station[:,0], global_day2_station[:,0]], ['Trop Afr Day 1', 'Trop Afr Day 2', 'Global Day 1', 'Global Day 2'], datelist, lon_west, lon_east, lat_south, lat_north, station_names[0]+' Jan-Nov 2020', 'Daily Maximum temperature (C)', savename='MaxTemp_all_models_timeseries_plot_'+station_names[0]+'_JanNov2020.png', obs=min_data_obs[:,0], obs_label='Observed') # Plot timeseries of model and obs max temp
    hits_site, misses_site, false_alarms_site, correct_negs_site = contingency_table_calc(ta_day1_station, max_data_obs, threshold, percentile_threshold=False, axis=0) # Calculate contingency table values for each site
    freq_bias_site, prop_correct_site, hit_rate_site, false_alarm_rate_site, false_alarm_ratio_site, csi_site, gss_site, hss_site, pss_site, OddsRatio_site, orss_site = binary_metrics(hits_site, misses_site, false_alarms_site, correct_negs_site)# Calculate metrics for each site
    bar_plot_by_site(np.array([hit_rate_site, false_alarm_ratio_site, false_alarm_rate_site, csi_site, prop_correct_site]), station_names, 'Tropical Africa day 1', 'Skill', savename='MaxTemp_TADay1_skill_plot_JanNov2020.png', legend=['Hit Rate', 'False Alarm Rate', 'False Alarm Ratio', 'CSI', 'PC']) # Produce bar plot showing some metrics for each site. NB same temperature threshold applied to all sites, means that colder sites never exceed this threshold, while warmer sites often do all the time. Consequently this plot has a lot of 0s and 1s.


def generate_datelist(start_date, end_date):
    '''
    Generate a list of datetimes given start and end date

    Given a start and end date, this function generates a list of all
    all the days between and including these dates, in order from oldest
    to newest.

    Args:
        start_date (object): Datetime object for first date
        end_date (object): Datetime object for last date

    Returns:
        datelist (object): List of datetime objects

    '''
    datelist = []
    n_days = 0
    today = datetime.datetime(1,1,1,0,0,0)
    while today < end_date:
        today = start_date+datetime.timedelta(days=n_days)
        datelist += [today]
        n_days += 1
    return datelist

            
def match_model_to_station(station_lons, station_lats, model_data, lon_west, lon_east, lat_south, lat_north):
    '''
    Extract data from nearest model gridpoint to station location.

    Given lists of the longitudes and latitudes of the stations, and the extent
    of the model domain, identifies the model gridbox with latitude and 
    longitude values that are closest to the station and saves these to an
    array.

    Note that this code assumes the model is on the IMERG grid (i.e. regular
    lat-lon with resolution of 0.1 degrees)

    Args:
        station_lons (ndarray): Array of station longitudes with shape (number 
            of stations)
        station_lats (ndarray): Array of station latitudes with shape (number 
            of stations)
        model_data (ndarray): Array of model data with shape (number of times, 
            number of latitudes, number of longitudes)
        lon_west (float): Longitude of western boundary of domain
        lon_east (float): Longitude of eastern boundary of domain
        lat_south (float): Latitude of southern boundary of domain
        lat_north (float): Latitude of northern boundary of domain

    Returns:
        model_data_for_station (ndarray): Masked array of model values closest
            to each station with shape (number of stations, number of times).
            Missing data is masked.

    '''
    lon = np.arange(lon_west, lon_east, 0.1)
    lat = np.arange(lat_south, lat_north, 0.1)
    model_data_for_station = []
    for s_lon, s_lat in zip(station_lons, station_lats):
        i1 = np.argmin(abs(lat-s_lat))
        i2 = np.argmin(abs(lon-s_lon))
        model_data_for_station += [model_data[:,i1,i2]]
    model_data_for_station = np.array(model_data_for_station).T
    model_data_for_station = np.ma.masked_array(model_data_for_station, mask = model_data_for_station < 0.)
    return model_data_for_station
                        
    
def read_um_data_global(forecast_day_list, varname, lon_west, lon_east, lat_south, lat_north, lead_time=0):
    '''
    Read regridded global UM data from 00Z run matching inputs.

    Global model data has 1 file for 0Z initialisation for each month and     
    variable, with two leads times (T+0-T+24 and T+24-T+48)

    Args:
        forecast_day_list (object): List of datetime objects desribing dates
            required 
        varname (str): Name of variable required, either 'RAIN' (i.e. daily 
            rainfall accumulation), TMAX' (i.e. daily maximum temperature),
            or 'TMIN' (i.e. daily minimum temperatue)
        lon_west (float): Longitude of western boundary of domain
        lon_east (float): Longitude of eastern boundary of domain
        lat_south (float): Latitude of southern boundary of domain
        lat_north (float): Latitude of northern boundary of domain
        lead_time (int, optional): Either 0 for first day of forecast 
            (T+0 to T+24) or 1 for second day of forecast (T+24 to T+48). 
            Defaults to 0.

    Returns:
        model_data(ndarray): Masked array of model data on regular lat-lon grid
            with shape (number of times, number of latitudes, number of 
            longitudes). Missing data is masked.

    '''
    data = []
    missing_days=[]
    for i,day in enumerate(forecast_day_list):
        if(((day - datetime.timedelta(days=lead_time)) == datetime.datetime(2019,9,20,0,0,0)) | ((day - datetime.timedelta(days=lead_time)) == datetime.datetime(2019,2,7,0,0,0))): # These days are missing from the global model data.
            missing_days += [i]
        else:
            if ((day.day == 1) & (lead_time == 1)): # data is stored in file for previous month
                new_day = day-datetime.timedelta(hours=24)
                filename = UM_dir_global+'/GLOBAL_{:0>2}'.format(new_day.month)+'{:0>4}'.format(new_day.year)+'_00Z_'+varname+'_0p1deg.nc'
            else:
                filename = UM_dir_global+'GLOBAL_{:0>2}'.format(day.month)+'{:0>4}'.format(day.year)+'_00Z_'+varname+'_0p1deg.nc'
            ncfile = Dataset(filename)
            lat_ind1 = np.where(ncfile.variables['latitude'][:] > lat_south)[0][0]
            lat_ind2 = np.where(ncfile.variables['latitude'][:] < lat_north)[0][-1]
            lon_ind1 = np.where(ncfile.variables['longitude'][:] > lon_west)[0][0]
            lon_ind2 = np.where(ncfile.variables['longitude'][:] < lon_east)[0][-1]
            model_day = np.array([(datetime.datetime(1970,1,1,0,0,0,0)+datetime.timedelta(hours=h)).day for h in ncfile.variables['time'][:,lead_time]])
            time_ind = np.where(model_day == day.day)[0][0]
            data += [ncfile.variables[varname_dict[varname]][lead_time, time_ind, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]]
    for ind in missing_days:
        data.insert(ind, np.zeros(data[ind-1].shape)-9999)
    data = np.array(data)
    model_data = np.ma.masked_array(data, mask=data<-9998)
    return model_data


def read_um_data_ta(forecast_day_list, varname, lon_west,lon_east, lat_south, lat_north, lead_time=0):
    '''
    Read regridded Tropical Africa UM data from 18Z run matching inputs.

    Tropical Africa model data has 1 file for 18Z initialisation for each 
    month and variable, with two leads times (T+6-T+30 and T+30-T+54)

    Args:
        forecast_day_list (object): List of datetime objects desribing dates
            required 
        varname (str): Name of variable required, either 'RAIN' (i.e. daily 
            rainfall accumulation), TMAX' (i.e. daily maximum temperature),
            or 'TMIN' (i.e. daily minimum temperatue)
        lon_west (float): Longitude of western boundary of domain
        lon_east (float): Longitude of eastern boundary of domain
        lat_south (float): Latitude of southern boundary of domain
        lat_north (float): Latitude of northern boundary of domain
        lead_time (int, optional): Either 0 for first full day of forecast 
            (T+12 to T+36) or 1 for second full day of forecast (T+36 to T+60).
            Defaults to 0.

    Returns:
        model_data(ndarray): Masked array of model data on regular lat-lon grid
            with shape (number of times, number of latitudes, number of 
            longitudes). Missing data is masked.

     '''
    data = []
    missing_days=[]
    for i,day in enumerate(forecast_day_list):
        if((day - datetime.timedelta(days=lead_time+1)) == datetime.datetime(2020,7,31,0,0,0)): # These days are missing from the ta model data.
            missing_days += [i]
        else:
            if ((day.day == 1) | ((day.day == 2) & (lead_time == 1))): # data is stored in file for previous month
                new_day = day-datetime.timedelta(days=day.day)
                filename = UM_dir_ta+'TAM_{:0>2}'.format(new_day.month)+'{:0>4}'.format(new_day.year)+'_18Z_'+varname+'_0p1deg.nc'
            else:
                filename = UM_dir_ta+'TAM_{:0>2}'.format(day.month)+'{:0>4}'.format(day.year)+'_18Z_'+varname+'_0p1deg.nc'
            ncfile = Dataset(filename)
            lat_ind1 = np.where(ncfile.variables['latitude'][:] > lat_south)[0][0]
            lat_ind2 = np.where(ncfile.variables['latitude'][:] < lat_north)[0][-1]
            lon_ind1 = np.where(ncfile.variables['longitude'][:] > lon_west)[0][0]
            lon_ind2 = np.where(ncfile.variables['longitude'][:] < lon_east)[0][-1]
            model_day = np.array([(datetime.datetime(1970,1,1,0,0,0,0)+datetime.timedelta(hours=h)).day for h in ncfile.variables['time'][:,lead_time]])
            if day.day in model_day:
                time_ind = np.where(model_day == day.day)[0][0]
                data += [ncfile.variables[varname_dict[varname]][lead_time, time_ind, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]]
            else:
                model_day0 = np.array([(datetime.datetime(1970,1,1,0,0,0,0)+datetime.timedelta(hours=h)).day for h in ncfile.variables['time_0'][:,lead_time]])
                time_ind = np.where(model_day0 == day.day)[0][0]
                data += [ncfile.variables[varname_dict[varname]+'_0'][lead_time, time_ind, lat_ind1:lat_ind2+1, lon_ind1:lon_ind2+1]]            
    for ind in missing_days:
        data.insert(ind, np.zeros(data[ind-1].shape)-9999)
    data = np.array(data)
    model_data = np.ma.masked_array(data, mask=data<-9998)
    return model_data


def read_imerg_data(forecast_day_list, lon_west, lon_east, lat_south, lat_north):
    '''
    Read accumulated daily rainfall from IMERG dataset

    Args:
        forecast_day_list (object): List of datetime objects desribing dates
            required 
        lon_west (float): Longitude of western boundary of domain
        lon_east (float): Longitude of eastern boundary of domain
        lat_south (float): Latitude of southern boundary of domain
        lat_north (float): Latitude of northern boundary of domain

    Returns:
        rain(ndarray): Array of rainfall data on regular lat-lon grid with
            shape (number of times, number of latitudes, number of longitudes)

    '''
    data = []
    for day in forecast_day_list:
        filename = IMERG_dir+'3B-DAY.MS.MRG.3IMERG.{:0>2}'.format(day.month)+'{:0>4}'.format(day.year)+'.nc4'
        ncfile = Dataset(filename)
        lat_ind1 = np.where(ncfile.variables['lat'][:] > lat_south)[0][0]
        lat_ind2 = np.where(ncfile.variables['lat'][:] < lat_north)[0][-1]
        lon_ind1 = np.where(ncfile.variables['lon'][:] > lon_west)[0][0]
        lon_ind2 = np.where(ncfile.variables['lon'][:] < lon_east)[0][-1]
        imerg_day = np.array([(datetime.datetime(1970,1,1,0,0,0,0)+datetime.timedelta(days=d)).day for d in ncfile.variables['time'][:]])
        time_ind = np.where(imerg_day == day.day)[0][0]
        data += [ncfile.variables['precipitationCal'][time_ind, lon_ind1:lon_ind2+1, lat_ind1:lat_ind2+1].T]
    rain = np.array(data)
    return rain


def read_station_temperature_data(forecast_day_list):
    '''
    Read daily maximum and minimum temperatures for Kenyan sations

    Reads temperatures from csv files exported from original excel files.
    There is one file per month, with row corresponding to different stations
    and column corresponding to days of month. Maximum temperature is first
    X rows, then minimum temperature on next X rows. Code will need to be 
    modified to handle different file layouts.

    Args:
        forecast_day_list (object): List of datetime objects desribing dates
            required 

    Returns:
        station_names (ndarrray): Array of strings containing station names with
            shape (number of stations)
        station_lons (ndarray): Array of floats containing station longitudes
            with shape (number of stations)
        station_lats (ndarray): Array of floats containing station latitudes
            with shape (number of stations)
        max_data (ndarray): Array of floats containing daily maximum temperature
            for each station on each day with shape (number of days, 
            number of stations)
        min_data (ndarray): Array of floats containing daily minimum temperature
            for each station on each day with shape (number of days, 
            number of stations)

    '''
    max_data = []
    min_data = []
    missing_days=[]
    for i,day in enumerate(forecast_day_list):
        filename = temperature_dir + day.strftime('%B')[:3] + day.strftime('%Y') + 'Temp.csv'
        if not(isfile(filename)): filename = temperature_dir + day.strftime('%B')[:4] + day.strftime('%Y') + 'Temp.csv'
        if not(isfile(filename)):
            missing_days += [i]
        else:
            with open(filename, 'r') as f:
                dailydata = f.readlines()
            ind = [i for i in range(len(dailydata)) if dailydata[i].startswith('MINIMUM')][0]
            dailydata_max = [d for d in dailydata[1:ind] if not(d.startswith('STATION'))]
            dailydata_max = [d for d in dailydata_max if len(d.replace(',', '').replace('\n', '')) > 0]
            station_lons = np.array([float(d.split(',')[1]) for d in dailydata_max])
            station_lats = np.array([float(d.split(',')[2]) for d in dailydata_max])
            station_names = np.array([d.split(',')[0] for d in dailydata_max])
            maxtemp_list = []
            for d in dailydata_max:
                station_temp = []
                for d1 in d.split(',')[3:-1]:
                    if ((d1 == '') | (d1 == '\n')):
                        station_temp += [-9999]
                    else:
                        station_temp += [float(d1)]
                maxtemp_list += [station_temp]
            maxtemp = np.array(maxtemp_list)
            dailydata_min = [d for d in dailydata[ind+1:] if not(d.startswith('STATION'))]
            mintemp_list = []
            for d in dailydata_min:
                station_temp = []
                if not(d.startswith(',,')):
                    for d1 in d.split(',')[3:-1]:
                        if ((d1 == '') | (d1 == ' ') | (d1 == '  ') | (d1 == '   ')| (d1 == '\n')| (d1 == ' \n')):
                            station_temp += [-9999]
                        else:
                            station_temp += [float(d1)]
                    mintemp_list += [station_temp]
            mintemp = np.array(mintemp_list)
            max_data += [maxtemp[:,day.day-1]]
            min_data += [mintemp[:,day.day-1]]
    for ind in missing_days:
        min_data.insert(ind, np.zeros(min_data[ind-1].shape)-9999)    
        max_data.insert(ind, np.zeros(max_data[ind-1].shape)-9999)
    max_data = np.ma.masked_array(np.array(max_data), mask =np.array(max_data) < 0.)
    min_data = np.ma.masked_array(np.array(min_data), mask =np.array(min_data) < 0.) 
    return station_names, station_lons, station_lats, max_data, min_data
   
    
def read_station_rainfall_data(forecast_day_list):
    '''
    Read daily accumulated precipitation for Ghana sations

    Reads rainfall from csv files exported from original excel files.
    There is one file per month, with row corresponding to different stations
    and column corresponding to days of month. Code may need to be modified to
    handle different spreadsheet layouts

    Args:
        forecast_day_list (object): List of datetime objects desribing dates
            required 

    Returns:
        station_names (ndarrray): Array of strings containing station names with
            shape (number of stations)
        station_lons (ndarray): Array of floats containing station longitudes
            with shape (number of stations)
        station_lats (ndarray): Array of floats containing station latitudes
            with shape (number of stations)
        rain_obs (ndarray): Array of floats containing daily accumulated
            precipitation for each station on each day with shape (number of 
            days, number of stations)

    '''
    rain_obs = []
    missing_days=[]
    for i,day in enumerate(forecast_day_list):
        filename = rainfall_dir + day.strftime('%B')[:3] + day.strftime('%Y') + 'RAIN.csv'
        if not(isfile(filename)): filename = rainfall_dir + day.strftime('%B')[:4] + day.strftime('%Y') + 'RAIN.csv'
        if not(isfile(filename)):
            missing_days += [i]
        else:
            with open(filename, 'r') as f:
                dailydata = f.readlines()
            dailydata = dailydata[2:]
            dailydata = [d for d in dailydata if len(d.replace(',', '').replace('\n', '')) > 0]
            station_lons = np.array([float(d.split(',')[1]) for d in dailydata])
            station_lats = np.array([float(d.split(',')[2]) for d in dailydata])
            station_names = np.array([d.split(',')[0] for d in dailydata])
            rain_list = []
            for d in dailydata:
                station_rain = []
                for d1 in d.split(',')[3:-1]:
                    if ((d1 == '') | (d1 == '\n')):
                        station_rain += [-9999]
                    else:
                        station_rain += [float(d1)]
                rain_list += [station_rain]
            rain = np.array(rain_list)
            rain_obs += [rain[:,day.day-1]]
    for ind in missing_days:
        rain_obs.insert(ind, np.zeros(rain_obs[ind-1].shape)-9999)
    rain_obs = np.ma.masked_array(np.array(rain_obs), mask =np.array(rain_obs) < 0.)
    return station_names, station_lons, station_lats, rain_obs
   
    
def contingency_table_calc(model, obs, threshold, percentile_threshold=False, axis=None):
    '''
    Calculates number of hits, misses, false alarms and correct negatives

    Given model and observation data and a threshold to convert each to binary
    data, calculate the terms in the contingency table. Either a physical or 
    percentile threshold can be used. An optional axis parameter enables the 
    calculation of the contingency table to keep a spatial or temporal dimesion

    Args:
        model (ndarray): Array containing model data, shape must match obs, but
            otherwise is not restricted
        obs (ndarray): Array containing observation data, shape must match 
            model, but otherwise is not restricted
        threshold (float): Threshold used to convert the model and observations
            to binary data.
        percentile_threshold (bool, optional): True if the threshold is a
            percentile rather than a physical value. Defaults to False.
        axis (tuple, optional): index of axes through which the contingency
            table terms will be summed. Defaults to None, which will result in 
            single value for each term in the contingency table.

    Returns:
        hit_sum (ndarray): Array containing number of hits. Shape depends on
           shape of model/obs and value for axis
        miss_sum (ndarray): Array containing number of misses. Shape depends on
           shape of model/obs and value for axis
        false_alarm_sum (ndarray): Array containing number of false alarms.
           Shape depends on shape of model/obs and value for axis
        correct_neg_sum (ndarray): Array containing number of correct negatives.
           Shape depends on shape sof model/obs and value for axis

    '''
    if percentile_threshold:
        # If masked array, take temporary copy where we replace missing data 
        # with NaNs, so we can use nanpercentile instead of percentile of 
        # array.compressed() which doesn't work with the axis argument.
        if np.ma.isMaskedArray(obs):
            obs_temp = np.ma.MaskedArray.copy(obs)
            obs_temp[obs_temp.mask] = np.nan
        else:
            obs_temp = np.copy(obs)
        if np.ma.isMaskedArray(model):
            model_temp = np.ma.MaskedArray.copy(model)
            model_temp[model_temp.mask] = np.nan
        else:
            model_temp = np.copy(model)
        truth_threshold = np.nanpercentile(obs_temp, threshold, axis=axis )
        threshold = np.nanpercentile(model_temp, threshold, axis=axis )
    else:
        truth_threshold = threshold
        threshold = threshold
    truth_exceed = obs > truth_threshold
    model_exceed = model > threshold
    hit = ((model_exceed) & (truth_exceed)).astype(int)
    miss = (np.invert(model_exceed) & (truth_exceed)).astype(int)
    false_alarm = ((model_exceed) & np.invert(truth_exceed)).astype(int)
    correct_neg = (np.invert(model_exceed) & np.invert(truth_exceed)).astype(int)
    hit_sum = hit.sum(axis=axis)
    miss_sum = miss.sum(axis=axis)
    false_alarm_sum = false_alarm.sum(axis=axis)
    correct_neg_sum = correct_neg.sum(axis=axis)
    return hit_sum, miss_sum, false_alarm_sum, correct_neg_sum
    


def binary_metrics(hits, misses, false_alarms, correct_negs):
    '''
    Given contingency table values, calculates binary verification metrics


    Args:
        hit_sum (ndarray): Array containing number of hits. Shape must match
            other arguments, but otherwise not restricted.
        miss_sum (ndarray): Array containing number of misses. Shape must match
            other arguments, but otherwise not restricted. 
        false_alarm_sum (ndarray): Array containing number of false alarms.
           Shape must match other arguments, but otherwise not restricted.
        correct_neg_sum (ndarray): Array containing number of correct negatives.
           Shape must match other arguments, but otherwise not restricted.

    Returns:
        freq_bias (ndarray): Frequency bias, array with same shape as arguments.
        prop_correct (ndarray): Proportion correct, array with same shape as
            arguments.
        hit_rate (ndarray): Hit rate, array with same shape as arguments.
        false_alarm_rate (ndarray): False alarm rate, array with same shape as
            arguments.
        false_alarm_ratio (ndarray): False alarm ratio, array with same shape as
            arguments.
        csi (ndarray): Critical Success Index, array with same shape as
            arguments.
        gss (ndarray): Gilbert Skill Score, array with same shape as arguments.
        hss (ndarray): Heidke Skill Score, array with same shape as arguments.
        pss (ndarray): Peirce SKill Score, array with same shape as arguments.
        OddsRatio (ndarray): Odds ratio, array with same shape as arguments.
        orss (ndarray): Odds Ratio Skill Score, array with same shape as
            arguments.

    '''
    tot_forecasts = hits + correct_negs + false_alarms + misses
    freq_bias = (hits + false_alarms) / (hits + misses)
    prop_correct = (hits + correct_negs) / (tot_forecasts)
    hit_rate = hits / (hits + misses)
    false_alarm_rate = false_alarms / (correct_negs + false_alarms)
    false_alarm_ratio = false_alarms / (hits + false_alarms)
    csi = hits / (hits + false_alarms + misses)
    hits_random = ((hits + false_alarms)*(hits + misses)) / tot_forecasts 
    correct_negs_random = ((correct_negs + false_alarms)*(correct_negs + misses)) / tot_forecasts 
    gss = (hits - hits_random) / (hits + false_alarms + misses - hits_random)
    hss = ((hits + correct_negs) - (hits_random + correct_negs_random)) / (tot_forecasts - (hits_random + correct_negs_random))
    pss = ((hits*correct_negs)-(false_alarms*misses))/((false_alarms+correct_negs)*(hits+misses))
    OddsRatio = (hits*correct_negs)/(misses*false_alarms)
    orss = (OddsRatio - 1) / (OddsRatio + 1)
    return freq_bias, prop_correct, hit_rate, false_alarm_rate, false_alarm_ratio, csi, gss, hss, pss, OddsRatio, orss


def continuous_metrics(model, observations, axis=None):
    '''
    Calculates verification metrics for continuous quantities

    Given model and observation data calculates verification metrics for the
    model. An optional axis parameter enables the calculation to be performed
    across a subset of the dimensions of the input.

    Args:
        model (ndarray): Array containing model data, shape must match obs, but
            otherwise is not restricted
        observations (ndarray): Array containing observation data, shape must
            match model, but otherwise is not restricted
        axis (tuple, optional): index of axes through which the metrics will be 
            calculate. Defaults to None, which will result in single value for
            each metric.

    Returns:
        bias (ndarray): Array containing calculated bias. Shape depends on
           shape of model/obs and value for axis
        mse (ndarray): Array containing mean sqaure errors. Shape depends on
           shape of model/obs and value for axis
        rmse (ndarray): Array containing calculated root mean square errors.
           Shape depends on shape of model/obs and value for axis
        mae (ndarray): Array containing calculated mean absolute errors.
           Shape depends on shape of model/obs and value for axis
        correlation (ndarray): Array containing calculated correlations.
           Shape depends on shape of model/obs and value for axis

    '''
    bias = np.mean(model-observations, axis=axis)
    mse = np.mean((model-observations)**2, axis=axis)
    rmse = np.sqrt(mse)
    mae = np.mean(abs(model-observations), axis=axis)
    # Python has built in correlation functions, but they don't allow you to
    # choose which axis to calculate correlation for
    covariance = np.mean(((model-model.mean(axis=axis, keepdims=True))*(observations-observations.mean(axis=axis, keepdims=True))), axis=axis)
    obs_variance = np.var(observations, axis=axis)
    model_variance = np.var(model, axis=axis)
    correlation = covariance / np.sqrt(obs_variance*model_variance)
    return bias, mse, rmse, mae, correlation


def continuous_generalised_skill_score(forecast, reference, perfect):
    '''
    Calculate generalised skill score comparing forecast to some reference

    Args:
        model (ndarray): Array containing values of any verification metric for
            model. The shape must match reference and perfect, but otherwise is 
            not restricted
        reference (ndarray): Array containing values of any verification metric 
            for reference forecast (e.g. persistence or random). The shape must 
            match model and perfect, but otherwise is not restricted.
        perfect (ndarray): Array containing values of any verification metric 
            for perfect forecast. The shape must match model and reference, but 
            otherwise is not restricted.

    Returns:
        skill (ndarray): Array containing calculated skill score. Has same 
            shape as arguments.
    
    '''
    skill = (forecast - reference) / (perfect - reference)
    return skill
    


def map_plot(data, title, lon_west, lon_east, lat_south, lat_north, levels=None, cmapname='viridis', savename=''):
    '''
    Produces plot showing map of data

    Map plot on PlateCarre projection with coastlines and borders. If savename
    is not an empty string will save the image with this filename 

    Args:
        data (ndarray): Array of values to plot with shape (number of latitudes
            , number of longitudes)
        title (str): Text for plot title
        lon_west (float): Longitude of western boundary of domain
        lon_east (float): Longitude of eastern boundary of domain
        lat_south (float): Latitude of southern boundary of domain
        lat_north (float): Latitude of northern boundary of domain
        levels(list, optional): Boundaries for the colorscale used in the plot. 
            Defaults to None, in which case the levels are chosen automatically.
        cmapname (str, optional): Name of colormap to use for plot, Defaults to 
            viridis
        savename (str, optional): Location and name of file to save image to. 
            Defaults to an empty string, which will mean the image is not saved 
            to disk.

    '''
    cmap = plt.get_cmap(cmapname)
    lon = np.arange(lon_west, lon_east, 0.1)
    lat = np.arange(lat_south, lat_north, 0.1)
    fig = plt.figure(0)
    proj=ccrs.PlateCarree(central_longitude=0.0)
    mymap = fig.add_subplot('111', projection=proj)    
    mymap.set_extent((lon_west, lon_east, lat_south, lat_north), crs=proj)
    mymap.add_feature(feature.COASTLINE, linewidth=1)
    mymap.add_feature(feature.BORDERS, linewidth=1)
    if levels == None:
        levels = MaxNLocator(nbins=10).tick_values(data.min(), data.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    else:
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    cs = mymap.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=-20)
    cb = fig.colorbar(cs, orientation='vertical')
    plt.title(title)
    fig.tight_layout()
    if savename != '':    
        plt.savefig(plot_dir+savename)
    plt.show()


def timeseries_plot(data_list, data_labels, timelist, lon_west, lon_east, lat_south, lat_north, title, ylabel, obs=[None], obs_label='Observations', savename=''):
    '''
    Produces plot showing timeseries of data

    Timeseries plot to compare data from multiple source. If savename
    is not an empty string will save the image with this filename 

    Args:
        data_list (list): List of length (number of lines to plot) of arrays of 
            shape(number of times) to plot
        data_labels (list): List of strings of length (number of lines to plot)
            corresponding to labels for the arrays in data_list
        timelist (list): List of datetime objectsof length (number of times).
        lon_west (float): Longitude of western boundary of domain
        lon_east (float): Longitude of eastern boundary of domain
        lat_south (float): Latitude of southern boundary of domain
        lat_north (float): Latitude of northern boundary of domain
        title (str): Text for plot title
        ylabel (str): Text for y axis of plot
        obs (ndarray, optional): Array of shape (number of times) containing
            timeseries of observations. Will be plotted in black with thicker
            line than other timeseries. Defaults to [None], which will result
            in no such line being plotted
        obs_label (str, optional): Text label for observations. Defaults to 
            'Observations'
        savename (str, optional): Location and name of file to save image to. 
            Defaults to an empty string, which will mean the image is not saved 
            to disk. 

    '''
    fig = plt.figure()
    if obs[0] != None:
        plt.plot(timelist, obs, label=obs_label, color='k', linewidth=3.0)
    for i, (data, name) in enumerate(zip(data_list, data_labels)):
        plt.plot(timelist, data, label=name)
    plt.legend(loc=0, handlelength=1.0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Date')
    fig.tight_layout() 
    if savename != '':
        plt.savefig(plot_dir+savename)
    plt.show()


def bar_plot_by_site(data, station_names, title, ylabel, legend=[], savename=''):
    '''
    Produces bar plot for different metrics at different observations sites

    Bar plot shows different observation sites along its x axis and different
    metrics at each site distinguished by the colour of the bar

    Args:
        data (ndarray): Array of shape (number of metrics, number of stations)
            containing values of verification metrics at each station.
        station_names (ndarrray): Array of strings containing station names with
            shape (number of stations)
        title (str): Text for plot title
        ylabel (str): Text for y axis of plot
        legend (list, optional): List of strings containing text name for each 
           metric plotted. Defaults to empty list, which will result in no
           legend.
        savename (str, optional): Location and name of file to save image to. 
            Defaults to an empty string, which will mean the image is not saved 
            to disk. 

    '''
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.bar(np.arange(len(station_names))+i*0.8/data.shape[0], data[i,:], width=0.8/data.shape[0])
    if len(legend) > 0 :
        plt.legend(legend)
#    fig.tight_layout()
    plt.title(title)
    plt.xticks(np.arange(len(station_names))+0.4, labels=station_names, rotation=90)
    fig.subplots_adjust(bottom=0.2)
    if savename != '':
        plt.savefig(plot_dir+savename)
    plt.show()
