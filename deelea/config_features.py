_temporal_features = [
    "MonthSin",
    "MonthCos",
    "DaySin",
    "DayCos",
    "HourSin",
    "HourCos",
]

_cont_names_syn_wind = [
    "T_HAG_2_M",  # temperature
    "RELHUM_HAG_2_M",  # relative humidity
    "PS_SFC_0_M",  # surface pressure
    "ASWDIFDS_SFC_0_M",  # surface down solar diffuse radiation
    "ASWDIRS_SFC_0_M",  # urface down solar direct radiation
    "WindSpeed58m",
    "SinWindDirection58m",
    "CosWindDirection58m",
    "WindSpeed60m",
    "SinWindDirection60m",
    "CosWindDirection60m",
    "WindSpeed58mMinus_t_1",
    "SinWindDirection58mMinus_t_1",
    "CosWindDirection58mMinus_t_1",
    "WindSpeed60mMinus_t_1",
    "SinWindDirection60mMinus_t_1",
    "CosWindDirection60mMinus_t_1",
    "WindSpeed58mPlus_t_1",
    "SinWindDirection58mPlus_t_1",
    "CosWindDirection58mPlus_t_1",
    "WindSpeed60mPlus_t_1",
    "SinWindDirection60mPlus_t_1",
    "CosWindDirection60mPlus_t_1",
] + _temporal_features

_cont_names_real_wind = [
    "ICON_EU_SURFACE_DOWN_DIRECT_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0",
    "ICON_EU_SURFACE_DOWN_DIFFUSE_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0",
    "ICON_EU_TEMPERATURE_HEIGHT_ABOVE_GROUND_AT2",
    "ICON_EU_PRESSURE_SURFACE_AT0",
    "ICON_EU_RELATIVE_HUMIDITY_HEIGHT_ABOVE_GROUND_AT2",
    "ICON_EU_SNOW_DEPTH_SURFACE_AT0",
    "ICON_EU_TOTAL_CLOUD_COVER_SURFACE_AT0",
    "ICON_EU_WW_WEATHER_INTERPRETATION_SURFACE_AT0",
    "ICON_EU_TOTAL_PRECIPITATION_SURFACE_AT0",
    "WindSpeed58m",
    "SinWindDirection58m",
    "CosWindDirection58m",
    "WindSpeed10m",
    "SinWindDirection10m",
    "CosWindDirection10m",
    "WindSpeed58m_t_m_1",
    "WindSpeed58m_t_p_1",
    "SinWindDirection58m_t_m_1",
    "SinWindDirection58m_t_p_1",
    "CosWindDirection58m_t_m_1",
    "CosWindDirection58m_t_p_1",
    "WindSpeed10m_t_m_1",
    "WindSpeed10m_t_p_1",
    "SinWindDirection10m_t_m_1",
    "SinWindDirection10m_t_p_1",
    "CosWindDirection10m_t_m_1",
    "CosWindDirection10m_t_p_1",
] + _temporal_features


_cont_names_real_pv = [
    "ICON_EU_TEMPERATURE_HEIGHT_ABOVE_GROUND_AT2",
    "ICON_EU_PRESSURE_SURFACE_AT0",
    "ICON_EU_RELATIVE_HUMIDITY_HEIGHT_ABOVE_GROUND_AT2",
    "ICON_EU_SNOW_DEPTH_SURFACE_AT0",
    "ICON_EU_TOTAL_CLOUD_COVER_SURFACE_AT0",
    "ICON_EU_WW_WEATHER_INTERPRETATION_SURFACE_AT0",
    "ICON_EU_TOTAL_PRECIPITATION_SURFACE_AT0",
    "WindSpeed58m",
    "SinWindDirection58m",
    "CosWindDirection58m",
    "WindSpeed10m",
    "SinWindDirection10m",
    "CosWindDirection10m",
    "ICON_EU_SURFACE_DOWN_DIRECT_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0_INSTANT",
    "ICON_EU_SURFACE_DOWN_DIFFUSE_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0_INSTANT",
    "ICON_EU_SURFACE_DOWN_DIRECT_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0_INSTANT_t_m_1",
    "ICON_EU_SURFACE_DOWN_DIRECT_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0_INSTANT_t_p_1",
    "ICON_EU_SURFACE_DOWN_DIFFUSE_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0_INSTANT_t_m_1",
    "ICON_EU_SURFACE_DOWN_DIFFUSE_RADIATION_AVG_SINCE_MODELSTART_SURFACE_AT0_INSTANT_t_p_1",
] + _temporal_features

_cont_names_open_wind = [
    "AirPressure",
    "Temperature",
    "Humidity",
    "WindSpeed100m",
    "WindSpeed10m",
    "WindDirectionZonal100m",
    "WindDirectionMeridional100m",
] + _temporal_features

_cont_names_open_pv = [
    "SunPositionThetaZ",
    "SunPositionSolarAzimuth",
    "SunPositionExtraTerr",
    "SunPositionSolarHeight",
    "ClearSkyDiffuse",
    "ClearSkyDirect",
    "ClearSkyGlobal",
    "ClearSkyDiffuseAgg",
    "ClearSkyDirectAgg",
    "ClearSkyGlobalAgg",
    "Albedo",
    "WindDirectionZonal0m",
    "WindDirectionMeridional0m",
    "WindDirectionZonal100m",
    "WindDirectionMeridional100m",
    "DewpointTemperature",
    "Temperature",
    "PotentialVorticityAt1000",
    "PotentialVorticityAt950",
    "RelativeHumidity",
    "RelativeHumidityAt950",
    "RelativeHumidityAt0",
    "SnowDensity",
    "SnowDepth",
    "SnowfallPlusStratiformSurface",
    "SurfacePressure",
    "NetSolarRadiation",
    "SolarRadiationDirect",
    "SolarRadiationDiffuse",
    "CloudCover",
    "LowerWindSpeed",
    "LowerWindDirection",
    "LowerWindDirectionMath",
    "LowerWindDirectionCos",
    "LowerWindDirectionSin",
    "UpperWindSpeed",
    "UpperWindDirection",
    "UpperWindDirectionMath",
    "UpperWindDirectionCos",
    "UpperWindDirectionSin",
    "WindSpeed100m",
] + _temporal_features

_cont_names_syn_pv = [
    "T_HAG_2_M",
    "RELHUM_HAG_2_M",
    "PS_SFC_0_M",
    "ASWDIFDS_SFC_0_M_INSTANT",
    "ASWDIRS_SFC_0_M_INSTANT",
    "ASWDIFDS_SFC_0_M_INSTANT_m1",
    "ASWDIRS_SFC_0_M_INSTANT_m1",
    "ASWDIFDS_SFC_0_M_INSTANT_p1",
    "ASWDIRS_SFC_0_M_INSTANT_p1",
    "solar_azimuth",
    "solar_zenith",
    "WindSpeed60m",
    "SinWindDirection60m",
    "CosWindDirection60m",
] + _temporal_features

_cat_names_syn_wind = [
    "turbine",
    "hub_height_m",
    "rotor_diameter_m",
    "nominal_power_kW",
]

_cat_names_real_wind = [
    "nomCapWatt",
    "anonymizedManufactor",
    "anonymizedHubheight",
    "anonymizedDiameter",
    "generatorsInPark",
]

_cat_names_real_pv = [
    "nomCapWatt",
    "generatorsInPark",
    "tilt",
    "azimuth",
]

_cat_names_open_pv = ["TaskID"]
_cat_names_open_wind = ["TaskID"]

_cat_names_syn_pv = ["tilt_angle", "azimuth_angle"]

_timestamp_name = "TimeUTC"
