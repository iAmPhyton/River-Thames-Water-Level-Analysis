import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

river_thames = pd.read_csv('10-11_London_Bridge.txt') 
river_thames
# confirming that the last column is indeed empty 
river_thames.describe()
# creating a new DataFrame which takes only the first three columns 
new_df = river_thames[river_thames.columns[0:3]]

# renaming as datatime, water_level, is_high_tide
new_df.columns = ['datetime', 'water_level', 'is_high_tide']
new_df
# Converting to datetime
new_df['datetime'] = pd.to_datetime(new_df['datetime'], errors='coerce')  # Add errors='coerce' to handle invalid date values by converting them to NaT

# Converting to float
new_df['water_level'] = pd.to_numeric(new_df['water_level'], errors='coerce')  # Use pd.to_numeric to handle non-numeric values by converting them to NaN

# Create extra columns: month, year
new_df['month'] = new_df['datetime'].dt.month
new_df['year'] = new_df['datetime'].dt.year

new_df 

def clean_data(data):
    # using first 3 columns
    data = data[data.columns[0:3]]
    
    # renaming columns
    data.columns = ['datetime', 'water_level', 'is_high_tide']
    
    # converting 'datetime' to 'datetime' format
    data['datetime'] = pd.to_datetime(data['datetime']) 
    
    # convert 'water_level' to float format
    data['water_level'] = data['water_level'].astype(float)
    
    # create extra columns: month, year for easy access
    data['month'] = data['datetime'].dt.month
    data['year'] = data['datetime'].dt.year
    
    return data
clean_data(new_df) 
# Creating a histogram of new_df where is_high_tide = 0
plt.hist(new_df.query('is_high_tide==0')['water_level'], bins=100, alpha=0.7, label='Low Tide', color='blue')

# Creating a histogram of new_df where is_high_tide = 1
plt.hist(new_df.query('is_high_tide==1')['water_level'], bins=100, alpha=0.7, label='High Tide', color='orange')

# Adding labels and title
plt.xlabel('Water Level')
plt.ylabel('Frequency')
plt.title('Distribution of Water Level During High and Low Tides')

plt.legend()
plt.show()

# Using boxplots to give a sense of min, max, range, and outliers of the data
plt.figure(figsize=(10, 6))

# Adjusting box width, color, and style
sns.boxplot(data=new_df.query('is_high_tide==0'), x='water_level', color='lightblue', width=0.5, boxprops=dict(alpha=0.7))

# Adding labels and title
plt.xlabel('Water Level during Low Tide')
plt.ylabel('Water Level')
plt.title('Boxplot of Water Level during Low Tide')

plt.show()

# Using boxplots to give a sense of min, max, range, and outliers of the data
plt.figure(figsize=(10, 6))

# Adjusting box width, color, and style
sns.boxplot(data=new_df.query('is_high_tide==1'), x='water_level', color='Tomato', width=0.5, boxprops=dict(alpha=0.7))

# Adding labels and title
plt.xlabel('Water Level during Low Tide')
plt.ylabel('Water Level')
plt.title('Boxplot of Water Level during Low Tide')

plt.show()

# futher analysis
new_df.query('(is_high_tide==1)').describe()

# plotting the ratio of high tide days for each year
plt.figure(figsize=(10, 6))

# Calculating the ratio
all_days = new_df.query('is_high_tide==1').groupby('year').count()['water_level']
high_days = new_df.query('(water_level>3.7) & (is_high_tide==1)').groupby('year').count()['water_level']
ratio = (high_days / all_days).reset_index()

plt.plot(ratio.year, ratio.water_level, marker='o', color='skyblue', label='High Tide Ratio')

plt.xlabel('Year')
plt.ylabel('High Tide Ratio')
plt.title('Ratio of High Tide Days Each Year')

plt.legend()
plt.show()

# summarizing water_level where is_high_tide==0
# counting number of days of low tide per year in new_df
# counting number of days of low tide where water level was below the 25th percentile in new_df

new_df.query('(is_high_tide==0)').describe()
plt.figure(figsize=(10, 6))

# Calculate the ratio
all_days = new_df.query('is_high_tide==0').groupby('year').count()['water_level']
high_days = new_df.query('(water_level<-2.66) & (is_high_tide==0)').groupby('year').count()['water_level']
ratio = (high_days / all_days).reset_index()

# plotting the ratio of low tide days for each year
plt.plot(ratio.year, ratio.water_level, marker='o', color='red', label='Low Tide Ratio')

plt.xlabel('Year')
plt.ylabel('Low Tide Ratio')
plt.title('Ratio of Low Tide Days Each Year')

plt.legend()
plt.show()

# assessing monthly trends in water levels for 1927, 1928 and 1929
# looping through 1927, 1928, 1929

water_level = pd.DataFrame()

for year in [1927, 1928, 1929]:
    level_per_year = new_df.query(f'year=={year}').set_index('datetime')
    level_per_year = level_per_year.groupby('is_high_tide').resample('1M').median()['water_level'].reset_index()
    level_per_year['month'] = level_per_year.datetime.dt.month
    level_per_year['year'] = level_per_year.datetime.dt.year
    water_level = pd.concat([water_level, level_per_year]).reset_index(drop=True)

water_level[water_level['is_high_tide']==0].describe()

# plotting the high tide data for selected months indicating median of high tide in the months

sns.lineplot(data=water_level[water_level['is_high_tide']==0], y='water_level', x='month', hue='year')
plt.axhline(-2.46,0,12,linestyle='--',color='red')
plt.show()
water_level[water_level['is_high_tide']==1].describe()

# plotting the low tide data for selected months indicating median of low tide in the months

sns.lineplot(data=water_level[water_level['is_high_tide']==1],y='water_level',x='month',hue='year')
plt.axhline(3.24,0,12,linestyle='--',color='red')
plt.show()

#beginning a forecasting model for London Bridge: a taste of autocorrelation

new_df1 = new_df.query('(year==1928) & (month<=6)').reset_index()

# plotting the high tide data for new_df1

fig, ax = plt.subplots(figsize=(20,4))
sns.lineplot(data=new_df1.query('is_high_tide==1'),x='datetime',y='water_level',ax=ax)
plt.show()

# Defining the autocorr function
def autocorr(data, level='1D', flag=0):
    high_low = {0: 'high', 1: 'low'}
    level_dict = {'1D': 'daily', '15D': 'biweekly', '1M': 'monthly', '1Y': 'annual'}

    # Assuming 'is_high_tide', 'datetime', and 'water_level' columns exist in your dataset
    data = data[data['is_high_tide'] == flag].set_index('datetime').resample(level).mean()['water_level']

    # Checking if there is sufficient data for autocorrelation calculation
    if len(data) < 2:
        print("Insufficient data for autocorrelation calculation.")
        return

    diff = data.diff()
    autocorrelation = diff.autocorr()

    print(f"The autocorrelation of {level_dict[level]} {high_low[flag]} values is {autocorrelation}")

# List of parameter combinations
param_combinations = [('1Y', 1), ('1Y', 0), ('1M', 1), ('1M', 0), ('15D', 1), ('15D', 0), ('1D', 1), ('1D', 0)]

# Calls to the autocorr function using a loop
for level, flag in param_combinations:
    autocorr(new_df, level=level, flag=flag) 
