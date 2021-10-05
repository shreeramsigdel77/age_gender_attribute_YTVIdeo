import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, figure
import matplotlib.dates as md
import seaborn as sns

# plt.style.use('seaborn')
#read json file to handel line by line
file = 'result.jsonl'
def plot_df(dataframe,x_label:str,y_label:str, title:str="" ):
    """
    Needs plt.show() to display graph
    ]"""

    # ax = dataframe.plot(figsize=(20,10))
    
    fig,ax = plt.subplots(figsize=(15, 10))
    ax
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    hourlocator = md.HourLocator(interval = 1)

    # Set the format of the major x-ticks:
    majorFmt = md.DateFormatter('%H:%M')  

    ax.xaxis.set_major_locator(hourlocator)
    ax.xaxis.set_major_formatter(majorFmt)
    ax.plot(dataframe.index,dataframe.values)
    ax.legend(dataframe.columns.tolist())
    fig.autofmt_xdate() #makes 30deg tilt on tick labels


def plot_bar_df(dataframe,x_label:str,y_label:str, title:str="" ):
    """
    Needs plt.show() to display graph
    ]"""

    # ax = dataframe.plot(figsize=(20,10))
    labels = dataframe.index
    x = np.arange(len(labels))
    fig,ax = plt.subplots(figsize=(15, 10))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    hourlocator = md.HourLocator(interval = 1)

    # Set the format of the major x-ticks:
    # majorFmt = md.DateFormatter('%H:%M')  

    # ax.xaxis.set_major_locator(hourlocator)
    # ax.xaxis.set_major_formatter(majorFmt)

    no_of_items = len(dataframe.columns.tolist())
    width = 0.35 #width of the bars
    width_value = width/no_of_items
    for i in dataframe.columns.tolist():
       ax.bar(x-width_value,dataframe[i],width)
       width_value+=width/no_of_items
    ax.legend(dataframe.columns.tolist())
    fig.autofmt_xdate() #makes 30deg tilt on tick labels
    


def read_list(filename):
    dict_lines = []
    with open(filename, 'r') as fp:
        for line in fp:
            # print(line)
            dict_lines.append(json.loads(line))
    return dict_lines

# lines = read_list(file)




lines = read_list(file)

#mapping dict values


df = pd.DataFrame(lines)

# print(df)
#convert dataand time string to date and time



df_datetime_series = pd.to_datetime(df["Time"])

#create datetime index passing the date time series

df_datetime_index = pd.DatetimeIndex(df_datetime_series.values)

df_n = df.set_index(df_datetime_index)

#lets remove the old time column
df_n.drop("Time",axis=1,inplace=True)




#group data by the index hour value

# group_df = df.groupby(pd.Grouper(key="Time",freq='H')).sum()

# print(group_df)

#now resampling for each hour
df = df_n.resample('H').agg({'Female':'sum','Male':'sum','personalLess30':'sum', 'personalLess45':'sum','personalLess60':'sum','personalLarger60':'sum'})


# plt.setp(axes.spines.values(), visible=False) 

df_gender = df[["Male","Female"]].copy()
df_age = df[["personalLess30",  "personalLess45",  "personalLess60",  "personalLarger60"]].copy()


plot_df(dataframe=df_gender,x_label="Time",y_label="Count",title="Head Counts on Shibuya Scramble Crossing on the basis of Gender")
plot_df(dataframe=df_age,x_label="Time",y_label="Count",title="Head Counts on Shibuya Scramble Crossing on the basis of Age Group")


plot_bar_df(dataframe=df_gender,x_label="Time",y_label="Count",title="Head Counts on Shibuya Scramble Crossing on the basis of Gender")
plot_bar_df(dataframe=df_age,x_label="Time",y_label="Count",title="Head Counts on Shibuya Scramble Crossing on the basis of Age Group")

print("*"*20)
print("*"*20)
print("Total")
print("*"*20)
print("*"*20)
print(df.sum(axis=0))
print("*"*20)
print("*"*20)
plt.show()