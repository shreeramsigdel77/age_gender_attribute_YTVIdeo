import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, figure, xlim
import matplotlib.dates as md
import seaborn as sns

# plt.style.use('seaborn')
#read json file to handel line by line
file = '/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/result _aday.jsonl'
file = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/result_log/result_2021-09-28.jsonl"
def plot_df(dataframe,x_label:str,y_label:str, title:str="" ):
    """
    Needs plt.show() to display graph
    ]"""
    
    dataframe.reset_index(drop=True, inplace=True)
   
    ax = dataframe.plot(figsize =(20,10))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    old_tick_label = dataframe.Time.tolist()
    # print(old_tick_label)

    #adding space 
    ax.set_xticklabels(old_tick_label, rotation = 45)
    
    ax.legend(dataframe.columns.tolist()[1:])
    # fig.autofmt_xdate() #makes 30deg tilt on tick labels
    # plt.show()
    # exit()

def plot_bar_df(dataframe,x_label:str,y_label:str, title:str="" ,showtick:bool = False, bar_top_label:bool = False):
    """
    Needs plt.show() to display graph
    ]"""

    dataframe.reset_index(drop=True, inplace=True)
    
    ax = dataframe.plot.bar(figsize =(20,10))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # ax.tick_params list
    # ['size', 'width', 'color', 'tickdir', 'pad', 'labelsize', 'labelcolor', 'zorder', 'gridOn', 'tick1On', 'tick2On', 'label1On', 'label2On', 'length', 
    # 'direction', 'left', 'bottom', 'right', 'top', 'labelleft', 'labelbottom', 'labelright', 'labeltop', 'labelrotation', 'grid_agg_filter',
    #  'grid_alpha', 'grid_animated', 'grid_antialiased', 'grid_clip_box', 'grid_clip_on', 'grid_clip_path', 'grid_color', 'grid_contains', 
    # 'grid_dash_capstyle', 'grid_dash_joinstyle', 'grid_dashes', 'grid_data', 'grid_drawstyle', 'grid_figure', 'grid_fillstyle', 'grid_gid', 
    # 'grid_in_layout', 'grid_label', 'grid_linestyle', 'grid_linewidth', 'grid_marker', 'grid_markeredgecolor', 'grid_markeredgewidth', 
    # 'grid_markerfacecolor', 'grid_markerfacecoloralt', 'grid_markersize', 'grid_markevery', 'grid_path_effects',
    #  'grid_picker', 'grid_pickradius', 'grid_rasterized', 'grid_sketch_params', 'grid_snap', 'grid_solid_capstyle', 'grid_solid_joinstyle',
    #  'grid_transform', 'grid_url', 'grid_visible', 'grid_xdata', 'grid_ydata', 'grid_zorder', 'grid_aa', 'grid_c', 'grid_ds', 'grid_ls', 'grid_lw', 
    # 'grid_mec', 'grid_mew', 'grid_mfc', 'grid_mfcalt', 'grid_ms']

    #get size of row
    row, col = dataframe.shape
   
    # ax.set_xticks(range(-1,23,1))
    ax.set_xticks(range(-1,row-1,1))
    ax.tick_params(tick1On = showtick)
    old_stick_label = dataframe.Time.tolist()

    padding_value = 30 if row < 15 else 20

    new_pad_label = []
    for i in old_stick_label:
        # a = i.rjust(10)
        new_pad_label.append( i.rjust(padding_value))  
    
    ax.set_xticklabels(new_pad_label,rotation = 0,rotation_mode = "anchor",)
    
    if bar_top_label:
        x_offset = -0.2
        y_offset = 0.1
        for p in ax.patches:
            b = p.get_bbox()
            # val = "{:+.2f}".format(b.y1 + b.y0)  
            val = "{}".format(int(b.y1 + b.y0))       
            ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
    #    works for 3.4 and above
        # for container in ax.containers:
        #     ax.bar_label(container)
            
    ax.legend(dataframe.columns.tolist()[1:])
    # fig.autofmt_xdate() #makes 30deg tilt on tick labels
   


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
temp_dict = {}
for cols_name in df_n.columns.tolist():
    temp_dict[cols_name] = 'sum'


# df = df_n.resample('H').agg({'Female':'sum','Male':'sum','personalLess30':'sum', 'personalLess45':'sum','personalLess60':'sum','personalLarger60':'sum'})

df = df_n.resample('H').agg(temp_dict)

#new agg
df_hour = df_n


df_hour = df_hour[["Male","Female"]].copy()

df_hour["Max_Count"] = df_hour["Male"]+df_hour["Female"]

df_hour.drop("Male",axis=1,inplace=True)
df_hour.drop("Female",axis=1,inplace=True)

# df_hour_sum["Raw_sum"] = df_hour_sum

index_val_tmp = df_hour.index.values

# df_hour['Date'] = pd.to_datetime(index_val_tmp).date
df_hour['Time'] = pd.to_datetime(index_val_tmp).strftime('%H')
df_hour['Full_Time'] = pd.to_datetime(index_val_tmp).strftime('%H-%M-%S')
df_hour.reset_index(drop=True, inplace=True)

print(df_hour)
maximum_idx = df_hour.groupby(['Time'],sort=False)['Max_Count'].transform(max) == df_hour["Max_Count"]
print(df_hour[maximum_idx])

data_max = df_hour[maximum_idx]

#drop duplicate
data_max.drop_duplicates('Time',inplace=True)
data_max.drop("Time",axis=1,inplace=True)
print(data_max)

#dump json file

j_data = data_max.to_json('./max_count_info.json',orient='records')

print(j_data)


exit()
temp_df = df


print(df)
# print(temp_df)
#name index

print(temp_df.index.values)

index_val = temp_df.index.values

temp_df['Date'] = pd.to_datetime(index_val).date
temp_df['Time'] = pd.to_datetime(index_val).strftime('%H:%M')

#drop columns
# df.drop(['column_nameA', 'column_nameB'], axis=1, inplace=True)
temp_df.drop(['Date'], axis=1, inplace=True)
#reset index
temp_df.reset_index(drop=True, inplace=True)
# temp_df['Times'] = pd.to_datetime(temp_df["Time"]).dt.strftime('%H:%M')




temp_df = temp_df.set_index(temp_df["Time"])

# temp_df.drop("Time",axis=1,inplace=True)
# print(temp_df)

# exit()
df = temp_df
# plt.setp(axes.spines.values(), visible=False) 

df_gender = df[["Time","Male","Female"]].copy()
df_age = df[["Time","personalLess30",  "personalLess45",  "personalLess60",  "personalLarger60"]].copy()

sum_col = df_gender["Male"] + df_gender["Female"]

df_per_hr = df_gender[["Time"]].copy()

df_per_hr["Total"] = sum_col
# plot_df(dataframe=df_gender,x_label="Time",y_label="Count",title="Head Counts on Shibuya Scramble Crossing on the basis of Gender")
# plot_df(dataframe=df_age,x_label="Time",y_label="Count",title="Head Counts on Shibuya Scramble Crossing on the basis of Age Group")
# plt.show()


exit()
plot_bar_df(dataframe=df_gender,x_label="Time",y_label="Count",title="Scores by time and gender")
plot_bar_df(dataframe=df_age,x_label="Time",y_label="Count",title="Scores by time and age Group")
plot_bar_df(dataframe=df_per_hr,x_label="Time",y_label="Total",title="Scores by time and head count people",showtick= True, bar_top_label = True)
print("*"*20)
print("*"*20)
print("Total")
print("*"*20)
print("*"*20)
print(df.sum(axis=0))
print("*"*20)
print("*"*20)
plt.show()