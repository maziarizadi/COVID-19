#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np


# In[2]:


#!/usr/bin/env python
# from __future__ import print_function
import os
 
# path = '.'
 
# files = os.listdir(path)
# for name in files:
#     print(name)


# # In[3]:


# path = 'csse_covid_19_data/.'
 
# files = os.listdir(path)
# for name in files:
#     print(name)


# # In[4]:


# path = 'csse_covid_19_data/csse_covid_19_time_series/'
 
# files = os.listdir(path)
# for name in files:
#     print(name)


# In[5]:


data_conf_path = "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
data_reco_path = "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
data_death_path = "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
data_conf = pd.read_csv(data_conf_path)
data_reco = pd.read_csv(data_reco_path)
data_death = pd.read_csv(data_death_path)


# In[6]:
# Add a title
st.title('Visualised Epidemic of Covid-19 throughout the globe')
# Add some text
st.text('Based on data provided by https://github.com/CSSEGISandData')

# data_conf.shape


# In[7]:


# data_reco.shape


# In[8]:


# data_death.shape


# In[9]:


# data_death.head()


# In[10]:


# data_death.columns.format


# In[ ]:





# In[11]:


data_conf_country = data_conf.iloc[:, 1]
data_conf_date = data_conf.iloc[:, 4:]


# In[12]:


data_reco_country = data_reco.iloc[:, 1]
data_reco_date = data_reco.iloc[:, 4:]


# In[13]:


data_death_country = data_death.iloc[:, 1]
data_death_date = data_death.iloc[:, 4:]


# In[14]:


new_data_conf = pd.merge(data_conf_country,data_conf_date,left_index=True, right_index=True)
new_data_reco = pd.merge(data_reco_country,data_reco_date,left_index=True, right_index=True)
new_data_death = pd.merge(data_death_country,data_death_date,left_index=True, right_index=True)


# In[15]:


# new_data_conf.shape


# In[16]:


# new_data_conf


# In[17]:


country_list = sorted(list(set(data_conf_country)))


# In[18]:


# len(country_list)


# In[19]:


new_data_conf = new_data_conf.groupby(['Country/Region'])[new_data_conf.columns].sum()
new_data_reco = new_data_reco.groupby(['Country/Region'])[new_data_reco.columns].sum()
new_data_death = new_data_death.groupby(['Country/Region'])[new_data_death.columns].sum()


# In[20]:


new_data_conf=new_data_conf.transpose()
new_data_reco=new_data_reco.transpose()
new_data_death=new_data_death.transpose()


# In[21]:


# new_data_conf.shape


# In[22]:


# new_data_conf.head()


# In[23]:


new_data_conf.reset_index(level=0, inplace=True)


# In[24]:


# new_data_conf.head()


# In[25]:


new_data_reco.reset_index(level=0, inplace=True)


# In[26]:


# new_data_reco.head()


# In[27]:


new_data_death.reset_index(level=0, inplace=True)


# In[28]:


# new_data_death.head()


# In[29]:


new_data_conf.rename(columns={'index' : 'Date'}, inplace=True)
new_data_reco.rename(columns={'index' : 'Date'}, inplace=True)
new_data_death.rename(columns={'index' : 'Date'}, inplace=True)


# In[30]:


# new_data_conf.dtypes


# In[31]:


# new_data_conf['Date'] = new_data_conf['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))


# In[32]:


# new_data_reco['Date'] = new_data_reco['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))


# In[33]:


# new_data_death['Date'] = new_data_death['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))


# In[34]:


# new_data_conf.dtypes


# In[35]:


# new_data_reco.dtypes


# In[36]:


# new_data_death.dtypes


# In[37]:


# new_data_conf.head()


# In[38]:


cols = new_data_conf.columns.tolist()

new_data_conf = new_data_conf.melt(id_vars='Date', value_vars=cols[2:])
new_data_reco = new_data_reco.melt(id_vars='Date', value_vars=cols[2:])
new_data_death = new_data_death.melt(id_vars='Date', value_vars=cols[2:])


# In[39]:


# new_data_conf


# In[40]:


# new_data_reco


# In[41]:


# new_data_death


# In[42]:


newcols = new_data_conf.columns.to_list()
# newcols


# In[43]:


new_df = pd.merge(new_data_conf, new_data_reco,  how='left', left_on=['Date','Country/Region'], right_on = ['Date','Country/Region'])
# new_df.head()


# In[44]:


Group = new_df.groupby(['Date','Country/Region'])
Group.mean()


# In[45]:


final_df = pd.merge(new_df, new_data_death,  how='left', left_on=['Date','Country/Region'], right_on = ['Date','Country/Region'])
# final_df.head()


# In[46]:


Group2 = final_df.groupby(['Date','Country/Region'])
# Group2.mean()


# In[47]:


lastcols = final_df.columns.to_list()
# lastcols


# In[48]:


final_df.rename(columns={'Date' : 'Date',
                     'Country/Region' : 'Country/Region',
                     'value_x' : 'Confirmed_Cases',
                     'value_y' : 'Recovered_Cases',
                     'value' : 'Deaths'}, inplace=True)


# In[ ]:





# In[49]:


# df = new_data_conf
# # df['Countries']=np.nan
# df['Countries']=np.nan
# df.head()


# In[50]:


# cols = cols[0:1]+cols[-1:]+cols[1:-1]


# In[51]:


# df = df[cols]
# df.head()


# In[52]:


# country_dict = dict()
# df2 = new_data_conf

# df2 = df2.melt(id_vars='Date', value_vars=cols[2:])
# df2.tail()


# In[53]:


# df_dict = {'time':[1,2,3],'a':['a1','a2','a3'],'b':['b1','b2','b3'], 'c':['c1','c2','c3']}

# df = pd.DataFrame(df_dict)

# df


# In[54]:


# df.melt(id_vars='time', value_vars=['a', 'b', 'c'])


# In[55]:


# df_dict = {'time':[1,2,3,1,2,3,1,2,3],'new_column':['a','a','a','b','b','b','c','c','c'], 'values':['a1','a2','a3','b1','b2','b3','c1','c2','c3']}

# df = pd.DataFrame(df_dict)

# df


# In[56]:


# new_data_conf=new_data_conf.transpose()
# new_header = new_data_conf.iloc[0] 

# new_data_conf = new_data_conf[1:] 
# new_data_conf.columns = new_header
# new_data_conf.head()


# In[57]:


# new_data_reco = new_data_reco.transpose()
# new_header = new_data_reco.iloc[0] 

# new_data_reco = new_data_reco[1:]
# new_data_reco.columns = new_header
# new_data_reco.head()


# In[58]:


# new_data_death = new_data_death.transpose()
# new_header = new_data_death.iloc[0] 

# new_data_death = new_data_death[1:]
# new_data_death.columns = new_header
# new_data_death.head()


# In[59]:


# pip install motionchart


# In[60]:


# pip install pyperclip


# In[61]:


from motionchart.motionchart import MotionChart
# import matplotlib.pyplot as plt 
# import matplotlib.image as mpimg
# from scipy.stats import linregress
# import plotly
# import plotly.plotly as py
# import plotly.tools as tls


# In[62]:


# get_ipython().run_cell_magic('html', '', '<style>\n.output_wrapper, .output {\n    height:auto !important;\n    max-height:10000px;  /* your desired max-height here */\n}\n.output_scroll {\n    box-shadow:none !important;\n    webkit-box-shadow:none !important;\n}\n</style>')


# In[63]:


lastcols = final_df.columns.to_list()
# lastcols


# In[107]:


mChart = MotionChart(df = final_df, key='Date', x='Confirmed_Cases', y='Recovered_Cases', xscale='linear', yscale='linear',
                     size='Deaths', color='Country/Region', category='Country/Region')

mChart.to_browser()


# In[64]:


# sorted_df = final_df.sort_values(by=['Deaths'], ascending=False)


# In[65]:


# top_countries = list(sorted_df['Country/Region'])


# In[66]:


top_countries = final_df.loc[final_df['Confirmed_Cases'] > 10000]


# In[67]:


top_countries_set = set(top_countries['Country/Region'])
# top_countries_set


# In[68]:


top_countries_df = final_df.loc[final_df['Country/Region'].isin(top_countries_set)]
# top_countries_df.head()


# In[132]:


mChart = MotionChart(df = top_countries_df, key='Date', x='Confirmed_Cases', y='Recovered_Cases', xscale='linear', yscale='linear',
                     size='Deaths', color='Country/Region', category='Country/Region')

mChart.to_browser()


# In[2]:


# mChart = MotionChart(df = final_df, key='Date', x='Recovered_Cases', y='Deaths', xscale='linear', yscale='linear',
#                      size='Confirmed_Cases', color='Country/Region', category='Country/Region')

# mChart.to_notebook()


# In[1]:


# mChart = MotionChart(df = final_df, key='Date', x='Recovered_Cases', y='Deaths', xscale='linear', yscale='linear',
#                      size='Confirmed_Cases', color='Country/Region', category='Country/Region')

# mChart.to_notebook()


# In[ ]:




