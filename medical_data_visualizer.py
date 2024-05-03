import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Instruction 1: Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv('medical_examination.csv')

# Instruction 2: Create the overweight column in the df variable
df['overweight'] = np.where(df['weight']/((df['height']/100)**2) > 25, 1, 0)

# Instruction 3: Normalize data by making 0 always good and 1 always bad. If the value of cholesterol or gluc is 1, set the value to 0. If the value is more than 1, set the value to 1.
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)

def draw_cat_plot(df):
    # Instruction 4: Draw the Categorical Plot in the draw_cat_plot function
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Instruction 5: Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_grouped = df_cat.groupby(['cardio', 'variable', 'value']).size().unstack(fill_value=0)
    df_grouped.columns = ['cardio_0', 'cardio_1']

    # Instruction 6: Convert the data into long format and create a chart that shows the value counts of the categorical features using the following method provided by the seaborn library import : sns.catplot()
    fig = sns.catplot(data=df_grouped, kind='bar', height=4, aspect=1.5)
    return fig

def draw_heat_map(df):
    # Instruction 7: Clean the data in the df_heat variable by filtering out the following patient segments that represent incorrect data:
    df_heat = df[df['ap_lo'] <= df['ap_hi']]
    df_heat = df_heat[df_heat['height'] >= df_heat['height'].quantile(0.025)]
    df_heat = df_heat[df_heat['height'] <= df_heat['height'].quantile(0.975)]
    df_heat = df_heat[df_heat['weight'] >= df_heat['weight'].quantile(0.025)]
    df_heat = df_heat[df_heat['weight'] <= df_heat['weight'].quantile(0.975)]

    # Instruction 8: Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr()

    # Instruction 9: Generate a mask for the upper triangle and store it in the mask variable
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Instruction 10: Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Instruction 11: Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap()
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', square=True, linewidths=.5, ax=ax)
    return f
