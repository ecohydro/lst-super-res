import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string

def make_df(model, terrain, file_path, df):
    r2 = []
    rmse = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['landcover'] == terrain:
                if(model == 'Coarse'):
                    r2.append(float(row['r2_coarse']))
                    rmse.append(math.sqrt(float(row['mse_coarse']))) # rmse
                elif(model == 'Deep Learning'):
                    r2.append(float(row['r2_pred']))
                    rmse.append(math.sqrt(float(row['mse_pred']))) # rmse
                elif(model == 'Random Forest'):
                    r2.append(float(row['r2_pred']))
                    rmse.append(float(row['rmse_pred']))

    mean_r2 = np.mean(np.array(r2))
    mean_rmse = np.mean(np.array(rmse))

    error_r2 = 1.96 * (np.std(np.array(r2)) / math.sqrt(len(r2) - 1))
    error_rmse = 1.96 * (np.std(np.array(rmse)) / math.sqrt(len(r2) - 1))

    # get rid of punctuation for the RF terrain data 
    terrain = terrain.translate(str.maketrans('', '', string.punctuation))

    df.loc[len(df.index)] = [model, terrain, mean_r2, error_r2, mean_rmse, error_rmse]

def make_df_all(model, file_path, df):
    r2 = []
    rmse = []

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if(model == 'Coarse'):
                r2.append(float(row['r2_coarse']))
                rmse.append(math.sqrt(float(row['mse_coarse']))) # rmse
            elif(model == 'Deep Learning'):
                r2.append(float(row['r2_pred']))
                rmse.append(math.sqrt(float(row['mse_pred']))) # rmse
            elif(model == 'Random Forest'):
                r2.append(float(row['r2_pred']))
                rmse.append(float(row['rmse_pred']))

    mean_r2 = np.mean(np.array(r2))
    mean_rmse = np.mean(np.array(rmse))

    error_r2 = 1.96 * (np.std(np.array(r2)) / math.sqrt(len(r2) - 1))
    error_rmse = 1.96 * (np.std(np.array(rmse)) / math.sqrt(len(r2) - 1))

    df.loc[len(df.index)] = [model, "All", mean_r2, error_r2, mean_rmse, error_rmse]


def plot(df, y_pos, err):
    ax = sns.pointplot(x='Model', y=y_pos, hue='Terrain', data=df, dodge=True, join=False, errorbar=None, scale=0.65)
    x_coords = []
    y_coords = []

    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    
    errors = df[err]
    ax.errorbar(x_coords, y_coords, yerr=errors, color='black', fmt=' ', capsize=3, zorder=-1, elinewidth=1)
    
    return ax

# make df
columns = ["Model", "Terrain", "R-squared", "R-squared Error", "RMSE", "RMSE Error"]
data = pd.DataFrame(columns=columns) 

fp = 'experiments/exp3/prediction_metrics/val/prediction_metrics.csv'
rf_fp = 'RF/results_4.csv'

make_df('Coarse', 'Natural', fp, data)
make_df('Coarse', 'Agricultural', fp, data)
make_df('Coarse', 'Urban', fp, data)

make_df('Deep Learning', 'Natural', fp, data)
make_df('Deep Learning', 'Agricultural', fp, data)
make_df('Deep Learning', 'Urban', fp, data)

make_df('Random Forest', '[\'Natural\']', rf_fp, data)
make_df('Random Forest', '[\'Agricultural\']', rf_fp, data)
make_df('Random Forest', '[\'Urban\']', rf_fp, data)

make_df_all('Coarse', fp, data)
make_df_all('Deep Learning', fp, data)
make_df_all('Random Forest', rf_fp, data)

# plot(data, 'RMSE', 'RMSE Error')
plot(data, 'R-squared', 'R-squared Error')
plt.show()
plt.savefig('plot-tests.png', dpi=500)

print(data)
        