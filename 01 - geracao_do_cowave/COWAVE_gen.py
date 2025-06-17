
"""
@author: melpakkampradeep
"""

import pandas as pd
import numpy as np
import math
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox
import statsmodels.nonparametric.smoothers_lowess as sm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('max_colwidth', None)
pd.set_option("expand_frame_repr", False)


def labeller_1(WHO_Data):
    dataset = pd.DataFrame()
    datafull = WHO_Data.drop(columns=['Country', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths', 'WHO_region'])
    countrylist = datafull['Country_code'].unique()

    for c in countrylist:
        data = datafull[datafull.Country_code == c]
        country = data['New_cases'].to_numpy().flatten()

        if np.max(country) > 0:
            countrynorm = (country - np.mean(country)) / (np.max(country) - np.min(country))
        else:
            continue

        countrynormexp = SimpleExpSmoothing(countrynorm, initialization_method="estimated").fit()
        countrynorm = countrynormexp.fittedvalues

        x = np.arange(1, countrynorm.size + 1)
        countrynorm = sm.lowess(countrynorm, x, frac=1/14)[:, 1]

        wave = np.zeros(countrynorm.shape, dtype=int)
        index = []
        flag = 0

        for i, val in enumerate(countrynorm):
            if val >= 0:
                wave[i] = 1
            else:
                wave[i] = 0

        for i in range(len(wave)):
            if wave[i] == 1 and flag == 0:
                index.append(i)
                flag = 1
            if wave[i] == 0 and flag == 1:
                index.append(i)
                flag = 0

        correction_factor = 6

        for i in range(0, len(index) - 1, 2):
            wavelength = index[i + 1] - index[i]
            correction = math.floor(wavelength / correction_factor)
            start = max(index[i] - correction, 0)
            wave[start:index[i] + 1] = 1

        data = data.assign(Wave=wave)
        dataset = pd.concat([dataset, data], ignore_index=True)

    return dataset


def labeller_2(WHO_Data):
    datafull = WHO_Data
    result_list = []

    countrylist = datafull['Country_code'].unique()

    for c in countrylist:
        data = datafull[datafull.Country_code == c]
        country = data['New_cases'].to_numpy().flatten()
        wavenum = data['Wave'].to_numpy().flatten()
        date = data['Date_reported'].to_numpy().flatten()

        caselist = [country[0]]
        current_wave = wavenum[0]

        for j in range(1, len(wavenum)):
            if wavenum[j] == current_wave:
                caselist.append(country[j])
            else:
                result_list.append([date[j - len(caselist)], c, current_wave, caselist.copy()])
                caselist = [country[j]]
                current_wave = wavenum[j]

        result_list.append([date[len(wavenum) - len(caselist)], c, current_wave, caselist.copy()])

    dataset = pd.DataFrame(result_list, columns=['Date', 'Country_code', 'Wave', 'Cases'])
    dataset.to_csv('COVID19_dataset_v2.csv', index=False)

    return dataset


def feature_gen(WHO_Data):
    data = WHO_Data.copy()

    # Expandindo a coluna 'Cases' que contém listas em múltiplas linhas
    data_expanded = data.explode('Cases').reset_index(drop=True)
    data_expanded['Cases'] = pd.to_numeric(data_expanded['Cases'], errors='coerce').fillna(0)

    features = []

    for (country, wave), group in data_expanded.groupby(['Country_code', 'Wave']):
        if len(group) < 21:
            continue  # Ignora grupos menores que a janela de 21 dias

        for i in range(20, len(group)):
            window = group.iloc[i-20:i+1]
            values = window['Cases'].to_numpy().astype(float)

            date = window['Date'].iloc[-1]

            # Estatísticas básicas
            mean = np.mean(values)
            var = np.var(values)
            max_v = np.max(values)
            min_v = np.min(values)
            median = np.median(values)
            range_v = max_v - min_v

            # LogReg e PDF protegidos contra std = 0
            std = np.std(values)
            if std == 0:
                logreg = 0
                pdf = 0
            else:
                z = (values[-1] - mean) / std
                logreg = 1 / (1 + math.exp(-z))
                pdf = (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * z ** 2)

            # Box-Cox protegido contra dados constantes
            if np.all(values == values[0]):
                bc_value = 0
            else:
                try:
                    bc_value = boxcox(values + 1e-3)[0][-1]
                except:
                    bc_value = 0

            # Transformações
            sqroot = math.sqrt(abs(values[-1]))
            sq = values[-1] ** 2
            log = math.log(abs(values[-1]) + 1e-9)

            # Diferenciais (D1 a D7) com padding se faltar dados
            diffs = np.diff(values)
            diffs_padded = np.pad(diffs[-7:], (7 - len(diffs[-7:]), 0), 'constant')

            # Coeficiente de variação
            cv = (std / mean) if mean != 0 else 0

            # Entropia protegida contra soma zero
            if np.sum(values) == 0:
                entropy = 0
            else:
                probs = values / (np.sum(values))
                entropy = -np.sum(probs * np.log(probs + 1e-9))

            # Monta a linha de features
            features.append([
                date, country, wave, *values,
                mean, var, max_v, min_v, median, range_v,
                logreg, pdf, bc_value, sqroot, sq, log,
                *diffs_padded, cv, entropy
            ])

    # Define os nomes das colunas
    columns = (
        ['Date', 'Country_code', 'Wave'] +
        [f'T{i+1}' for i in range(21)] +
        ['Mean', 'Variance', 'MAX', 'MIN', 'Median', 'Range',
         'LogReg', 'PDF', 'Box_Cox', 'Sqroot', 'Sq', 'Log',
         'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
         'CV', 'Entropy']
    )

    # Cria o dataframe final
    features_df = pd.DataFrame(features, columns=columns)

    # Salva em CSV
    features_df.to_csv('COVID19_dataset_v3.csv', index=False)

    return features_df


if __name__ == "__main__":
    file = pd.read_csv("WHO-COVID-19-global-data.csv", na_filter=False)
    datas_1 = labeller_1(file)
    datas_1.to_csv('COVID19_dataset_v1.csv', index=False)
    datas_2 = labeller_2(datas_1)
    datas_3 = feature_gen(datas_2)
