# %% [markdown]
# ## Trabalho CPAM 2024
# #### Título:  A Influência da Precipitação na Elevação do Nível do Rio Cubatão do Sul em SC
# #### Data: Agosto/2024
# #### Equipe: 
# * Alexandre Nuernberg - alexandreberg@gmail.com
# * Eduardo Conceição
# * Gilmar Florêncio
# * Juliana Portella Bernardes
# * Nicolas Firmino Flores
# 

# %% [markdown]
# ***
# ***
# ***
# # Definição das bibliotecas
# ***
# ***
# ***

# %%
import os                           # faz a movimentação de arquivos, pastas e variáveis de ambiente
import pandas as pd                 # trabalhar com planilhas e DataFrames e ferramentas para manipulação e análise de dados
import matplotlib.pyplot as plt     # biblioteca para visualização de dados semelhante ao MATLAB, para criar gráficos e visualizações de alta qualidade, como gráficos de linha, histogramas, dispersões
import numpy as np                  # para computação científica, fornece suporte para arrays multidimensionais e funções matemáticas
import seaborn as sns               # para criar gráficos estatísticos atraentes e informativos, como gráficos de dispersão com várias variáveis e mapas de calor
from scipy.stats import norm        # estatística distribuição normal (gaussiana) como cálculos de probabilidade, geração de números aleatórios e ajuste de dados a uma distribuição normal
import netCDF4 as nc                # abrir arquivos gerados pelo Merge no formato netCDF4
import geopandas as gpd             # trabalhar com mapas de coordenadas geográficas
import itertools                    # combinaçãoes matemáticas

# %% [markdown]
# ***
# ***
# ***
# #### Análise das cotas da estação: ETA CASAN MONTANTE
# ***
# ***
# ***
# 

# %% [markdown]
# #### Análise das cotas da estação: ETA CASAN MONTANTE
# ##### Arquivo: Cotas_Estacao_100.csv
# ##### Cota de Alerta: 214 cm
# #####  Código: 84150100

# %%
# Caminho do arquivo (use r'' para evitar problemas com caracteres especiais)
caminho_arquivo = r'../input_data/Cotas_Estacao_100.csv'

# Leitura do CSV com separador ';'
df = pd.read_csv(caminho_arquivo, sep=';')

# Create a list to store the data in the desired format
data = []

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    data_hora = pd.to_datetime(row['Data'] + ' ' + row['hora'])
    for dia, cota in row.items():
        # Skip the 'Data' and 'hora' columns
        if dia in ['Data', 'hora']:
            continue
        # Convert the 'dia' to integer
        dia = int(dia)
        # Check if 'dia' is within valid range (1-31)
        if 1 <= dia <= 31:
            # Create a new datetime index and add data to the list
            data.append({'Data': data_hora + pd.to_timedelta(dia - 1, unit='d'), 'Cotas': cota, 'Data_hora': data_hora})

# Create a new DataFrame from the list of data
df_unpivot = pd.DataFrame(data)

# Get unique values from 'Data_hora' column
unique_data_hora = df_unpivot['Data_hora'].unique()

# If there are non-numeric values in the 'Data_hora' column, convert them to numeric, ignoring errors
if not all(isinstance(x, (int, float)) for x in unique_data_hora):
    df_unpivot['Data_hora'] = pd.to_numeric(df_unpivot['Data_hora'], errors='coerce')

# Convert the 'Data_hora' column to datetime64[ns]
df_unpivot['Data_hora'] = pd.to_datetime(df_unpivot['Data_hora'], unit='ns')

# Convertendo a coluna 'Data' para o tipo datetime
df_unpivot['Data'] = pd.to_datetime(df_unpivot['Data'])

# Criando as colunas separadas para data e hora
df_unpivot['data'] = df_unpivot['Data'].dt.date
df_unpivot['hora'] = df_unpivot['Data'].dt.time

# Renomeando a coluna "Cotas" para "cotas"
df_unpivot = df_unpivot.rename(columns={'Cotas': 'cotas'})

# Reset the index to make 'Data' a regular column
df_unpivot = df_unpivot.reset_index()

# Drop the 'Data' and 'Data_hora' columns
df_unpivot_filtered = df_unpivot.drop(columns=['Data', 'Data_hora'])

# Display the first 10 rows of the resulting DataFrame
print("Cabeçalho do arquivo Cotas_Estacao_100.csv:\n")
print(df_unpivot_filtered.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# Debug:
# Exibindo o DataFrame após a renomeação
display(df_unpivot_filtered)

# Exibindo informações iniciais
# print("\nHeader do arquivo Cotas_Estacao_100.csv:\n")
# df_unpivot_filtered.head(62)  # Use print() para exibir o resultado do head()

# %% [markdown]
# ### Agrupando o DataFrame pela coluna 'data' e calculando o máximo da coluna 'cotas' para cada dia
# ##### Isso é feito porque existem dois valores de cotas diárias um das 7:00 e outro das 17:00
# ##### Apenas o valor de cota mais alta do dia será usado na análise e o outro será excluido
# #### DF: *df_cs_max_cota_dia*

# %%

df_cs_max_cota_dia = df_unpivot_filtered.groupby('data')['cotas'].max().reset_index()

# Exibindo o cabeçalho do dataframe
print("\nDF df_cs_max_cota_dia (Máxima cota do dia da Estação CS - CASAN MONTANTE):")
print(df_cs_max_cota_dia.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# Debug:
# print("\nChecagem do tipo de dados (DType) das colunas:")
df_cs_max_cota_dia.info()

# %% [markdown]
# #### A estação ETA CASAN MONTANTE possui uma cota de alerta de inundação de 214cm
# ##### Filtrando os dados com cota igual ou acima do nivel de alerta 214 cm
# ##### Aqui serão filtrados exclusivamente os valores de cotas dentro desse range, os demais serão descartados.
# DF: *df_cs_max_cota_alerta*

# %%
# Classificando por data em ordem crescente (opcional, mas recomendado)
df_cs_max_cota_dia['data'] = df_cs_max_cota_dia['data'].sort_values()

# Filtrando as cotas maiores ou iguais a 214 cm no período
df_cs_max_cota_alerta = df_cs_max_cota_dia[df_cs_max_cota_dia['cotas'] >= 214]

# Salvando o DataFrame filtrado no período
df_cs_max_cota_alerta.to_csv('../output_data/database/df_cs_max_cota_alerta.csv')  # Nome do arquivo ajustado

#Debug:
# Exibindo informações sobre o DataFrame filtrado
print("\nInfo DF df_cs_max_cota_alerta:")
df_cs_max_cota_alerta.info()

# Exibindo o DataFrame resultante
print("\nDF df_cs_max_cota_alerta:")
print("Cotas acima da cota de alerta (214cm) da Estação CS - CASAN MONTANTE):")
# display(df_cs_max_cota_alerta)
print(df_cs_max_cota_alerta.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# %% [markdown]
# #### Pegando apenas os valores do período de 2000 à 2022 para análise
# ##### Convertendo a coluna data de texto para o formato de data
# ##### Nivelando os dados para a período comum entre os dados de nível, chuva e maré (2000-01-01 até 2022-12-31)
# ##### DF: *df_cs_max_cota_dia_periodo*

# %%
# Erro de lógica
# Pegando apenas os valores do período de 2000 à 2022 para análise
# Convertendo a coluna data de texto para o formato de data
# Nivelando os dados para a período comum entre os dados de nível, chuva e maré (2000-01-01 até 2022-12-31)
# Nesse período existem 8.401 dias, só que 275 dias estão faltando (sem dados).

# Convertendo a coluna 'data' para o tipo datetime
df_cs_max_cota_alerta['data'] = pd.to_datetime(df_cs_max_cota_alerta['data'])

# Filtrando o DataFrame pela coluna 'data' no intervalo desejado
df_cs_max_cota_dia_periodo = df_cs_max_cota_alerta[(df_cs_max_cota_alerta['data'] >= '2000-01-01') & (df_cs_max_cota_alerta['data'] <= '2022-12-31')]

#Debug:
# análises estatísticas dos dados entre 2000 e 2022:
display(df_cs_max_cota_dia_periodo.describe())

# contagem dos registros nesse periodo:
# df_cs_max_cota_dia_periodo.info()

# Exibindo o DataFrame resultante
print("\nDF df_cs_max_cota_dia_periodo:")
print(df_cs_max_cota_dia_periodo)
print(df_cs_max_cota_dia_periodo.head(10).to_markdown(index=False, numalign="left", stralign="left"))
# df_cs_max_cota_dia_periodo.info()


# %% [markdown]
# ### Substituir o nome da colunas "cotas" que é semelhante nos dados das duas estações de fluviométrics por "cotas_cs".
# ### Para identificar que é da estação CASAN MONTANTE (CS)
# 

# %%
df_cs_max_cota_dia_periodo = df_cs_max_cota_dia_periodo.rename(columns={'cotas': 'cotas_cs'})

# Exibindo o cabeçalho do dataframe
print("\nDF df_cs_max_cota_dia_periodo (Máxima cota do dia da Estação CS - CASAN MONTANTE):")
print(df_cs_max_cota_dia_periodo.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# Salva do DataFrame Eta casan cota máxima do dia de todo banco dados.
df_cs_max_cota_dia_periodo.to_csv('../output_data/database/df_cs_max_cota_dia_periodo.csv') 

# %% [markdown]
# #### Plotagem dos gráficos de Histograma e Boxplot para a estação ETA CASAN Montante

# %%
# Criação dos subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 linha, 2 colunas

# Histograma
sns.histplot(df_cs_max_cota_dia_periodo['cotas_cs'], ax=axes[0], kde=True, color='green')
axes[0].set_xlabel('cotas_cs (cm)', fontsize=18)
axes[0].set_ylabel('Ocorrências', fontsize=18)
axes[0].set_title('Histograma de cotas_cs', fontsize=24) 

# Criação da Tabela:
# Calcula as estatísticas para a tabela
count = df_cs_max_cota_dia_periodo['cotas_cs'].count()
min_cota = df_cs_max_cota_dia_periodo['cotas_cs'].min()
max_cota = df_cs_max_cota_dia_periodo['cotas_cs'].max()
media = df_cs_max_cota_dia_periodo['cotas_cs'].mean()
mediana = df_cs_max_cota_dia_periodo['cotas_cs'].median()
desvio_padrao = df_cs_max_cota_dia_periodo['cotas_cs'].std()  # Calcula o desvio padrão
nivel_alerta = 214

# Dados da tabela
dados_tabela = {
    'Estatísticas': ['N° Cotas', 'Cota Mínima', 'Cota Máxima', 'Média', 'Mediana', 'Desvio Padrão', 'Cota de Alerta'],
    'Valores': [count, min_cota, max_cota, f'{media:.1f}', f'{mediana:.1f}', f'{desvio_padrao:.1f}', nivel_alerta]
}

# Cria um DataFrame para a tabela
df_tabela = pd.DataFrame(dados_tabela)

# Cria a tabela no subplot do histograma
# tabela = axes[0].table(cellText=df_tabela.values, colLabels=df_tabela.columns, loc='upper right', cellLoc='center', bbox=[0.455, 0.3425, 0.5, 0.45])
tabela = axes[0].table(cellText=df_tabela.values, colLabels=df_tabela.columns, loc='upper right', cellLoc='center', bbox=[0.3425, 0.3425, 0.5, 0.45])

# Formata a tabela (opcional)
tabela.auto_set_font_size(False)
tabela.set_fontsize(14)
tabela.scale(1, 1.5)  # Ajusta a altura das células

# Fim tabela

# Calcula e plota a média no histograma
media = df_cs_max_cota_dia_periodo['cotas_cs'].mean()
axes[0].axvline(media, color='red', linestyle='dashed', linewidth=1, label=f'Média: {media:.1f} cm')
axes[0].legend()

# Boxplot
sns.boxplot(y=df_cs_max_cota_dia_periodo['cotas_cs'], ax=axes[1], color='lightgreen', showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"black", 
                       "markeredgecolor":"black",
                      "markersize":"10"})

# Anotações no boxplot (mediana e quartis)
mediana = df_cs_max_cota_dia_periodo['cotas_cs'].median()
q1 = df_cs_max_cota_dia_periodo['cotas_cs'].quantile(0.25)
q3 = df_cs_max_cota_dia_periodo['cotas_cs'].quantile(0.75)
iqr = q3 - q1
limite_inferior_outlier = q1 - 1.5 * iqr
limite_superior_outlier = q3 + 1.5 * iqr

axes[1].text(0.1, mediana, f'Mediana (50%): {mediana:.1f}', fontsize=10, color='black', 
             bbox=dict(facecolor='yellow', edgecolor='yellow', boxstyle='round,pad=0.3'))
axes[1].text(0.1, q1, f'Q1 (25%): {q1:.1f}', fontsize=10, color='black')
axes[1].text(0.1, q3, f'Q3 (75%): {q3:.1f}', fontsize=10, color='black')

# Calcula e plota a média no boxplot
axes[1].axhline(media, color='red', linestyle='dashed', linewidth=1, label=f'Média: {media:.1f} cm')
axes[1].legend()

# Destacar outliers no boxplot
outliers = df_cs_max_cota_dia_periodo[(df_cs_max_cota_dia_periodo['cotas_cs'] < limite_inferior_outlier) | (df_cs_max_cota_dia_periodo['cotas_cs'] > limite_superior_outlier)]['cotas_cs']
axes[1].scatter(np.zeros_like(outliers), outliers, color='red', label='Outliers')  
axes[1].legend()

axes[1].set_ylabel('cotas_cs (cm)', fontsize=18)
axes[1].set_title('Boxplot de cotas_cs', fontsize=24) 


# Título geral do gráfico
fig.suptitle('Estação ETA CASAN Montante - Cotas acima da Cota de Alerta', fontsize=24)

# Ajustes para melhor visualização
plt.tight_layout()

# Save the plot with specified resolution and size
plt.savefig('../output_data/images/grafico_histograma_ETA_CASAN_MONTANTE.png', dpi=900)

# Exibição do gráfico
plt.show()

#Debug:
# análises estatísticas dos dados entre 2000 e 2022:
display(df_cs_max_cota_dia_periodo.describe())


# %% [markdown]
# ***
# ***
# ***
# #### Análise das cotas da estação: POÇO FUNDO
# ***
# ***
# ***

# %% [markdown]
# #### Análise das cotas da estação: POÇO FUNDO
# ##### Arquivo: Cotas_Estacao_000.csv
# ##### Cota de Alerta: 234 cm
# #####  Código: 84100000

# %%
# Caminho do arquivo (use r'' para evitar problemas com caracteres especiais)
caminho_arquivo = r'../input_data/Cotas_Estacao_000.csv'

# Leitura do CSV com separador ';'
df = pd.read_csv(caminho_arquivo, sep=';')

# Create a list to store the data in the desired format
data = []

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    data_hora = pd.to_datetime(row['Data'] + ' ' + row['hora'])
    for dia, cota in row.items():
        # Skip the 'Data' and 'hora' columns
        if dia in ['Data', 'hora']:
            continue
        # Convert the 'dia' to integer
        dia = int(dia)
        # Check if 'dia' is within valid range (1-31)
        if 1 <= dia <= 31:
            # Create a new datetime index and add data to the list
            data.append({'Data': data_hora + pd.to_timedelta(dia - 1, unit='d'), 'Cotas': cota, 'Data_hora': data_hora})

# Create a new DataFrame from the list of data
df_unpivot = pd.DataFrame(data)

# Get unique values from 'Data_hora' column
unique_data_hora = df_unpivot['Data_hora'].unique()

# If there are non-numeric values in the 'Data_hora' column, convert them to numeric, ignoring errors
if not all(isinstance(x, (int, float)) for x in unique_data_hora):
    df_unpivot['Data_hora'] = pd.to_numeric(df_unpivot['Data_hora'], errors='coerce')

# Convert the 'Data_hora' column to datetime64[ns]
df_unpivot['Data_hora'] = pd.to_datetime(df_unpivot['Data_hora'], unit='ns')

# Convertendo a coluna 'Data' para o tipo datetime
df_unpivot['Data'] = pd.to_datetime(df_unpivot['Data'])

# Criando as colunas separadas para data e hora
df_unpivot['data'] = df_unpivot['Data'].dt.date
df_unpivot['hora'] = df_unpivot['Data'].dt.time

# Renomeando a coluna "Cotas" para "cotas"
df_unpivot = df_unpivot.rename(columns={'Cotas': 'cotas'})

# Reset the index to make 'Data' a regular column
df_unpivot = df_unpivot.reset_index()

# Drop the 'Data' and 'Data_hora' columns
df_unpivot_filtered = df_unpivot.drop(columns=['Data', 'Data_hora'])

# Display the first 10 rows of the resulting DataFrame
print("Cabeçalho do arquivo Cotas_Estacao_000.csv:\n")
print(df_unpivot_filtered.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# Debug:
# Exibindo o DataFrame após a renomeação
display(df_unpivot_filtered)

# Exibindo informações iniciais
# print("\nHeader do arquivo Cotas_Estacao_100.csv:\n")
# df_unpivot_filtered.head(62)  # Use print() para exibir o resultado do head()

# %% [markdown]
# 

# %% [markdown]
# ### Agrupando o DataFrame pela coluna 'data' e calculando o máximo da coluna 'cotas' para cada dia
# ##### Isso é feito porque existem dois valores de cotas diárias um das 7:00 e outro das 17:00
# ##### Apenas o valor de cota mais alta do dia será usado na análise e o outro será excluido
# #### DF: *df_pf_max_cota_dia*

# %%

df_pf_max_cota_dia = df_unpivot_filtered.groupby('data')['cotas'].max().reset_index()

# Exibindo o cabeçalho do dataframe
print("\nDF df_pf_max_cota_dia (Máxima cota do dia da Estação PF - Poço Fundo):")
print(df_pf_max_cota_dia.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# Debug:
# print("\nChecagem do tipo de dados (DType) das colunas:")
df_pf_max_cota_dia.info()

# %% [markdown]
# #### A estação Poço Fundo possui uma cota de alerta de inundação de 234
# ##### Filtrando os dados com cota igual ou acima do nivel de alerta 234 cm
# ##### Aqui serão filtrados exclusivamente os valores de cotas dentro desse range, os demais serão descartados.
# DF: *df_pf_max_cota_alerta*

# %%
# Classificando por data em ordem crescente (opcional, mas recomendado)
df_pf_max_cota_dia['data'] = df_pf_max_cota_dia['data'].sort_values()

# Filtrando as cotas maiores ou iguais a 214 cm no período
df_pf_max_cota_alerta = df_pf_max_cota_dia[df_pf_max_cota_dia['cotas'] >= 234]

# Salvando o DataFrame filtrado no período
df_pf_max_cota_alerta.to_csv('../output_data/database/df_pf_max_cota_alerta.csv')  # Nome do arquivo ajustado

#Debug:
# Exibindo informações sobre o DataFrame filtrado
# print("\nInfo DF df_pf_max_cota_alerta:")
# df_pf_max_cota_alerta.info()

# Exibindo o DataFrame resultante
print("\nDF df_pf_max_cota_alerta:")
print("Cotas acima da cota de alerta (234cm) da Estação PF - Poço Fundo):")
# display(df_pf_max_cota_alerta)
print(df_pf_max_cota_alerta.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# %% [markdown]
# #### Pegando apenas os valores do período de 2000 à 2022 para análise
# ##### Convertendo a coluna data de texto para o formato de data
# ##### Nivelando os dados para a período comum entre os dados de nível, chuva e maré (2000-01-01 até 2022-12-31)
# ##### DF: *df_pf_max_cota_dia_periodo*

# %%
# Erro de lógica
# Pegando apenas os valores do período de 2000 à 2022 para análise
# Convertendo a coluna data de texto para o formato de data
# Nivelando os dados para a período comum entre os dados de nível, chuva e maré (2000-01-01 até 2022-12-31)

# Convertendo a coluna 'data' para o tipo datetime
df_pf_max_cota_alerta['data'] = pd.to_datetime(df_pf_max_cota_alerta['data'])

# Filtrando o DataFrame pela coluna 'data' no intervalo desejado
df_pf_max_cota_dia_periodo = df_pf_max_cota_alerta[(df_pf_max_cota_alerta['data'] >= '2000-01-01') & (df_pf_max_cota_alerta['data'] <= '2022-12-31')]

#Debug:
# análises estatísticas dos dados entre 2000 e 2022:
display(df_pf_max_cota_dia_periodo.describe())

# contagem dos registros nesse periodo:
# df_pf_max_cota_dia_periodo.info()

# Exibindo o DataFrame resultante
print("\nDF df_pf_max_cota_dia_periodo:")
# print(df_pf_max_cota_dia_periodo)
print(df_pf_max_cota_dia_periodo.head(10).to_markdown(index=False, numalign="left", stralign="left"))
# df_pf_max_cota_dia_periodo.info()

# %% [markdown]
# ### Substituir o nome da colunas "cotas" que é semelhante nos dados das duas estações de fluviométrics por "cotas_pf".
# ### Para identificar que é da estação  Poço Fundo (PF)

# %%
df_pf_max_cota_dia_periodo = df_pf_max_cota_dia_periodo.rename(columns={'cotas': 'cotas_pf'})

# Exibindo o cabeçalho do dataframe
print("\nDF df_pf_max_cota_dia (Máxima cota do dia da Estação Poço Fundo):")
print(df_pf_max_cota_dia_periodo.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# Salva do DataFrame Eta casan cota máxima do dia de todo banco dados.
df_pf_max_cota_dia_periodo.to_csv('../output_data/database/df_pf_max_cota_dia_periodo.csv') 

# %% [markdown]
# #### Plotagem dos gráficos de Histograma e Boxplot para a estação Poço Fundo

# %%
# Criação dos subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 linha, 2 colunas

# Histograma
sns.histplot(df_pf_max_cota_dia_periodo['cotas_pf'], ax=axes[0], kde=True, color='magenta')
axes[0].set_xlabel('cotas_pf (cm)', fontsize=18)
axes[0].set_ylabel('Ocorrências', fontsize=18)
axes[0].set_title('Histograma de cotas_pf', fontsize=24) 

# Criação da Tabela:
# Calcula as estatísticas para a tabela
count = df_pf_max_cota_dia_periodo['cotas_pf'].count()
min_cota = df_pf_max_cota_dia_periodo['cotas_pf'].min()
max_cota = df_pf_max_cota_dia_periodo['cotas_pf'].max()
media = df_pf_max_cota_dia_periodo['cotas_pf'].mean()
mediana = df_pf_max_cota_dia_periodo['cotas_pf'].median()
desvio_padrao = df_pf_max_cota_dia_periodo['cotas_pf'].std()  # Calcula o desvio padrão
nivel_alerta = 234

# Dados da tabela
dados_tabela = {
    'Estatísticas': ['N° Cotas', 'Cota Mínima', 'Cota Máxima', 'Média', 'Mediana', 'Desvio Padrão', 'Cota de Alerta'],
    'Valores': [count, min_cota, max_cota, f'{media:.1f}', f'{mediana:.1f}', f'{desvio_padrao:.1f}', nivel_alerta]
}

# Cria um DataFrame para a tabela
df_tabela = pd.DataFrame(dados_tabela)

# Cria a tabela no subplot do histograma
# tabela = axes[0].table(cellText=df_tabela.values, colLabels=df_tabela.columns, loc='upper right', cellLoc='center', bbox=[0.455, 0.3425, 0.5, 0.45])
tabela = axes[0].table(cellText=df_tabela.values, colLabels=df_tabela.columns, loc='upper right', cellLoc='center', bbox=[0.3425, 0.3425, 0.5, 0.45])

# Formata a tabela (opcional)
tabela.auto_set_font_size(False)
tabela.set_fontsize(14)
tabela.scale(1, 1.5)  # Ajusta a altura das células

# Fim tabela

# Calcula e plota a média no histograma
media = df_pf_max_cota_dia_periodo['cotas_pf'].mean()
axes[0].axvline(media, color='red', linestyle='dashed', linewidth=1, label=f'Média: {media:.1f} cm')
axes[0].legend()

# Boxplot
sns.boxplot(y=df_pf_max_cota_dia_periodo['cotas_pf'], ax=axes[1], color='pink', showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"black", 
                       "markeredgecolor":"black",
                      "markersize":"10"})

# Anotações no boxplot (mediana e quartis)
mediana = df_pf_max_cota_dia_periodo['cotas_pf'].median()
q1 = df_pf_max_cota_dia_periodo['cotas_pf'].quantile(0.25)
q3 = df_pf_max_cota_dia_periodo['cotas_pf'].quantile(0.75)
iqr = q3 - q1
limite_inferior_outlier = q1 - 1.5 * iqr
limite_superior_outlier = q3 + 1.5 * iqr

axes[1].text(0.1, mediana, f'Mediana (50%): {mediana:.1f}', fontsize=10, color='black', 
             bbox=dict(facecolor='yellow', edgecolor='yellow', boxstyle='round,pad=0.3'))
axes[1].text(0.1, q1, f'Q1 (25%): {q1:.1f}', fontsize=10, color='black')
axes[1].text(0.1, q3, f'Q3 (75%): {q3:.1f}', fontsize=10, color='black')

# Calcula e plota a média no boxplot
axes[1].axhline(media, color='red', linestyle='dashed', linewidth=1, label=f'Média: {media:.1f} cm')
axes[1].legend()

# Destacar outliers no boxplot
outliers = df_pf_max_cota_dia_periodo[(df_pf_max_cota_dia_periodo['cotas_pf'] < limite_inferior_outlier) | (df_pf_max_cota_dia_periodo['cotas_pf'] > limite_superior_outlier)]['cotas_pf']
axes[1].scatter(np.zeros_like(outliers), outliers, color='red', label='Outliers')  
axes[1].legend()

axes[1].set_ylabel('cotas_pf (cm)', fontsize=18)
axes[1].set_title('Boxplot de cotas_pf', fontsize=24) 


# Título geral do gráfico
fig.suptitle('Estação Poço Fundo - Cotas acima da Cota de Alerta', fontsize=24)

# Ajustes para melhor visualização
plt.tight_layout()

# Save the plot with specified resolution and size
plt.savefig('../output_data/images/grafico_histograma_Poco_Fundo.png', dpi=900)

# Exibição do gráfico
plt.show()

#Debug:
# análises estatísticas dos dados entre 2000 e 2022:
display(df_pf_max_cota_dia_periodo.describe())


# %% [markdown]
# ***
# ***
# ***
# ####  Tratamento dos Dados de Maré
# ***
# ***
# ***

# %% [markdown]
# ### Tratamento dos Dados de Maré
# #### Arquivo: 60246008710101199431122024PREVMAXMINCOL.txt
# #### Estação: CAPITANIA DOS PORTOS DE SANTA CATARINA

# %% [markdown]
# ### Lendo o arquivo .txt e especificando a codificação:

# %%
# Dados de maré fornecidos pela marinha

# Lendo o arquivo .txt com o pandas e especificando a codificação
df_mare = pd.read_csv("../input_data/60246008710101199431122024PREVMAXMINCOL.txt", delimiter='\t', encoding='ISO-8859-1')
                       
df_mare.info()

# Exibindo as primeiras linhas do DataFrame
print(df_mare.head(20))

# %% [markdown]
# #### Limpeza e preparação da base de dados das marés:

# %%
# Excluíndo o cabeçalho que se encontra no intervalo do índice 0 até 12
df_mare = df_mare.drop(index=range(0, 13))

#Debug:
# df_mare.info()
# Exibindo o DataFrame atualizado
# display(df_mare)

# Exibindo o cabeçalho do dataframe
print(df_mare.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# %% [markdown]
# #### Ajustando a data e a hora
# ##### DF: *df_mare*

# %%
# Separando data/hora
# Criando novas colunas com base nos caracteres específicos
df_mare['data'] = df_mare['PREVISÃO MÁXIMAS E MÍNIMAS - 01/01/1994 A 31/12/2024'].str[:10]
df_mare['hora'] = df_mare['PREVISÃO MÁXIMAS E MÍNIMAS - 01/01/1994 A 31/12/2024'].str[10:16]
df_mare['altura_mare'] = df_mare['PREVISÃO MÁXIMAS E MÍNIMAS - 01/01/1994 A 31/12/2024'].str[16:]

# Removendo a coluna original
df_mare = df_mare.drop(columns=['PREVISÃO MÁXIMAS E MÍNIMAS - 01/01/1994 A 31/12/2024'])

# Exibindo o DataFrame resultante
# display(df_mare)

# Convertendo a coluna 'data' para o tipo datetime
df_mare['data'] = pd.to_datetime(df_mare['data'], format='%d/%m/%Y')

# Removendo espaços em branco no início e no final da string da coluna 'hora'
df_mare['hora'] = df_mare['hora'].str.strip()

# Convertendo a coluna 'hora' para o tipo datetime
df_mare['hora'] = pd.to_datetime(df_mare['hora'], format='%H:%M').dt.time

# Convertendo a coluna 'altura_mare' para o tipo inteiro
df_mare['altura_mare'] = df_mare['altura_mare'].astype(float)

# DEBUG: 
# Exibindo as primeiras linhas do DataFrame atualizado
# df_mare.info()

# Exibindo o cabeçalho do dataframe
print("\nDF: df_mare (PREVISÃO MÁXIMAS E MÍNIMAS - 01/01/1994 A 31/12/2024):")
print(df_mare.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# %% [markdown]
# #### Agrupando o DataFrame pela coluna 'data' e calculando o máximo da coluna 'altura_mare' para cada dia
# #### DF: *df_mare_max_cota_dia*

# %%
df_mare_max_cota_dia = df_mare.groupby('data')['altura_mare'].max().reset_index()

# Exibindo o DataFrame resultante
# print(df_mare_max_cota_dia)

# Exibindo o cabeçalho do dataframe
print("\nDF df_cs_max_cota_dia (Máxima cota de maré do dia da Estação Capitana dos Portos de SC):")
# Primeiras 5 linhas
print("Primeiras 5 linhas:")
print(df_mare_max_cota_dia.head().to_markdown(index=False, numalign="left", stralign="left"))

# Últimas 5 linhas
print("\nÚltimas 5 linhas:")
print(df_mare_max_cota_dia.tail().to_markdown(index=False, numalign="left", stralign="left"))

# %% [markdown]
# #### Filtrando o intervalo de datas de 01/01/2000 até 01/12/2022
# ##### DF: *df_mare_max_cota_dia_periodo*

# %%
df_mare_max_cota_dia_periodo = df_mare_max_cota_dia[(df_mare_max_cota_dia['data'] >= '2000-01-01') & (df_mare_max_cota_dia['data'] <= '2022-12-31')]

# print(df_mare_max_cota_dia_periodo)
# Exibindo o cabeçalho do dataframe
print("\nDF df_mare_max_cota_dia_periodo (Máxima cota de maré do dia da Estação Capitana dos Portos de SC):")
# Primeiras 5 linhas
print("Primeiras 5 linhas:")
print(df_mare_max_cota_dia_periodo.head().to_markdown(index=False, numalign="left", stralign="left"))

# Últimas 5 linhas
print("\nÚltimas 5 linhas:")
print(df_mare_max_cota_dia_periodo.tail().to_markdown(index=False, numalign="left", stralign="left"))

# Salva do DataFrame Eta casasn cota máxima do dia de todo banco dados.
df_mare_max_cota_dia_periodo.to_csv('../output_data/database/df_mare_max_cota_dia_periodo.csv') 

# %% [markdown]
# ***
# ***
# ***
# ### Tratando os mapas e bacia hidrográficas
# ***
# ***
# ***

# %% [markdown]
# ### Tratando os mapas e bacia hidrográficas

# %% [markdown]
# ###  Abrindo um arquivo NETCDF
# #### Arquivo: MERGE_CPTEC_DAILY_PREC_SC_2000_2022.nc

# %%
# Caminho para o arquivo NetCDF
caminho_arquivo = r'../input_data/MERGE_CPTEC_DAILY_PREC_SC_2000_2022.nc'

# Abre o arquivo NetCDF para leitura
arquivo_netcdf = nc.Dataset(caminho_arquivo)
print(arquivo_netcdf.variables.keys())

# Seleciona a variável "precipitacao"
variavel_precipitacao = arquivo_netcdf.variables['prec']
variavel_latitudes=arquivo_netcdf.variables['lat']
variavel_longitudes=arquivo_netcdf.variables['lon']
variavel_tempos=arquivo_netcdf.variables['time']

# Lê os dados da variável de precipitação
prec_data = variavel_precipitacao[:]
latitudes = variavel_latitudes[:]
longitudes = variavel_longitudes[:]
tempos=variavel_tempos[:]

# %% [markdown]
# ### Associando as variáveis trazidas do NetCdf a um DataFrame

# %%
# Obtém as dimensões dos dados
time_dim = prec_data.shape[0]
lat_dim = prec_data.shape[1]
lon_dim = prec_data.shape[2]

# Cria os índices de tempo, latitude e longitude
index_time = pd.date_range(start='2000-01-01', periods=time_dim, freq='D')
index_lat = range(lat_dim)
index_lon = range(lon_dim)

# Cria o DataFrame
df_prec = pd.DataFrame(prec_data.reshape((time_dim, -1)),
                        index=index_time,
                        columns=pd.MultiIndex.from_product([index_lat, index_lon], names=['lon', 'lat']))

df_prec = df_prec.rename_axis('data')


# Mostra as primeiras linhas do DataFrame
#print(df_prec.head())


# Salva o DataFrame em um arquivo CSV
df_prec.to_csv(r'../output_data/database/dadosB_precipitacao.csv')

# Supondo que index_lon, index_lat e index_time estão definidos
dados_empilhados = []  # Inicializa uma lista vazia para armazenar os dados

for i in index_lon:
    for j in index_lat:
        # Supondo que df_prec é seu DataFrame original
        # Adiciona os dados de df_prec[i][j][index_time] à lista dados_empilhados
        dados_empilhados.append(df_prec[j][i][index_time])

# Crie o DataFrame a partir dos dados empilhados
dfEmpilhado = pd.concat(dados_empilhados, axis=1, keys=[f'Lon_{longitudes[i].round(2)} Lat_{latitudes[j].round(2)}' for i in index_lon for j in index_lat])

# Verifica as primeiras linhas do DataFrame empilhado
print("DF: dfEmpilhado:")
dfEmpilhado.head()

# %%
# Renomeando o dataframe:
df_precipitacao_bacia = dfEmpilhado
df_precipitacao_bacia.columns

# %% [markdown]
# #### Definindo as coordenadas que estão dentro da Bacia Hidrográfica do Rio Cubatão do Sul - SC

# %%
coordenadas_bacia_cubatao_sul = [
    'Lon_-48.95 Lat_-27.75',
    'Lon_-48.95 Lat_-27.65',
    'Lon_-48.85 Lat_-27.85',
    'Lon_-48.85 Lat_-27.65',
    'Lon_-48.75 Lat_-27.75',
    'Lon_-48.75 Lat_-27.65',
    'Lon_-48.65 Lat_-27.55',
    'Lon_-48.55 Lat_-27.65',
    'Lon_-48.45 Lat_-27.55',
    'Lon_-48.45 Lat_-27.45'
]

# ponto excluído da bacia - vargem do braço (-27.75, -48.55),  por estar após a estação ETA casan montante.

print("DF: df_precipitacao_bacia:")
df_precipitacao_bacia[coordenadas_bacia_cubatao_sul]
display(df_precipitacao_bacia[coordenadas_bacia_cubatao_sul])

# %% [markdown]
# #### Criando uma lista para armazenar os resultados

# %%
# Criando uma lista para armazenar os resultados
resultados = []

# Percorra as linhas do DataFrame
for indice, linha in df_precipitacao_bacia.iterrows():
    # Mantenha o índice como "data"
    data = indice
    
    # Some todas as colunas para a linha atual
    soma_linha = linha.sum()
    
    # Adicione a soma à lista de resultados
    resultados.append({'data': data, 'soma': soma_linha})

# Converta a lista de resultados em um DataFrame
df_precipitacao_bacia_total_dia = pd.DataFrame(resultados)
df_precipitacao_bacia_total_dia = df_precipitacao_bacia_total_dia.rename(columns={'soma': 'precipitacao'})

# Exiba o DataFrame resultante
print("DF: df_precipitacao_bacia_total_dia:")
display(df_precipitacao_bacia_total_dia)

# %% [markdown]
# #### Calculando as precipitações acumuladas (Semanal, 14 dias, 21 dias, 28 dias e 1 mês)

# %%
# Converta a coluna 'data' para o tipo datetime, se ainda não estiver
df_precipitacao_bacia_total_dia['data'] = pd.to_datetime(df_precipitacao_bacia_total_dia['data'])

# Ordene o DataFrame pela coluna 'data' se ainda não estiver ordenado
df_precipitacao_bacia_total_dia = df_precipitacao_bacia_total_dia.sort_values(by='data')

# Calcule a soma acumulada dos 7 dias anteriores e desloque os valores para representar os 7 dias anteriores
df_precipitacao_bacia_total_dia['d-7'] = df_precipitacao_bacia_total_dia['precipitacao'].rolling(window=7).sum().shift(periods=1)

# Calcule a soma acumulada dos 14 dias anteriores e desloque os valores para representar os 14 dias anteriores
df_precipitacao_bacia_total_dia['d-14'] = df_precipitacao_bacia_total_dia['precipitacao'].rolling(window=14).sum().shift(periods=1)

# Calcule a soma acumulada dos 21 dias anteriores e desloque os valores para representar os 21 dias anteriores
df_precipitacao_bacia_total_dia['d-21'] = df_precipitacao_bacia_total_dia['precipitacao'].rolling(window=21).sum().shift(periods=1)

# Calcule a soma acumulada dos 28 dias anteriores e desloque os valores para representar os 28 dias anteriores
df_precipitacao_bacia_total_dia['d-28'] = df_precipitacao_bacia_total_dia['precipitacao'].rolling(window=28).sum().shift(periods=1)

# Calcule a soma acumulada dos 30 dias anteriores e desloque os valores para representar os 30 dias anteriores
df_precipitacao_bacia_total_dia['d-30'] = df_precipitacao_bacia_total_dia['precipitacao'].rolling(window=30).sum().shift(periods=1)

# Exiba o DataFrame resultante
print("DF: df_precipitacao_bacia_total_dia:")
display(df_precipitacao_bacia_total_dia)

# %%
# Estatísticas para a conferência
df_precipitacao_bacia_total_dia.info()

# %% [markdown]
# #### Concatenando os DataFrames com base na coluna 'data'

# %%
# Concatenando os DataFrames com base na coluna 'data'
df_dados_resumo = pd.concat([df_cs_max_cota_dia_periodo, df_pf_max_cota_dia_periodo ,df_mare_max_cota_dia_periodo, df_precipitacao_bacia_total_dia], ignore_index=True)

print("Exibindo o cabeçalho do dataframe resultante:")
# Primeiras 5 linhas
print("Primeiras 5 linhas:")
print(df_dados_resumo.head(100).to_markdown(index=False, numalign="left", stralign="left"))

# Últimas 5 linhas
print("\nÚltimas 5 linhas:")
print(df_dados_resumo.tail(100).to_markdown(index=False, numalign="left", stralign="left"))

# %% [markdown]
# #### Unindo os DataFrames usando o método merge

# %%
# Unindo os DataFrames usando o método merge
df_dados_resumo = df_cs_max_cota_dia_periodo.merge(df_pf_max_cota_dia_periodo, on='data', how='outer') \
    .merge(df_mare_max_cota_dia_periodo, on='data', how='outer') \
    .merge(df_precipitacao_bacia_total_dia, on='data', how='outer')

# Resetando o índice do DataFrame
df_dados_resumo = df_dados_resumo.reset_index(drop=True)

#df_dados_resumo
# Exibindo o cabeçalho do dataframe
# Primeiras 5 linhas
print("Primeiras 5 linhas:")
print(df_dados_resumo.head().to_markdown(index=False, numalign="left", stralign="left"))

# Últimas 5 linhas
print("\nÚltimas 5 linhas:")
print(df_dados_resumo.tail().to_markdown(index=False, numalign="left", stralign="left"))

# %% [markdown]
# #### Salvando Dataframes em um arquivo csv

# %%
# Salvando o DataFrame df_dados_resumo em um arquivo CSV
df_dados_resumo.to_csv(r'../output_data/database/dados_resumo.csv', index=False)

# %% [markdown]
# #### Debug de testes

# %%
# Checando os tipos de variáveis
x = pd.read_csv('../output_data/database/dados_resumo.csv')
x.info()

# %%
# Contagem de dados em cada coluna
contagem_por_coluna = df_dados_resumo.count()
print("Quantidade de dados em cada coluna:")
print(contagem_por_coluna)


# %%
# Inicial e final de cada coluna com dados
inicial_final_por_coluna = {}

# Iterar sobre cada coluna
for coluna in df_dados_resumo.columns:
    # Encontrar o índice do primeiro valor não nulo na coluna
    indice_inicial = df_dados_resumo[coluna].first_valid_index()
    
    # Encontrar o índice do último valor não nulo na coluna
    indice_final = df_dados_resumo[coluna].last_valid_index()
    
    # Adicionar os índices inicial e final à lista
    inicial_final_por_coluna[coluna] = (indice_inicial, indice_final)

# Exibir os resultados
print("Inicial e final de cada coluna com dados:")
for coluna, (indice_inicial, indice_final) in inicial_final_por_coluna.items():
    print(f"{coluna}: Inicial - {indice_inicial}, Final - {indice_final}")

# %%
# Inicial e final de cada coluna com dados
inicial_final_por_coluna = {}

# Iterar sobre cada coluna
for coluna in df_dados_resumo.columns:
    # Encontrar o índice do primeiro valor não nulo na coluna
    indice_inicial = df_dados_resumo[coluna].first_valid_index()
    # Converter o índice inicial para o formato de data
    data_inicial = df_dados_resumo.loc[indice_inicial, 'data'].strftime('%d/%m/%Y') if indice_inicial is not None else None
    
    # Encontrar o índice do último valor não nulo na coluna
    indice_final = df_dados_resumo[coluna].last_valid_index()
    # Converter o índice final para o formato de data
    data_final = df_dados_resumo.loc[indice_final, 'data'].strftime('%d/%m/%Y') if indice_final is not None else None
    
    # Adicionar as datas inicial e final à lista
    inicial_final_por_coluna[coluna] = (data_inicial, data_final)

# Exibir os resultados
print("Inicial e final de cada coluna com dados:")
for coluna, (data_inicial, data_final) in inicial_final_por_coluna.items():

    display(f"{coluna}: Inicial - {data_inicial}, Final - {data_final}")

# %% [markdown]
# ***
# ***
# ***
# #### Plotando os Mapas com as coordenadas
# ***
# ***
# ***

# %% [markdown]
# #### Abre o Shapefile Brasil Chama coordenadas no estado do NETCDF e plota ambos:

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import itertools
import pandas as pd

# Lendo o shapefile do Brasil
br = gpd.read_file(r'../input_data/shape_files/BR_UF_2022.shp')

# Gerando todas as combinações de latitudes e longitudes
combinacoes = list(itertools.product(latitudes, longitudes))

# Criando um DataFrame com as combinações de latitudes e longitudes
df_pontos = pd.DataFrame(combinacoes, columns=['lon', 'lat'])

# Criando o GeoDataFrame com os pontos
geometry = [Point(xy) for xy in zip(df_pontos['lat'], df_pontos['lon'])]
vcm1 = gpd.GeoDataFrame(df_pontos, geometry=geometry, crs="EPSG:4674")

# Plotando o mapa
fig, ax = plt.subplots(figsize = (15, 12))
coord_atlantico = [(-90, -40),(-30, -40),
	               (-30, 10),(-90, 10)]
atlantico_poly = Polygon(coord_atlantico)
atlantico = gpd.GeoDataFrame(geometry = [atlantico_poly])
atlantico.plot(ax = ax, color = "lightblue") # atlantico ~ base
ax.set_aspect("auto")

# Adicionando o shapefile do Brasil ao plot
br.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plotando os pontos
vcm1.plot(ax=ax, color='red', markersize=0.5)


# Configurando o título e os rótulos dos eixos
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Pontos sobre o Mapa do Brasil')
plt.xlim(-55, -47)
plt.ylim(-30, -25)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"VCM1")
plt.grid(True)
plt.savefig('../output_data/images/GraficoPontosSobreoMapadoBrasil.png', dpi=900, bbox_inches='tight')

# Exibindo o gráfico
plt.show()

# %% [markdown]
# #### Cria filtro sobre pontos de grade no estado

# %%
import shapefile
from shapely.geometry import Point, shape
        
# Carregar o arquivo KMZ da bacia hidrográfica

bacia_hidrografica = shapefile.Reader(r'../input_data/shape_files/bacia_hidro_SC/bacia_hidro_SC.shp')
# Extrair os limites da bacia hidrográfica
poligonos = []
for shape_rec in bacia_hidrografica.shapeRecords():
    poligonos.append(shape(shape_rec.shape))

# Lista de coordenadas (latitude, longitude)
coordenadas = combinacoes

# Filtrar as coordenadas para manter apenas aquelas dentro da bacia hidrográfica
coordenadas_filtradas = []
for lat, lon in coordenadas:
    ponto = Point(lon, lat)
    dentro_bacia = False
    for poligono in poligonos:
        if ponto.within(poligono):
            dentro_bacia = True
            break
    if dentro_bacia:
        coordenadas_filtradas.append((lat, lon))

# coordenadas_filtradas agora contém apenas as coordenadas que estão dentro da bacia hidrográfica
print(coordenadas_filtradas)

# %% [markdown]
# #### Plota os novos pontos de grade após filtro:

# %%

# coordenadas_filtradas = Lon_-48,95 Lat_-27,75	Lon_-48,95 Lat_-27,65	Lon_-48,85 Lat_-27,85	Lon_-48,85 Lat_-27,75	Lon_-48,85 Lat_-27,65	Lon_-48,75 Lat_-27,75	Lon_-48,75 Lat_-27,65	Lon_-48,65 Lat_-27,55	Lon_-48,55 Lat_-27,75	Lon_-48,55 Lat_-27,65	Lon_-48,45 Lat_-27,55	Lon_-48,45 Lat_-27,45

# Gerando todas as combinações de latitudes e longitudes
combinacoes = coordenadas_filtradas

# Criando um DataFrame com as combinações de latitudes e longitudes
df_pontos = pd.DataFrame(combinacoes, columns=['lon', 'lat'])

# Criando o GeoDataFrame com os pontos
geometry = [Point(xy) for xy in zip(df_pontos['lat'], df_pontos['lon'])]
vcm1 = gpd.GeoDataFrame(df_pontos, geometry=geometry, crs="EPSG:4674")

# Plotando o mapa
fig, ax = plt.subplots(figsize = (15, 12))
coord_atlantico = [(-90, -40),(-30, -40),
	               (-30, 10),(-90, 10)]
atlantico_poly = Polygon(coord_atlantico)
atlantico = gpd.GeoDataFrame(geometry = [atlantico_poly])
atlantico.plot(ax = ax, color = "lightblue") # atlantico ~ base
ax.set_aspect("auto")

# Adicionando o shapefile do Brasil ao plot
br.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plotando os pontos
vcm1.plot(ax=ax, color='red', markersize=1)


# Configurando o título e os rótulos dos eixos
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Pontos sobre o Mapa do Brasil')
plt.xlim(-55, -47)
plt.ylim(-30, -25)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"VCM1")
plt.grid(True)

#Salvando o mapa:
plt.savefig('../output_data/images/Pontos_Bacia_hidro_SC.png', dpi=900, bbox_inches='tight')

# Exibindo o gráfico
plt.show()




# %% [markdown]
# #### Bacia Rio Cubatão do Sul (pontos)

# %%

# coordenadas_filtradas = [(-27.75, -48.95), (-27.65, -48.95), (-27.85, -48.85), (-27.75, -48.85), (-27.65, -48.85), (-27.75, -48.75), (-27.65, -48.75), (-27.55, -48.65),(-27.65, -48.55), (-27.55, -48.45), (-27.45, -48.45)]
coordenadas_filtradas = [(-27.75, -48.95), (-27.65, -48.95), (-27.85, -48.85),(-27.65, -48.85), (-27.75, -48.75), (-27.65, -48.75), (-27.55, -48.65),(-27.65, -48.55), (-27.55, -48.45), (-27.45, -48.45)]

# Gerando todas as combinações de latitudes e longitudes

combinacoes = coordenadas_filtradas

# Criando um DataFrame com as combinações de latitudes e longitudes
df_pontos = pd.DataFrame(combinacoes, columns=['lon', 'lat'])

# Criando o GeoDataFrame com os pontos
geometry = [Point(xy) for xy in zip(df_pontos['lat'], df_pontos['lon'])]
vcm1 = gpd.GeoDataFrame(df_pontos, geometry=geometry, crs="EPSG:4674")

# Plotando o mapa
fig, ax = plt.subplots(figsize = (15, 12))
coord_atlantico = [(-90, -40),(-30, -40),
	               (-30, 10),(-90, 10)]
atlantico_poly = Polygon(coord_atlantico)
atlantico = gpd.GeoDataFrame(geometry = [atlantico_poly])
atlantico.plot(ax = ax, color = "lightblue") # atlantico ~ base
ax.set_aspect("auto")

# Adicionando o shapefile do Brasil ao plot
br.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plotando os pontos
vcm1.plot(ax=ax, color='red', markersize=10) # 5 é o tamanho dos dots das coordenadas


# Configurando o título e os rótulos dos eixos
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.title('Pontos da Bacia do Rio Cubatão do Sul')
plt.xlim(-55, -47)
plt.ylim(-30, -25)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Pontos da Bacia do Rio Cubatão do Sul")
plt.grid(True)

# Salvando o gráfico
plt.savefig('../output_data/images/Pontos_Bacia_hidro_Rio_Cubatao_do_Sul.png', dpi=900, bbox_inches='tight')

# Exibindo o gráfico
plt.show()




# %% [markdown]
# ### Mostrando as coordenadas filtradas

# %%
# Obtenha as colunas como uma lista
colunas = dfEmpilhado.columns.tolist()

# Crie um DataFrame com as colunas
df_colunas = pd.DataFrame({'Colunas': colunas})

# Salve as colunas em um arquivo CSV
df_colunas.to_csv('../output_data/database/colunas_dfEmpilhado.csv', index=False)

# Salvando o dafaframe num arquivo csv
dfEmpilhado.to_csv('../output_data/database/precipitacao_Empilhadoc.csv')


