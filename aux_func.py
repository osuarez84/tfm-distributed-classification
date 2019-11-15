from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import random
import math

################################################################################
# BALANCED PARTITIONS
################################################################################
def create_balanced_partitions(df_o, partitions, m):
    print('######################################')
    print('GENERATING THE BALANCED NODES...')
    print('######################################')

    # Muestreo del dataset original para quedarnos con el sampleado
    # manteniendo la proporcion de clases original
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m)

    for train_index, test_index in sss.split(df_o.iloc[:, 0:-1], df_o.iloc[:, -1]):
        df  = df_o.iloc[train_index]
        print(df.shape)
        print(df.iloc[:,-1].value_counts(normalize=True) * 100)

    # how many classes
    # IMPORTANT
    # The class should be in the last column
    #df = df_o.copy()
    n = list(df.iloc[:,-1].unique())
    print(f'The classes in the dataset are: {n}')
    #n_classes = df.groupby(df.columns[-1]).size().shape[0]
    
    # Extract the observations based on the class 
    l_df = [
        df.loc[df.iloc[:,-1] == x] for x in n
    ]
    
    # shuffle the dataframe rows for each class so every time the functon is used
    # the results are different
    l_df = [
        df.sample(frac=1, axis=0).reset_index(drop=True) for df in l_df
    ]
    
    l = [pd.DataFrame([]) for x in range(0, partitions)]
    print(len(l))

    for pdf in l_df:
        # Calculate the number of rows we need to extract for each partiton
        # based on the number of rows for each class form the original df
        n_rows = math.floor(pdf.shape[0] / partitions)
        print(f'Number of observations for the class: {n_rows}')
        
        # modify each Dataframe adding rows from each class
        # After add those rows, the rows are deleted from the original
        # dataframes so we don't use the same rows twice
        for i in range(0, len(l)):
            l[i] = l[i].append(pdf.iloc[0:n_rows, :], ignore_index=True)
            pdf = pdf.iloc[n_rows:]
            print(l[i].shape)
                
        print(f'Size of the class bucket after repartition to the nodes: {pdf.shape}')

    # Convert to a csv files
    df.to_csv('sampled_centralized_balanced.csv', sep=',', header=True, index=False)

    for i, df in zip(range(0,m), l):
      df.to_csv('node_' + str(i) + '_distributed_balanced.csv', sep=',', header=True, index=False)
    return l


################################################################################
# UNBALANCED PARTITIONS
################################################################################
# Funcion para generar dataset de sample desbalanceado y particiones
def unbalanced_dataset_generation(df_training, nodes, m):
    """
    df_training (DataFrame): dataframe con los datos de training completos
    nodes (int): numero de nodos 
    m (int): muestra totales a utilizar del dataset de training completo
    """
    print('######################################')
    print('GENERATING THE UNBALANCED NODES...')
    print('######################################')
    # Recibimos el dataset de training completo
    # Sacamos las clases a DF individuales
    # Clases unicas
    n = list(df_training.iloc[:,-1].unique())

    l_df = [
        df_training.loc[df_training.iloc[:,-1] == x] for x in n
    ]

    # Numero de muestras por nodo
    m_por_nodo = math.floor(m / nodes)

    # Misma distribucion de clases que el dataset original
    for c in l_df:
        print(f'La clase {c.iloc[0,-1]} tiene un porcentaje del {(c.shape[0] / df_training.shape[0]) * 100} dentro del dataset inicial, que son {math.floor(m_por_nodo * (c.shape[0] / df_training.shape[0]))}')
    
    l_df_nodes = []
    df_completo = pd.DataFrame([])
    # Para cada nodo...
    for i in range(0,nodes):
        # generamos una lista con la cantidad de instancias a samplear de cada clase
        df_node_n = pd.DataFrame([])

        # Repetir seleccion mientras haya 0 muestras para alguna clase en 
        # la reparticion entre todos los nodos
        if(i != 0):
            while(True):
                r = [random.random() for i in range(0, len(n))]
                s = sum(r)
                r = [ i/s for i in r ]
                r_aux = [math.floor(m_por_nodo*j) for j in r]
                if(not 0 in r_aux):
                    print('There are no 0s so it is good distribution!')
                    print(r_aux)
                    break
                print('There are 0s! Need to repeat!')
                print(r_aux)

        for c, j in zip(l_df, range(0,len(n))):
            print(f'Preparing node {i}...')
            if(i == 0): # If Node 0...
                l_instances = math.floor(m_por_nodo * (c.shape[0] / df_training.shape[0]))
            else:
                l_instances = math.floor(m_por_nodo * r[j])

            print(l_instances)

            node_n = c.sample(n=l_instances, replace=False)#, random_state=1)
            df_node_n = df_node_n.append(node_n, ignore_index=True)
            index_node_n = node_n.index.values.tolist()
            print(index_node_n)
            c = c.drop(index_node_n, axis=0)
            print(f'Instances added to Node {i} = {l_instances}')
            
            df_node_n.to_csv('nodo_' + str(i) + '_distributed_unbalanced.csv', sep=',', header=True, index=False)
            df_completo = df_completo.append(df_node_n, ignore_index=True)
        
        l_df_nodes.append(df_node_n)
        print('###########')
        print(f'Node {i} has been prepared with {df_node_n.shape[0]} instances.')
        print('###########')

    # Una vez tenemos todos los nodos distribuidos hacemos un append
    # de todos ellos para formar el nodo de training (en este caso desbalanceado)
    # lo hacemos asi para facilitar la generacion de los particionados 
    # (primero creamos particiones, despues las juntamos y creamos el completo)
    df_completo.to_csv('sampled_centralized_unbalanced.csv', sep=',', header=True, index=False)
    return l_df_nodes


################################################################################
# ENERGY STATISTIC
################################################################################
# Funcion para el computo de la distancia entre dos distribuciones
def distance_computing(a, b):
    # Extract the values from each row to compute
    el = 0
    for index_a, row_a in a.iterrows():
        for index_b, row_b in b.iterrows():
            l2_norm = np.sqrt(np.square(row_a.to_numpy() - row_b.to_numpy()).sum())
            el += l2_norm
    d = (1 / (a.shape[0] * b.shape[0])) * el
    return d

# Calculo entre una observacion y otra observacion
def distance_computing_c(a, b):
    l2_norm = np.sqrt(np.square(a.values - b.values).sum())
    d = (1 / 1) * l2_norm
    return d


# Funcion para el computo de la distancia entre una observacion y un conjunto de datos
def distance_computing_b(a, b):
    # a is a row
    # b is an entire dataset
    #
    # Extract the values from each row to compute
    el = 0
    for index_b, row_b in b.iterrows():
        #print(a)
        #print(row_b)
        l2_norm = np.sqrt(np.square(a.to_numpy() - row_b.to_numpy()).sum())
        el += l2_norm
    d = (1 / (b.shape[0])) * el
    return d

# La funcion para el computo de la distancia entre dos conjuntos de datos esta
# especificada en el primer apartado => distance_computing

# Funcion para el computo de la energy statistic entre dos distribuciones
def energy_statistic_b(row_a, sample_b):
    # row_a es una observación del conjunto de train
    # sample_b es el conjunto completo de test
    energy = 2*distance_computing_b(row_a, sample_b) - distance_computing_c(row_a, row_a) - distance_computing(sample_b, sample_b)
    return energy