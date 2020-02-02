from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import random
import math
import operator

################################################################################
# BALANCED PARTITIONS
################################################################################

def create_balanced_dataset(df_o, m):
    print('###########################')
    print('Generating balanced dataset...')
    print('###########################')

    # Muestreo del dataset original para quedarnos con el sampleado
    # manteniendo la proporcion de clases original
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m)

    for train_index, test_index in sss.split(df_o.iloc[:, 0:-1], df_o.iloc[:, -1]):
        df  = df_o.iloc[train_index]
        print(df.shape)
        print(df.iloc[:,-1].value_counts(normalize=True) * 100)

    return df


def create_balanced_partitions(df, partitions):
    print('######################################')
    print('GENERATING THE BALANCED NODES...')
    print('######################################')

    # how many classes
    # IMPORTANT
    # The class should be in the last column
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

    #for i, df in zip(range(0,m), l):
    #  df.to_csv('node_' + str(i) + '_distributed_balanced.csv', sep=',', header=True, index=False)
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


def create_unbalanced_partitions(df, partitions):
    print('######################################')
    print('GENERATING THE UNBALANCED NODES...')
    print('######################################')
    
    n = list(df.iloc[:,-1].unique())

    l_df = [
        df.loc[df.iloc[:,-1] == x] for x in n
    ]

    l_df_obs = [
                len(x) for x in l_df
    ]

    print(f'Observaciones iniciales del dataset: {l_df_obs}')
    print(df.iloc[:,-1].value_counts())

    l_total_nodes_instances = []

    for i in range(0, partitions):
        # Cantidad de instancias que restan
        l_df_node = []
        remaining_c = 0
        l_remaining_c = []
        for j in l_df:
            remaining_c += len(j)
            l_remaining_c.append(len(j))
        print(f'Cantidad de muestras de cada clase que restan: {l_remaining_c}')
        print(f'Cantidad de muestas totales que restan {remaining_c}')


        while (True):
            list_calc = []
            if (i == 0):
                obs =  list(map(lambda x: math.floor(x / partitions), l_df_obs))
            else:
                if (partitions == 2):
                    obs = list(map(lambda x: math.floor(x / partitions), l_df_obs))
                    break
                else:
                    obs = list(map(lambda x: math.floor(x / partitions), l_df_obs))
                    while(True):
                        r = [random.random() for i in range(0, len(n))]
                        s = sum(r)
                        r = [ i/s for i in r ]
                        r_aux = [math.floor(sum(obs)*j) for j in r]
                        if(all((item >= 0.1 and item <= 0.7) for item in r)):
                            #print('This is perfect!')
                            #print(r)
                            #print(r_aux)
                            obs = r_aux
                            break
        
            # Comprobamos que la seleccion de observaciones computada no supera
            # las observaciones disponibles para cada clase, daria un error si no
            list_calc = list(map(operator.sub, l_remaining_c, obs))
            if (all((item > 0) for item in list_calc)):
                # Si no las supera todo esta correcto, salimos del bucle para
                # la siguiente iteracion
                break

        print(f'Observaciones para nodo {i} = {obs}. Total = {sum(obs)}')  
        for df_c, df_c_obs in zip(l_df, obs):
            print(f'Observaciones para clase {df_c.iloc[:,-1].unique()} = {df_c_obs}')


        l_df_node = [x.sample(n=y) for x, y in zip(l_df, obs)]
        
        l_df = [x.drop(y.index) for x, y in zip(l_df, l_df_node)]

        # Cada elemento de l_total_nodes_instances contiene las muestras de cada nodo
        l_total_nodes_instances.append(l_df_node)
        print(len(l_total_nodes_instances))



    # Convert the list of lists to a list of DataFrames, each DF contain 
    # all instances of one node.
    l_df_total_nodes_instances = []
    for l in l_total_nodes_instances:
        tmp_df = pd.DataFrame([])
        for df in l:
            tmp_df = tmp_df.append(df)

        l_df_total_nodes_instances.append(tmp_df)

    for i in l_df:
        print(f'Observaciones que quedan sin repartir para la clase {i.iloc[:,-1].unique()} = {len(i)}')

    # Convert to a csv files
    df.to_csv('sampled_centralized_unbalanced.csv', sep=',', header=True, index=False)
    
    # Se devuelve:
    # l_df_total_nodes_instances (particiones de nodos)
    # df (nodo centralizado)

    return l_df_total_nodes_instances


################################################################################
# ENERGY STATISTIC
################################################################################
# Funcion para el computo de la distancia entre dos distribuciones
def distance_computing(a,b):
  step_5 = 0
  for index, row in a.iterrows():
    np_arr = np.full((b.shape[0], b.shape[1]), row.to_numpy())
    step_1 = np_arr - b.to_numpy()
    step_2 = np.square(step_1)
    step_3 = np.sum(step_2, axis=1)
    step_4 = np.sqrt(step_3)
    step_5 += np.sum(step_4)

  step_6 = (1 / (a.shape[0] * b.shape[0])) * step_5
  return step_6



def distance_computing_b(a,b):
  np_arr = np.full((b.shape[0], b.shape[1]), a.to_numpy())

  step_1 = np_arr - b.to_numpy()
  step_2 = np.square(step_1)
  step_3 = np.sum(step_2, axis=1)
  step_4 = np.sqrt(step_3)
  step_5 = np.sum(step_4)
  step_6 = (1 / b.shape[0]) * step_5
  return step_6


# Calculo entre una observacion y otra observacion
def distance_computing_c(a, b):
    l2_norm = np.sqrt(np.square(a.values - b.values).sum())
    d = (1 / 1) * l2_norm
    return d

# Funcion para el computo de la energy statistic entre dos distribuciones
def energy_statistic_b(row_a, sample_b):
    # row_a es una observación del conjunto de train
    # sample_b es el conjunto completo de test
    energy = 2*distance_computing_b(row_a, sample_b) - distance_computing_c(row_a, row_a) - distance_computing(sample_b, sample_b)
    return energy