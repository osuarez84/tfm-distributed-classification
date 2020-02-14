from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import random
import math
import operator
import sys

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

################################################################################
# BALANCED PARTITIONS
################################################################################



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
    l_x = [i.iloc[:,:-1] for i in l]
    l_y = [i.iloc[:,-1] for i in l]
    return l_x, l_y


################################################################################
# UNBALANCED PARTITIONS
################################################################################
# Funcion para generar dataset de sample desbalanceado y particiones

def create_unbalanced_partitions(df, nodes):
    print('######################################')
    print('GENERATING THE UNBALANCED NODES...')
    print('######################################')
    n = list(df.iloc[:,-1].unique())
    C = len(n)
    remaining_classes = df.iloc[:,-1]
    remaining_set = df

    partitions = []


    for i in range(1, nodes+1):
        N = len(remaining_classes)
        print(f'Remaining classes: {N}')

        P = nodes - i + 1
        print(f'Partitions: {P}')

        prop = remaining_classes.value_counts() / N

        print(f'Proportions of classes: \n{prop}')

        dev = prop * np.random.uniform(0.1, 1.9, len(n))
        print(f'Dev: \n{dev}')

        dev = dev / sum(dev)
        print(f'Normalized proportions: \n{dev}')

        # Some partitions have the same proportion as the original
        if(i == 1):
            dev = prop

        print(f'Final proportions for node {i}: \n{dev}')

        observations = pd.Series(list(map(lambda x: math.floor(x), dev.to_numpy() * (N/P))), index=dev.index.tolist())
        print(f'Observations for node {i}: \n{observations}')

        print('#####################################')
        print(f'Computing partition for node {i}...')
        print('#####################################')
        partition_df = pd.DataFrame([])

        if(i != nodes):
            for j in n:
                # Indexes for j class
                rem = remaining_classes[remaining_classes == j].index.tolist()

                if(rem == 0):
                    raise Exception(f'Error no elements of class {C}')
                print(rem)

                nobs = observations[j]
                print(f'Number of observations for class {j}: \n{nobs}')

                # At least one observation per class
                if(nobs == 0):
                    nobs = 1

                nremclass = len(rem)
                print(f'Number remaining instances for class {j}: {nremclass}')

                nobs = min(nobs, nremclass)
                print(f'Number of final observations: {nobs}')

                selectedobs = random.sample(rem, nobs)
                print(f'Selected observations (indexes) from the remaninig: {selectedobs}')

                partition_df = partition_df.append(remaining_set.loc[selectedobs])
                
                # Drop the selected observations
                remaining_classes = remaining_classes.drop(selectedobs)
                remaining_set = remaining_set.drop(selectedobs)
                print(f'Remaining instances: {remaining_classes.shape}')
            
            print(f'Number of instances for node {i}: {partition_df.shape}')
            print(f'Remaining instances for each class: \n{remaining_classes.value_counts()}')
            partitions.append(partition_df)
            print(f'Partition for node {i}: \n{partitions[i-1].head()}')
        else:
            # Check at least one class
            if(all(n_instances > 0 for n_instances in remaining_classes.value_counts().tolist())):
                partition_df = partition_df.append(remaining_set)
                partitions.append(partition_df)
                print(f'Number of instances for node {i}: {partition_df.shape}')
                print(f'Classes for node {i}: \n{partition_df.iloc[:,-1].value_counts()}')
            else:
                print(f'Remaining classes for the last node: \n{remaining_classes.value_counts()}')
                #raise Exception('There are not enough instances of each class for the last node!')


    partitions_X = [i.iloc[:,:-1] for i in partitions]
    partitions_Y = [i.iloc[:,-1] for i in partitions]

    print(f'Number of partitions en the list: {len(partitions_X)}')
    for i in partitions_X:
        print(i.shape) 
    return partitions_X, partitions_Y  



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