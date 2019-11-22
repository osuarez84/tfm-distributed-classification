from aux_func import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
import sys
import time

# Read the datasets
# df_original_train = pd.read_csv('../01_datasets/Datasets_Omar/Reales/connect-4Train.csv',
#                                 sep=',', header=None)
# df_original_test = pd.read_csv('../01_datasets/Datasets_Omar/Reales/connect-4Test.csv',
#                                sep=',', header=None)

# List of nodes to test
n = [2]
# Number of samples per node
m = 500


df_original = pd.read_csv('../01_datasets/Datasets_Omar/Reales/spambase.data',
                          sep=',', header=None)
df_original_train = df_original.sample(frac=0.7)
df_original_test = df_original.drop(df_original_train.index)
df_original_test = df_original_test.sample(n=m)
print(df_original_test.shape)

#print(df_original_train.info())
#print(df_original_test.info())

# Initilize the list of partitioned nodes
l_df_balanced_partitioned_nodes = []
l_df_unbalanced_partitioned_nodes = []

# Init the centralized DataFrames
df_centralized_balanced = pd.DataFrame([])
df_centralized_unbalanced = pd.DataFrame([])

for number_of_nodes in n:
    # Get list of balanced partitioned nodes
    start_balanced_partition = time.time()
    l_df_balanced_partitioned_nodes = create_balanced_partitions(
        df_original_train,
        number_of_nodes,
        m
    )
    end_balanced_partition = time.time()
    print(f'Elapsed time in balanced partition :{end_balanced_partition - start_balanced_partition}.')

    # Get list of unbalanced partitioned nodes
    print(f'Number of nodes in the balanced list: {len(l_df_balanced_partitioned_nodes)}')

    # l_df_unbalanced_partitioned_nodes = unbalanced_dataset_generation(
    #     df_original_train,
    #     number_of_nodes,
    #     m
    # )
    # print(f'Number of nodes in the unbalanced list: {len(l_df_unbalanced_partitioned_nodes)}')

    # Compute Energy Statistic for each node
    ###############
    # CASE BALANCED
    ###############
    # Prepare the list with Energy Statistic info
    col_names = ['Node', 'Index', 'Energy_statistic']
    df_dist_balanced_list_all_nodes = pd.DataFrame([], columns=col_names)

    start_balanced_enegery_distance_computing = time.time()
    # Iterate over each node of the list
    for node in l_df_balanced_partitioned_nodes:
        # Iterate over all the rows in the DataFrame of the node
        print('Computing Energy Statistic for balanced nodes...')
        for index, row in node.iterrows():
            e = energy_statistic_b(row, df_original_test)
            #print(e)
            df_dist_balanced_list_all_nodes = df_dist_balanced_list_all_nodes.append(pd.DataFrame([[0, index, e]], columns=col_names), ignore_index=True)
        
    # Sort the list using the Energy Statistic distance and taking into account all the nodes
    df_dist_balanced_list_all_nodes = df_dist_balanced_list_all_nodes.sort_values(by=['Energy_statistic'], ascending=True)
    print(df_dist_balanced_list_all_nodes.head())
    # Take only the m better observations order by Energy Statistic
    df_training_balanced_all_nodes = df_dist_balanced_list_all_nodes.iloc[:m,:]

    df_training_distributed_balanced_final_node = pd.DataFrame([])
    for i in range(0, number_of_nodes):
        df_training_distributed_balanced_final_node = df_training_distributed_balanced_final_node.append(l_df_balanced_partitioned_nodes[i].iloc[df_training_balanced_all_nodes[df_training_balanced_all_nodes['Node'] == i].Index], ignore_index=True)

    end_balanced_energy_distance_computing = time.time()
    print(f'Elapsed time in computing energy distance in balanced partitions: {end_balanced_energy_distance_computing - start_balanced_enegery_distance_computing}')

    # #################
    # # CASE UNBALANCED
    # #################
    # # TODO
    # df_dist_unbalanced_list_all_nodes = pd.DataFrame([], columns=col_names)
    # for node in l_df_unbalanced_partitioned_nodes:
    #     print('Computing Energy Statistic for unbalanced nodes...')
    #     for index, row in node.iterrows():
    #         e = energy_statistic_b(row, df_original_test)
    #         print(e)
    #         df_dist_unbalanced_list_all_nodes = df_dist_unbalanced_list_all_nodes.append(pd.DataFrame([[0, index, e]], columns=col_names), ignore_index=True)


    # # Sort the list using the Energy Statistic distance and taking into account all the nodes
    # df_dist_unbalanced_list_all_nodes = df_dist_unbalanced_list_all_nodes.sort_values(by=['Energy_statistic'], ascending=True)
    # print(df_dist_unbalanced_list_all_nodes.head())
    # # Take only the m better observations order by Energy Statistic
    # df_training_unbalanced_all_nodes = df_dist_unbalanced_list_all_nodes.iloc[:m,:]

    # df_training_distributed_unbalanced_final_node = pd.DataFrame([])
    # for i in range(0, number_of_nodes):
    #     df_training_distributed_unbalanced_final_node = df_training_distributed_unbalanced_final_node.append(l_df_balanced_partitioned_nodes[i].iloc[df_training_balanced_all_nodes[df_training_balanced_all_nodes['Node'] == i].Index], ignore_index=True)


    ############################################################################
    # Train the classifier with partitions
    ############################################################################
    clf = RandomForestClassifier()

    #################################
    # TRAINING BALANCED DISTRIBUTIONS
    #################################
    X_train_balanced = df_training_distributed_balanced_final_node.iloc[:,:-1]
    y_train_balanced = df_training_distributed_balanced_final_node.iloc[:, -1]

    # Entrenamos el modelo
    start_train_balanced = time.time()
    clf.fit(X_train_balanced, y_train_balanced)
    end_train_balanced = time.time()
    print(f'Elapsed time in training for balanced partition: {end_train_balanced - start_train_balanced}')

    # Hacemos las predicciones sobre el test
    X_test = df_original_test.iloc[:,:-1]
    y_test = df_original_test.iloc[:, -1]

    y_pred_balanced = clf.predict(X_test)

    recall_distributed_balanced = recall_score(y_test, y_pred_balanced, average='macro')
    precision_distributed_balanced = precision_score(y_test, y_pred_balanced, average='macro')

    print(f'The recall for {number_of_nodes} nodes in balanced partitioned is {recall_distributed_balanced}.')
    print(f'The precision for {number_of_nodes} nodes in balanced partitioned is {precision_distributed_balanced}.')

    # ###################################
    # # TRAINING UNBALANCED DISTRIBUTIONS
    # ###################################
    # X_train_unbalanced = df_training_distributed_unbalanced_final_node.iloc[:,:-1]
    # y_train_unbalanced = df_training_distributed_unbalanced_final_node.iloc[:, -1]

    # # Entrenamos el modelo
    # clf.fit(X_train_unbalanced, y_train_unbalanced)

    # # Hacemos predicciones sobre el test
    # y_pred_unbalanced = clf.predict(X_test)

    # recall_distributed_unbalanced = recall_score(y_test, y_pred_unbalanced, average='macro')
    # precision_distributed_unbalanced = precision_score(y_test, y_pred_unbalanced, average='macro')

    # print(f'The recall for {number_of_nodes} nodes in unbalanced partitioned is {recall_distributed_unbalanced}.')
    # print(f'The precision for {number_of_nodes} nodes in unbalanced partitioned is {precision_distributed_unbalanced}.')


    ###############################################
    # Train the classifier with centralized dataset
    ###############################################
    # TODO
    # Parametrizar el acceso a las etiquetas
    X_train_centralized = pd.read_csv('sampled_centralized_balanced.csv', sep=',', header=0).iloc[:,:-1]
    y_train_centralized = pd.read_csv('sampled_centralized_balanced.csv', sep=',', header=0).iloc[:, -1]
    
    # Entrenamos el modelo
    start_train_centralized = time.time()
    clf.fit(X_train_centralized, y_train_centralized)
    end_train_centralized = time.time()
    print(f'Elapsed time training in centralized approach: {end_train_centralized - start_train_centralized}')

    # Hacemos las predicciones sobre el test
    y_pred_centralized = clf.predict(X_test)

    recall_centralized_balanced = recall_score(y_test, y_pred_centralized, average='macro')
    precision_centralized_balanced = precision_score(y_test, y_pred_centralized, average='macro')

    print(f'The recall for {number_of_nodes} nodes in balanced centralized is {recall_centralized_balanced}.')
    print(f'The precision for {number_of_nodes} nodes in balanced centralized is {precision_centralized_balanced}.')





