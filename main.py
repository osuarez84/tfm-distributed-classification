from aux_func import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
import xgboost as xgb
import sys
import time

# Read the datasets
# df_original_train = pd.read_csv('../01_datasets/Datasets_Omar/Reales/connect-4Train.csv',
#                                 sep=',', header=None)
# df_original_test = pd.read_csv('../01_datasets/Datasets_Omar/Reales/connect-4Test.csv',
#                                sep=',', header=None)

##################
# GLOBAL VARIABLES
##################
# List of nodes to test
n = [2, 4, 7, 11]
# Number of samples per node
m = 200
# Number of executions per dataset
nexec = 5
is_balanced = True

#########################
# Setting the classifiers
#########################
# TODO
# XGBOOST classifier
clf_rf = RandomForestClassifier()
clf_svm = svm.SVC(kernel='rbf', gamma='auto')
clf_lda = LDA()
clf_mlr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
clf_xgb = xgb.XGBClassifier()

list_classifiers = [
    clf_rf,
    clf_svm,
    clf_lda,
    clf_mlr,
    clf_xgb
]

list_classifiers_names = [
    'Random Forest',
    'SVM',
    'LDA',
    'Multinomial Logistic Regression',
    'XGBoost'
]

######################
# READING DATASETS
#####################
# Reading the dataset
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

# Preparing the DataFrame with all the info from the experiment
info_col_names = [
    'n_nodes',
    'n_exec',
    'classifier',
    'dataset',
    'recall',
    'precision',
    'time_training',
    'time_energy_distance'
]

info_col_names_central = [
    'n_exec',
    'classifier',
    'dataset',
    'recall',
    'precision',
    'time_training'
]

df_exp_info_balanced = pd.DataFrame(
    [],
    columns=info_col_names
)

df_exp_info_central_balanced = pd.DataFrame(
    [],
    columns=info_col_names_central
)

df_exp_info_unbalanced = pd.DataFrame(
    [],
    columns=info_col_names
)

df_exp_info_central_unbalanced = pd.DataFrame(
    [],
    columns=info_col_names_central
)

#########################################
# BEGINING THE EXPERIMENT
#########################################
for n_exec in range(0, nexec):
    print(f'Execution number: {n_exec}.\n')
    # Iterate computation for each number of partitions in the list

    # Prepare the datast for the execution
    # BALANCED DATASET
    balanced_dataset = create_balanced_dataset(
        df_original_train,
        m
    )

    # UNBALANCED DATASET
    # TODO


    for number_of_nodes in n:
        print(f'Preparing experiment for {number_of_nodes} nodes...\n')
        if (is_balanced):
            print('Balanced partition...')
            # Get list of balanced partitioned nodes
            start_balanced_partition = time.time()
            l_df_balanced_partitioned_nodes = create_balanced_partitions(
                balanced_dataset,
                number_of_nodes
            )
            end_balanced_partition = time.time()
            print(f'Elapsed time in balanced partition : \
                {end_balanced_partition - start_balanced_partition}.')

            # Get list of unbalanced partitioned nodes
            print(f'Number of nodes in the balanced list: {len(l_df_balanced_partitioned_nodes)}')
        else:
            print('Unbalanced partition...')
            l_df_unbalanced_partitioned_nodes = unbalanced_dataset_generation(
                df_original_train,
                number_of_nodes,
                m
            )
            print(f'Number of nodes in the unbalanced list: {len(l_df_unbalanced_partitioned_nodes)}')

        ############################################################################
        # COMPUTING ENERGY DISTANCE
        ############################################################################
        
        ###############
        # CASE BALANCED
        ###############
        if (is_balanced):
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

        #################
        # CASE UNBALANCED
        #################
        else:
            df_dist_unbalanced_list_all_nodes = pd.DataFrame([], columns=col_names)
            for node in l_df_unbalanced_partitioned_nodes:
                print('Computing Energy Statistic for unbalanced nodes...')
                for index, row in node.iterrows():
                    e = energy_statistic_b(row, df_original_test)
                    #print(e)
                    df_dist_unbalanced_list_all_nodes = df_dist_unbalanced_list_all_nodes.append(pd.DataFrame([[0, index, e]], columns=col_names), ignore_index=True)


            # Sort the list using the Energy Statistic distance and taking into account all the nodes
            df_dist_unbalanced_list_all_nodes = df_dist_unbalanced_list_all_nodes.sort_values(by=['Energy_statistic'], ascending=True)
            print(df_dist_unbalanced_list_all_nodes.head())
            # Take only the m better observations order by Energy Statistic
            df_training_unbalanced_all_nodes = df_dist_unbalanced_list_all_nodes.iloc[:m,:]

            df_training_distributed_unbalanced_final_node = pd.DataFrame([])
            for i in range(0, number_of_nodes):
                df_training_distributed_unbalanced_final_node = df_training_distributed_unbalanced_final_node.append(l_df_balanced_partitioned_nodes[i].iloc[df_training_balanced_all_nodes[df_training_balanced_all_nodes['Node'] == i].Index], ignore_index=True)



        #################################
        # PREPARE TEST DATASET
        #################################
        X_test = df_original_test.iloc[:,:-1]
        y_test = df_original_test.iloc[:, -1]

        ############################################################################
        # TRAINING CLASSIFIERS DISTRIBUTED DATASETS
        ############################################################################

        #################################
        # BALANCED PARTITIONS
        #################################
        if (is_balanced):
            X_train_balanced = df_training_distributed_balanced_final_node.iloc[:,:-1]
            y_train_balanced = df_training_distributed_balanced_final_node.iloc[:, -1]

            # Loop over each classifier
            for classifier, name in zip(list_classifiers, list_classifiers_names):
                print(f'\nThe classifier: {name}\n')
                start = time.time()
                classifier.fit(X_train_balanced.values, y_train_balanced.values)
                end = time.time()
                print(f'Elapsed time in training for balanced partition: {end - start}')

                y_pred = classifier.predict(X_test.values)
                print(f'The recall for {number_of_nodes} nodes in balanced partitioned ' \
                    f'is {recall_score(y_test, y_pred, average="macro")}.')
                print(f'The precision for {number_of_nodes} nodes in balanced partitioned ' \
                    f'is {precision_score(y_test, y_pred, average="macro")}.')

                # Append the info to the DataFrame...
                df_exp_info_balanced = df_exp_info_balanced.append(
                    pd.DataFrame([[
                        number_of_nodes,
                        n_exec,
                        name,
                        'spambase',
                        recall_score(y_test, y_pred, average="macro"),
                        precision_score(y_test, y_pred, average="macro"),
                        (end - start),
                        end_balanced_energy_distance_computing - start_balanced_enegery_distance_computing
                    ]],
                    columns=info_col_names),
                    ignore_index=True
                )

        ###################################
        # UNBALANCED PARTITIONS
        ###################################
        else:
            X_train_unbalanced = df_training_distributed_unbalanced_final_node.iloc[:,:-1]
            y_train_unbalanced = df_training_distributed_unbalanced_final_node.iloc[:, -1]

            for classifier, name in zip(list_classifiers,list_classifiers_names):
                print(f'\nThe classifier: {name}\n')
                start = time.time()
                classifier.fit(X_train_unbalanced.values, y_train_unbalanced.values)
                end = time.time()
                print(f'Elapsed time in training for unbalanced partition: {end - start}')

                y_pred = classifier.predict(X_test.values)
                print(f'The recall for {number_of_nodes} nodes in unbalanced '\
                    f'partitioned is {recall_score(y_test, y_pred, average="macro")}.')
                print(f'The precision for {number_of_nodes} nodes in unbalanced partitioned ' \
                    f'is {precision_score(y_test, y_pred, average="macro")}.')

    ############################################################################
    # TRAINING CLASSIFIERS CENTRALIZED DATASETS
    ############################################################################

    ######################
    # BALANCED PARTITIONS
    ######################
    print('Centralized training...\n')
    if (is_balanced):
        X_train_centralized_balanced = pd.read_csv(
            'sampled_centralized_balanced.csv', 
            sep=',',
            header=0
        ).iloc[:,:-1]
        
        y_train_centralized_balanced = pd.read_csv(
            'sampled_centralized_balanced.csv', 
            sep=',', 
            header=0
        ).iloc[:, -1]

        for classifier, name in zip(list_classifiers,list_classifiers_names):
            print(f'\nThe classifier: {name}\n')
            start = time.time()
            classifier.fit(X_train_centralized_balanced.values, y_train_centralized_balanced.values)
            end = time.time()
            print(f'Elapsed time in training for balanced centralized: {end - start}')

            y_pred = classifier.predict(X_test.values)
            print(f'The recall for {number_of_nodes} nodes in balanced centralized ' \
                f'is {recall_score(y_test, y_pred, average="macro")}.')
            print(f'The precision for {number_of_nodes} nodes in balanced centralized ' \
                f'is {precision_score(y_test, y_pred, average="macro")}.')

            # Append the info to the DataFrame...
            df_exp_info_central_balanced = df_exp_info_central_balanced.append(
                pd.DataFrame([[
                    n_exec,
                    name,
                    'spambase',
                    recall_score(y_test, y_pred, average="macro"),
                    precision_score(y_test, y_pred, average="macro"),
                    (end - start)
                ]],
                columns=info_col_names_central),
                ignore_index=True
            )

    #######################
    # UNBALANCED PARTITIONS
    #######################
    else:
        X_train_centralized_unbalanced = pd.read_csv(
            'sampled_centralized_unbalanced.csv', 
            sep=',',
            header=0
        ).iloc[:, :-1]

        y_train_centralized_unbalanced = pd.read_csv(
            'sampled_centralized_unbalanced.csv',
            sep=',',
            header=0
        ).iloc[:, -1]

        for classifier, name in zip(list_classifiers,list_classifiers_names):
            print(f'\nThe classifier: {name}\n')
            start = time.time()
            classifier.fit(X_train_unbalanced.values, y_train_unbalanced.values)
            end = time.time()
            print(f'Elapsed time in training for unbalanced centralized: {end - start}')

            y_pred = classifier.predict(X_test.values)
            print(f'The recall for {number_of_nodes} nodes in unbalanced centralized ' \
                f'is {recall_score(y_test, y_pred, average="macro")}.')
            print(f'The precision for {number_of_nodes} nodes in unbalanced centralized ' \
                f'is {precision_score(y_test, y_pred, average="macro")}.')


# End of the experiments
# Get the .csv file with the experiment info
print('Getting the .csv data with all the info...')
df_exp_info_balanced.to_csv('experiment_info_balanced.csv')
df_exp_info_central_balanced.to_csv('experiment_info_central_balanced.csv')
print('Experiment finished!')