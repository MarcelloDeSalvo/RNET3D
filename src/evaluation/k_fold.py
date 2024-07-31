import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import datetime

import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt

import tabulate
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger

# Main functions ----------------------------------------------------------
def k_fold_loop(train_gen, val_gen, model_callback, fve_callbacks, config, train_and_test_ids, metrics_dictionary):
    history_k, fve_evaluations = run_k_fold_training(train_gen, val_gen, model_callback, fve_callbacks, config, train_and_test_ids, metrics_dictionary)

    for i, fve_callback in enumerate(fve_callbacks):
        summarize_and_plot_results(history_k, fve_evaluations[i], config, metrics_dictionary)
        
    save_config(config)

def run_k_fold_training(train_gen, val_gen, model_callback, fve_callbacks, config, train_and_test_ids, metrics_dictionary):
    history_k = []
    kfold = KFold(n_splits=config['k'], shuffle=True, random_state=42)
    
    # Create list of lists to store the metrics for each evaluation based on the number of callbacks
    fve_evaluations = [[] for _ in range(len(fve_callbacks))]

    for fold, (train_ids_k, val_ids_k) in enumerate(kfold.split(train_and_test_ids)):
        train_ids_k = [train_and_test_ids[i] for i in train_ids_k]
        val_ids_k = [train_and_test_ids[i] for i in val_ids_k]

        train_ids_k *= config['data_multiplier']
        val_ids_k *= config['data_multiplier']

        steps = len(train_ids_k) // config['batch_size']
        val_steps = len(val_ids_k) // config['batch_size']

        print_fold_info(fold, train_ids_k, val_ids_k)

        path = create_output_dirs(config, fold)
        save_train_val_ids(train_ids_k, val_ids_k, path, fold)

        callbacks, checkpoint_path = create_callbacks(fold)
        model = model_callback()
        k_history = model.fit(train_gen, validation_data=val_gen, epochs=config['epochs'], steps_per_epoch=steps, validation_steps=val_steps, callbacks=callbacks)

        history_k.append(k_history.history)
        model.load_weights(checkpoint_path).expect_partial()
        model.save(f'{path}/model_fold_{fold}.h5')

        plot_loss(k_history, fold, config)
        plot_metrics(k_history, fold, config)
        save_history(k_history, fold, config)

        # Evaluate the model on the full volumes
        for fve_callback in fve_callbacks:
            full_volume_eval = fve_callback(model, val_ids_k, config)
            fold_dataframe = full_volume_eval.evaluate()
            fve_evaluations[fve_callbacks.index(fve_callback)].append(fold_dataframe)

    return history_k, fve_evaluations

def summarize_and_plot_results(history_k, full_volume_metrics, config, metrics_dictionary, name):
    create_final_output_dirs(config, name)
    best_metrics = compute_best_metrics(history_k, metrics_dictionary, config)
    avg_metrics = compute_avg_metrics(best_metrics, metrics_dictionary)
    full_volume_metrics_df = compute_full_metrics_dataframe(full_volume_metrics, config, name)

    plot_k_val_loss(history_k, config, name)
    plot_k_train_val_loss(history_k, config, name)
    plot_k_val_metrics(history_k, config, metrics_dictionary, name)
    plot_k_train_val_metrics(history_k, config, metrics_dictionary, name)
    plot_bar_k_val_metrics(best_metrics, config, metrics_dictionary, name)
    plot_bar_mean_metrics(avg_metrics, metrics_dictionary, config, name)
    plot_boxplot_metrics(full_volume_metrics_df, config, name)

def print_fold_info(fold, train_ids_k, val_ids_k):
    unique_train_ids = np.unique(train_ids_k)
    unique_val_ids = np.unique(val_ids_k)
    
    print(f'\n{"="*30}')
    print(f'FOLD {fold}')
    print(f'{"="*30}')
    print(f'Training on {len(train_ids_k)} examples ({len(unique_train_ids)} unique IDs)')
    print(f'Unique Train IDs: {unique_train_ids}')
    print(f'{"-"*30}')
    print(f'Validating on {len(val_ids_k)} examples ({len(unique_val_ids)} unique IDs)')
    print(f'Unique Validation IDs: {unique_val_ids}')
    print(f'{"="*30}\n')


# Utility functions -------------------------------------------------------------
def create_output_dirs(config, fold):
    main_path = f'../results/{config["model_name"]}/fold_{fold}'
    region_path = f'{main_path}/regions'
    brain_path = f'{main_path}/brain_mask'
    
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    if not os.path.exists(region_path):
        os.makedirs(region_path)
    if not os.path.exists(brain_path):
        os.makedirs(brain_path)
    return main_path

def create_final_output_dirs(config, name):
    main_path = f'../results/{config["model_name"]}/{name}'

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    return main_path

def save_train_val_ids(train_ids_k, val_ids_k, path, fold):
    with open(f'{path}/train_val_ids_fold_{fold}.txt', 'w') as f:
        for item in np.unique(train_ids_k):
            f.write(f"{item}\n")
        for item in np.unique(val_ids_k):
            f.write(f"{item}\n")
            
def save_config(config):
    with open(f'../results/{config["model_name"]}/config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f'{key}:{value}\n')

def save_history(k_history, fold, config):
    # save in a excel file
    df = pd.DataFrame(k_history)
    df.to_excel('../results/'+config['model_name']+'/fold_'+str(fold)+'/'+'/history_fold_'+str(fold)+'.xlsx', index=False)

def create_callbacks(fold, config):
    csv_logger = CSVLogger('../results/'+config['model_name']+'/training.log', separator=',', append=False)
    log_dir = "../results/"+config['model_name']+"/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "../results/"+config['model_name']+"/checkpoint/"
    checkpoint_path = os.path.join(checkpoint_path, f"fold_{fold}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='min'),
            csv_logger
        ]
    return callbacks, checkpoint_path

# Metrics functions ----------------------------------------------------------
def compute_best_metrics(history_k, metrics_dic, config):
    print("Best metrics for each fold")

    # define a list of lists to store the best values of each metric in validation
    best_metrics_val = []

    for i in range(config['k']):
        print(f"|- Fold {i} -----------------")
        best_metric_k = {}

        best_epoch_k = np.argmin(history_k[i]['val_loss'])
        print(f"|- Best epoch: {best_epoch_k}")

        # take last value of each metric in validation
        best_metric_k['val_loss'] = history_k[i]['val_loss'][best_epoch_k]
        for key in metrics_dic.keys():
            best_metric_k['val_'+key] = history_k[i]['val_'+key][best_epoch_k]
            print(f"|- {'val_'+key}: {history_k[i]['val_'+key][best_epoch_k]}")

        best_metrics_val.append(best_metric_k)

    return best_metrics_val

def compute_avg_metrics(best_metrics_val, metrics_dic):
    # compute average for each fold
    print("Average metrics for each fold")

    average_metrics = {}

    average_metrics['val_loss'] = np.mean([h['val_loss'] for h in best_metrics_val])
    for key in metrics_dic.keys():
        average_metrics['val_'+key] = np.mean([h['val_'+key] for h in best_metrics_val])

    # print average metrics
    print("Average metrics")
    for key in average_metrics.keys():
        print(f"|- AVG VAL - {key}: {average_metrics[key]}")

    return average_metrics


def compute_full_metrics_dataframe(full_volume_metrics, config, name):
    # Convert the list of dictionaries to a pandas dataframe by adding the fold number as a column
    full_volume_metrics_df = pd.concat(full_volume_metrics, axis=0)
    full_volume_metrics_df['Fold'] = np.concatenate([[i]*len(full_volume_metrics[i]) for i in range(config['k'])])
    # Save the dataframe to an excel file

    print(tabulate.tabulate(full_volume_metrics_df, headers='keys', tablefmt='pretty'))

    full_volume_metrics_df.to_excel('../results/'+config['model_name']+'/'+name+'/full_volume_metrics.xlsx', index=False)
    return full_volume_metrics_df

# Plotting functions ----------------------------------------------------------
def plot_loss(k_history, fold, config):
    plt.figure()
    plt.plot(k_history.history['loss'], label='Train')
    plt.plot(k_history.history['val_loss'], label='Validation')
    plt.title(f'Fold {fold} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../results/'+config['model_name']+'/fold_'+str(fold)+'/'+'loss_fold_'+str(fold)+'.png')
    plt.show()    
    plt.close()

def plot_metrics(k_history, fold, config):
    '''
    Plots the history of a model training.
    :param _history: The history of the model training.
    :param config: The configuration of the model.
    '''
    # Set seaborn style
    sns.set_style('whitegrid')

    def count_metrics(history):
        i = 0
        for key in history.keys():
            if 'val' not in key:
                i += 1
        return i
    
    # Count the number of metrics
    n_metrics = count_metrics(k_history)

    columns = 5
    rows = n_metrics // columns
    
    # add 1 to rows if there is a remainder
    if n_metrics % (columns * rows) != 0:
       rows += 1

    # Create a figure with a subplot for each metric in a grid
    fig, axs = plt.subplots(rows,columns, figsize=(60,30))
    axs = axs.ravel()

    # Plot each metric
    for i in range(n_metrics):
        metric = list(k_history.keys())[i]
        if 'val' in metric:
            continue
        axs[i].plot(k_history[metric])
        axs[i].plot(k_history['val_' + metric])
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel('Epoch')
        axs[i].legend(['Train', 'Validation'], loc='upper left')

    # Save the plot
    plt.tight_layout()
    plt.savefig('../results/'+config['model_name']+'/fold_'+str(fold)+'/'+'metrics_fold_'+str(fold)+'.png')
    plt.show()
    plt.close()


def plot_loss(k_history, fold, config):
    # plot loss and val loss
    plt.figure(figsize=(15, 5))
    plt.plot(k_history['loss'])
    plt.plot(k_history['val_loss'])
    plt.title('model loss')
    plt.ylabel('log-loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('../results/'+config['model_name']+'/fold_'+str(fold)+'/'+'loss_fold_'+str(fold)+'.png')
    plt.show()

def plot_k_val_loss(history_k, config, name):
    plt.figure(figsize=(30, 15))
    for i in range(config['k']):
        plt.plot(np.arange(len(history_k[i]['val_loss'])),history_k[i]['val_loss'], label='val_fold_'+str(i))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(np.arange(0, len(history_k[0]['val_loss']), step=5), labels=np.arange(0, len(history_k[0]['val_loss']), step=5))
    plt.legend()

    plt.savefig('../results/'+config['model_name']+'/'+name+'/val_loss.png')
    plt.show()

def plot_k_train_val_loss(history_k, config, name):
    plt.figure(figsize=(30, 15))
    sns.set_style("whitegrid")
    train_colors = plt.cm.get_cmap('Greens', config['k'])  # Green palette for train
    val_colors = plt.cm.get_cmap('Reds', config['k'])     # Reds palette for val

    for i in range(config['k']):
        plt.plot(np.arange(len(history_k[i]['loss'])), history_k[i]['loss'], label='train_fold_'+str(i), color=train_colors(i))
        plt.plot(np.arange(len(history_k[i]['val_loss'])),history_k[i]['val_loss'], label='val_fold_'+str(i), color=val_colors(i))

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('../results/'+config['model_name']+'/'+name+'/train_val_loss.png')
    plt.show()

def plot_k_train_val_metrics(history_k, config, metrics_dictionary, name):
    fig, axs = plt.subplots(4, 3, figsize=(40, 20))
    axs = axs.ravel()

    # Define palettes
    train_palette = sns.color_palette("Greens", n_colors=config['k'])
    val_palette = sns.color_palette("Reds", n_colors=config['k'])

    # For each metric subplot, plot the history of each fold
    for i, key in enumerate(metrics_dictionary.keys()):
        axs[i].set_title('Train - Val ' + key)
        for j in range(config['k']):
            axs[i].plot(np.array(history_k[j][key]), color=train_palette[j], label=f'Fold {j} Train')
            axs[i].plot(np.array(history_k[j]['val_'+key]), color=val_palette[j], label=f'Fold {j} Val')
        # Legend with fold number
        axs[i].legend()

    plt.savefig('../results/'+config['model_name']+'/'+name+'/k_train_val_metrics.png')
    plt.show()

def plot_k_val_metrics(history_k, config, metrics_dic, name):
    fig, axs = plt.subplots( 4, 3, figsize=(40, 20))
    val_palette = sns.color_palette("Reds", n_colors=config['k'])
    axs = axs.ravel()
    # for each metric subplot, plot the history of each fold
    for i, key in enumerate(metrics_dic.keys()):
        axs[i].set_title('Val ' + key)
        for j in range(config['k']):
            axs[i].plot(np.array(history_k[j]['val_'+key]), color=val_palette[j])
            # legend with fold number
            axs[i].legend([f'Fold {i} Val' for i in range(config['k'])], loc='lower right')

    plt.savefig('../results/'+config['model_name']+'/'+name+'/k_val_metrics.png')
    plt.show()

def plot_bar_k_val_metrics(best_metrics_val, config, metrics_dic, name):
    for key in metrics_dic.keys():

        # bar plot with seaborn
        plt.figure(figsize=(10, 5))
        sns.barplot(x=[f'Fold {i}' for i in range(config['k'])], y=[h['val_'+key] for h in best_metrics_val], palette="deep", width=0.6)
        plt.xlabel('Fold')

        plt.legend()
        
        # Set y-axis limits starting from 0.5
        plt.ylim(bottom=0.5)

        plt.ylabel(key)
        plt.title('Best '+key+' for each fold')
        # add exact value
        for i, v in enumerate([h['val_'+key] for h in best_metrics_val]):
            plt.text(i, v, str(round(v, 3)), color='blue', fontweight='bold')
        
        plt.savefig('../results/'+config['model_name']+'/'+name+'/best_val_'+key+'.png')
        plt.show()

def plot_bar_mean_metrics(average_metrics, metrics_dic, config, name):
    # last plot with the mean of the metrics for each fold
    plt.figure(figsize=(20, 10))

    # add more ticks to the y axis
    plt.yticks(np.arange(0, 1, 0.1))

    # Plot the average k fold metrics of the mean of the last epoch in one single plot
    ax  = sns.barplot(x=list(average_metrics.keys()), y=list(average_metrics.values()), palette="deep", hue=list(average_metrics.keys()))
    for i in range(len(average_metrics)):
        ax.bar_label(ax.containers[i])

    plt.ylabel('Value')
    plt.title('Average metrics')
    plt.tight_layout()
    plt.savefig('../results/'+config['model_name']+'/'+name+'/k_val_average_metrics.png')
    plt.show()


def plot_boxplot_metrics(all_fold_metrics_df, config, name):

    def plot_fold_comparisons(metric, stride, all_fold_metrics_df, ax, low_limit=0):
        # Extract the 'Mean' column of each metric for each fold associated with the lower stride for 'dice_coeff'
        fold_metrics = all_fold_metrics_df[all_fold_metrics_df['Metric'] == metric]
        fold_metrics = fold_metrics[fold_metrics['Stride'] == stride]

        # Make 'Subject' a column and not an index
        fold_metrics = fold_metrics.reset_index()

        # Reset the index
        # print(tabulate.tabulate(fold_metrics, headers='keys', tablefmt='pretty'))
        melted_df = pd.melt(fold_metrics, id_vars=['Subject', 'Fold', 'Modality', 'Metric'], value_vars=['Mean'], var_name='Class', value_name='Means')


        # Plot the comparison between the folds for each class
        sns.boxplot(x='Fold', y='Means', hue='Modality', data=melted_df, ax=ax, palette='deep')

        ax.set_title(f'{metric} per Fold for each Class')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)

        # Limit the y-axis to 0 to 1
        ax.set_ylim(low_limit, 1)

    # Extract unique metrics column values
    metrics = all_fold_metrics_df['Metric'].unique()
    strides = all_fold_metrics_df['Stride'].unique()
    _min_stride = np.min(strides)

    # Create a figure with a subplot for each metric in a grid
    _n_col = 3
    _n_row = len(metrics) // _n_col


    fig, axs = plt.subplots(_n_row, _n_col, figsize=(20, 15))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        plot_fold_comparisons(metric, _min_stride, all_fold_metrics_df, axs[i], low_limit=0)

    plt.tight_layout()
    plt.savefig('../results/'+config['model_name']+'/'+name+'/fold_comparison_metrics_0_1.png')
    plt.show()


    fig, axs = plt.subplots(_n_row, _n_col, figsize=(20, 15))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        plot_fold_comparisons(metric, _min_stride, all_fold_metrics_df, axs[i], low_limit=0.5)

    plt.tight_layout()
    plt.savefig('../results/'+config['model_name']+'/'+name+'/fold_comparison_metrics_05_1.png')
    plt.show()

    fig, axs = plt.subplots(_n_row, _n_col, figsize=(20, 15))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        plot_fold_comparisons(metric, _min_stride, all_fold_metrics_df, axs[i], low_limit=0.7)

    plt.tight_layout()
    plt.savefig('../results/'+config['model_name']+'/'+name+'/fold_comparison_metrics_07_1.png')
    plt.show()




