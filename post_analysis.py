import argparse
import os
import numpy as np
import time
import copy
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import ttest_ind, f_oneway, kruskal

# Keisuke Fujii, 2023

parser = argparse.ArgumentParser(description='Grid search')


# Experiment
parser.add_argument('--experiment', type=str, default="flies", help="Experiment to be performed (default: 'flies')")
parser.add_argument('--data_dir', type=str, default="./datasets", help="Experiment to be performed(default: './datasets')")
parser.add_argument('--video_dir', type=str, default="./video", help="Experiment to be performed (default: './video')")
parser.add_argument('--figure_dir', type=str, default="./figure", help="Experiment to be performed (default: './figure')")
parser.add_argument("--test_samples", type=int, default=10)

# Model specification
parser.add_argument('--model', type=str, default='gvar', help="Model to train (default: 'gvar')")
parser.add_argument('--K', type=int, default=5, help='Model order (default: 5)')

# Conditions
parser.add_argument('--example_figure', action='store_true')
parser.add_argument('--plot_trajectory', action='store_true')
parser.add_argument('--create_video', action='store_true')


# Parsing args
args = parser.parse_args()
 
print( str(args.experiment) + " datasets...")

video_dir = os.path.join(args.video_dir, args.experiment) 
figure_dir = os.path.join(args.figure_dir, args.experiment) 
os.makedirs(video_dir,exist_ok=True)
os.makedirs(figure_dir,exist_ok=True)

num_files = args.test_samples

example_figure = args.example_figure
plot_trajectory = args.plot_trajectory
create_video = args.create_video

# input lists
if args.experiment == 'mice': 
    K_all = 3
    Fs = 30
    n_T = 30
    T = 10*Fs
    dim_xy = 2 
    List = ['1','2','3']
    
    count_interact = np.zeros((num_files, n_T, K_all, K_all, 3))
    count_interact_gvar = np.zeros((num_files, n_T, K_all, K_all, 3))
elif args.experiment == 'flies':
    K_all = 8
    Fs = 30
    n_T = 12
    T = 20*Fs
    dim_xy = 2
    List = ['1','2','3','4','5','6','7','8']

    count_interact = np.zeros((num_files, n_T, K_all, K_all, 3)) 
    count_interact_gvar = np.zeros((num_files, n_T, K_all, K_all, 3))
else:
    print('TBD')
    import pdb; pdb.set_trace()

weights_dir = os.path.join('weights', f'{args.experiment}_gvar_{num_files}')

# Final paths
mat_dir1 = os.path.join(weights_dir, '_TEST_bidirection')
mat_dir2 = os.path.join(weights_dir, '_TEST_percept_CF_pred_self')


for f in range(num_files):
    # GVAR
    mat_file = os.path.join(mat_dir1, f'coeffs_{f+1}.mat')
    gvar_data = loadmat(mat_file)
    coeffs_raw_gvar = gvar_data["coeffs_raw"]
    coeffs_gvar = gvar_data["coeffs"]
    data_gvar = gvar_data["data"]
    coeffs_time_gvar = gvar_data["coeffs_time"]
    args_gvar = gvar_data["args"]
    preds_gvar = gvar_data["preds"]    
    
    coeffs_time_gvar = coeffs_time_gvar / np.max(np.abs(coeffs_gvar))
    coeffs_gvar_ = coeffs_time_gvar
    y_max_gvar = np.median(np.max(np.max(np.abs(coeffs_time_gvar), axis=0), axis=0))

    # ABM (our method)
    mat_file2 = os.path.join(mat_dir2, f'coeffs_{f+1}.mat')
    abm_data = loadmat(mat_file2)
    coeffs_raw = abm_data["coeffs_raw"]
    coeffs = abm_data["coeffs"]
    data_abm = abm_data["data"]
    coeffs_time = abm_data["coeffs_time"]
    args_abm = abm_data["args"]
    preds_abm = abm_data["preds"] 

    order = args_abm["K"][0][0][0][0]
    num_dims = args_abm["num_dims"][0][0][0][0]
    Start = 0
    End = data_abm.shape[1] - order
    K = coeffs_time.shape[1]
    coeffs_ = coeffs_time
        
    # Normalize 
    # coeffs_ = coeffs_ / np.max(coeffs_)

    # Get max value 
    y_max = np.median(np.max(np.max(np.abs(coeffs_), axis=0), axis=0))
    
    # Reshape data
    # vel,loc,range,v_dir,dist
    dataK = np.reshape(data_abm[0,order:order+End,:], (End, K, num_dims))
    dataK = np.transpose(dataK,[0,2,1])

    # Get positions
    pos = dataK[:,dim_xy:dim_xy*2,:] 

    # Get min/max positions
    max_xy = np.max(np.max(pos, axis=0), axis=1)
    min_xy = np.min(np.min(pos, axis=0), axis=1)

    # Get distances
    dist = dataK[:,(-K+2):, :]   

    # Create time vector
    Time = np.arange(1/Fs, End/Fs, 1/Fs)

    # Get thresholds
    max_coeffs_ = np.max(coeffs_)
    min_coeffs_ = np.min(coeffs_)

    # Initialize binary array
    coeffs_binary = np.zeros((coeffs_.shape[0], K, K-1))
    # coeffs_binary[:,:,:] = np.nan

    # GVAR thresholds
    if 'mice' in args.experiment or 'flies' in args.experiment:
        max_coeffs_gvar = np.max(coeffs_gvar_)
        min_coeffs_gvar = np.min(coeffs_gvar_)
        
        coeffs_binary_gvar = np.zeros((coeffs_gvar_.shape[0], K, K-1))
        # coeffs_binary_gvar[:,:,:] = np.nan

    for k in range(K):
        jj = 0
        for j in range(K):
            if j != k:
                # Thresholding
                coeffs_binary[coeffs_[:,k,jj] >= max_coeffs_/2, k, jj] = 1
                coeffs_binary[coeffs_[:,k,jj] <= min_coeffs_/2, k, jj] = -1
                
                # Analysis
                # diff_coeff = np.diff(coeffs_binary[:,k,jj])
                
                if 'sula' in args.experiment:
                    count_interact[f,k,j,1] += np.sum(coeffs_binary[:,k,jj]==1) 
                    count_interact[f,k,j,2] += np.sum(coeffs_binary[:,k,jj]==-1) 
                    
                elif 'mice' in args.experiment or 'flies' in args.experiment:
                    
                    coeffs_binary_gvar[coeffs_gvar_[:,k,jj] >= max_coeffs_gvar/2, k, jj] = 1
                    coeffs_binary_gvar[coeffs_gvar_[:,k,jj] <= min_coeffs_gvar/2, k, jj] = -1
                    
                    for t in range(n_T):
                    
                        if t < n_T:
                            End_ = (t+1)*T
                        else:
                            End_ = coeffs_binary_gvar.shape[0]
                            
                        # Increment counts
                        count_interact[f, t, k, j, 0] = np.sum(coeffs_binary[t*T:End_, k, jj]==1)
                        count_interact[f, t, k, j, 1] = np.sum(coeffs_binary[t*T:End_, k, jj]==-1)  
                        count_interact[f, t, k, j, 2] = np.sum(coeffs_binary[t*T:End_, k, jj]==0)

                        count_interact_gvar[f, t, k, j, 0] = np.sum(coeffs_binary_gvar[t*T:End_, k, jj]==1)
                        count_interact_gvar[f, t, k, j, 1] = np.sum(coeffs_binary_gvar[t*T:End_, k, jj]==-1)
                        count_interact_gvar[f, t, k, j, 2] = np.sum(coeffs_binary_gvar[t*T:End_, k, jj]==0)
                            
                # Get distances
                if 'sula' in args.experiment:
                    dist = np.sqrt(np.sum((pos[:,:,k]-pos[:,:,j])**2, axis=2))
                    count_interact[f,k,j,3] += np.sum(dist <= 1000)
                    
                jj += 1

    if example_figure:
        # Create figure 
        fig, axs = plt.subplots(K, K, figsize=(10,8))

        for k in range(K):

            # Create legend labels
            labels = list(List) 

            for j in range(K):
                if j == k:
                    labels.pop(j) 

            # for j, ax in enumerate(axs[k]):
            for j in range(K-1):    
                jj = int(labels[j])-1
                # if j != k and j < K-1:
                ax = axs[k][jj]

                # Plot data
                ax.plot(coeffs_[:,k,j]/y_max, label='ABM (ours)') 
                ax.plot(coeffs_gvar_[:,k,j]/y_max_gvar, label='GVAR')
                
                # Set labels
                ax.set_ylabel(f"{k+1}<-{labels[j]}")
                
                # Set limits
                ax.set_xlim(0, End)
                ax.set_ylim(-1, 1)

                if j == 1 and k == 0:
                    plt.legend() # did not work

            for j in range(K):
                if j == k: 
                    # Don't plot on diagonal
                    ax = axs[k][j]
                    ax.axis('off') 
                    
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(figure_dir, f'example_{f+1}.png'))

    # Set title
    if 'sula' in args.experiment:
        print('TBD') # title_str = f"{args.experiment}{metadata[f,1]}-T-{metadata[f,2]}-{metadata[f,3]}"
    elif 'mice' in args.experiment or 'flies' in args.experiment:
        title_str = f"{args.experiment}{f}-T-{Start}-{End}"
    else:
        title_str = f"{args.experiment}-T-{Start}-{End}"      
         
    if plot_trajectory:
        # Create figure 
        fig = plt.figure()
        if 'peregrine' in args.experiment:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)


            
        ax.set_title(title_str, fontsize=8)

        # Plot each trajectory
        for k in range(K):
            xy = pos[:,:,k]
            # Set color
            if k%8 == 0: clr = 'r' 
            elif k%8 == 1: clr = 'g'
            elif k % 8 == 2: clr = 'b'
            elif k % 8 == 3: clr = 'k'
            elif k % 8 == 4: clr = 'm'
            elif k % 8 == 5: clr = 'c'
            elif k % 8 == 6: clr = [1, 0.4, 0.6]  
            elif k % 8 == 7: clr = [0.5, 0.5, 0.5]
            
            # Plot 
            if 'peregrine' in args.experiment:
                ax.plot3D(xy[:,0], xy[:,1], xy[:,2], '-', c=clr)
                ax.text(xy[0,0], xy[0,1], xy[0,2], s=str(k))
            else:
                ax.plot(xy[:,0], xy[:,1], '-', c=clr)
                
                if 'flies' not in args.experiment:
                    try: ax.text(xy[0,0], xy[0,1], s=str(k))
                    except: import pdb; pdb.set_trace()
                    
        # Set axes limits  
        if 'peregrine' in args.experiment:
            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
            ax.set_zlim(min_xy[2], max_xy[2])
            ax.view_init(-10, 40)
            ax.set_zlabel('z')
        elif 'mice' in args.experiment or 'sula' in args.experiment or 'flies' in args.experiment:
            ax.set_xlim(min_xy[0], max_xy[0])
            ax.set_ylim(min_xy[1], max_xy[1])
        else:
            ax.set_xlim(min_xy[f,0], max_xy[f,0])
            ax.set_ylim(min_xy[f,1], max_xy[f,1])
            
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Show legend
        ax.legend(List)

        # plt.show()
        plt.savefig(os.path.join(figure_dir, f'trajectory_{f+1}.png'))

    if create_video: # not worked
        if 'flies' not in args.experiment:
            KK = K  
            K_plot = K*(K-1)
        else:
            KK = 1  
            K_plot = (K-1)
        duration = Fs

        # Setup figure
        fig, axs = plt.subplots(K_plot, 2, figsize=(10, 5))
        fig.tight_layout()

        # Initialize video writer
        video_file = f'{title_str}_analyzed.mp4'

        writer = animation.FFMpegWriter(fps=Fs) 
        
        # Plot each frame
        with writer.saving(fig, os.path.join(video_dir,video_file), dpi=200):
            for t in range(Start, Start+100): #End-order): # for debugging
                jjj = 0
                # Plot timeseries 
                for k in range(KK):
                    # Create legend labels
                    labels = list(List) 

                    for j in range(K):
                        if j == k:
                            labels.pop(j) 

                    # for j, ax in enumerate(axs[k]):
                    # jj = int(labels[j])-1

                    for j in range(0, K-1):
                        ax = axs[jjj][1]
                        ax.clear()
                        # Plot data
                        ax.plot(coeffs_[0:End,k,j] / y_max)
                        # Add thresholds, vertical line
                        ax.axhline(max_coeffs_ / y_max / 2, c='m') 
                        ax.axhline(min_coeffs_ / y_max / 2, c='c') 
                        
                        # Plot horizontal line  
                        plt.plot([0, End], [0, 0], 'k-')

                        # Plot vertical line
                        plt.plot([t-order, t-order], [-1, 1], 'k-')  # not worked

                        # Set x and y limits
                        plt.xlim(0, End)
                        plt.ylim(-1, 1)

                        # Label y-axis 
                        plt.ylabel(f"{k}<-{labels[j]}") # not worked

                        jjj = jjj + 1 

                        writer.grab_frame()
                
                # Plot motion: not worked
                '''ax = axs[:, 0]
                # ax = plt.subplot(1,2,1)
                ax.clear()

                for k in range(K):
                    # Get positions
                    xy = pos[t,:,k]  

                    # Get long trajectory 
                    if t <= duration:
                        xy_long = pos[:t,:,k]
                    else:
                        xy_long = pos[t-duration:t,:,k]

                    # Set plot parameters
                    ms = 12
                    lw = 1
                    if k%5 == 0: clr = 'r'
                    elif k%5 == 1: clr = 'g'  
                    elif k%5 == 2: clr = 'b'
                    elif k%5 == 3: clr = 'k'
                    elif k%5 == 4: clr = 'm'

                    # Plot 
                    if 'peregrine' in args.experiment:
                        plt.plot(xy[0], xy[1], xy[2], 'o', ms=ms, lw=lw, c=clr)
                        plt.plot(xy_long[:,0], xy_long[:,1], xy_long[:,2], '-', c=clr)
                        plt.text(xy[0], xy[1], xy[2], str(k))
                    else:  
                        plt.plot(xy[0], xy[1], 'o', ms=ms, lw=lw, c=clr) 
                        plt.plot(xy_long[:,0], xy_long[:,1], '-', c=clr)
                        plt.text(xy[0], xy[1], str(k))

                    # plt.legend()

                # Set axis limits
                if 'peregrine' in args.experiment:
                    ax.set_xlim(min_xy[f,0], max_xy[f,0])
                    ax.set_ylim(min_xy[f,1], max_xy[f,1])
                    ax.set_zlim(min_xy[f,2], max_xy[f,2])
                    ax.view_init(-10, 40)
                    ax.set_zlabel('z')
                elif 'mice' in args.experiment or 'sula' in args.experiment or 'flies' in args.experiment:
                    ax.set_xlim(min_xy[0], max_xy[0]) 
                    ax.set_ylim(min_xy[1], max_xy[1])
                else:
                    ax.set_xlim(min_xy[f,0], max_xy[f,0])
                    ax.set_ylim(min_xy[f,1], max_xy[f,1])

                # Set labels
                ax.set_xlabel('x')  
                ax.set_ylabel('y')
                
                # Set title 
                ax.set_title(f'{title_str}, Frame {t} ({Fs}Hz)')

                # Turn off box
                ax.set_axisbelow(True) 

                # Redraw figure 
                writer.grab_frame()'''
                
# create Table

# Initialize count_interact_id with NaNs
count_interact_id = np.full((num_files, K_all, K_all, 3), np.nan)

if 'mice' in args.experiment:
    # Compute the count tables
    count_table = np.sum(np.sum(count_interact, axis=3), axis=2) / Fs
    count_table_gvar = np.sum(np.sum(count_interact_gvar, axis=3), axis=2) / Fs
    count_table = np.transpose(count_table,[0,2,1])
    count_table_gvar = np.transpose(count_table_gvar,[0,2,1])

    # Prepare mean and standard deviation tables
    count_interact_msd = np.empty((num_files, 2, 2))
    count_int_gvar_msd = np.empty((num_files, 2, 2))
    
    for f in range(num_files):
        for sgn in range(2): 
            count_interact_msd[f, sgn, 0] = np.mean(count_table[f, sgn, :])
            count_interact_msd[f, sgn, 1] = np.std(count_table[f, sgn, :]) / np.sqrt(n_T)
            count_int_gvar_msd[f, sgn, 0] = np.mean(count_table_gvar[f, sgn, :])
            count_int_gvar_msd[f, sgn, 1] = np.std(count_table_gvar[f, sgn, :]) / np.sqrt(n_T)
    
    # Statistical tests
    res = {}
    for sgn in range(2): 
        #res[sgn] = ttest_ind(count_table[0, sgn, :], count_table[1, sgn, :], equal_var=False)
        #res[sgn + 2] = ttest_ind(count_table_gvar[0, sgn, :], count_table_gvar[1, sgn, :], equal_var=False)
        stat, p = kruskal(count_table[0, sgn, :], count_table[1, sgn, :])
        res[sgn] = {'statistic': stat, 'p_value': p}
        stat_gvar, p_gvar = kruskal(count_table_gvar[0, sgn, :], count_table_gvar[1, sgn, :])
        res[sgn + 2] = {'statistic': stat_gvar, 'p_value': p_gvar}

elif 'flies' in args.experiment:
    # Initialize the count tables
    count_table = np.zeros((num_files, n_T, 3)) 
    count_table_gvar = np.zeros((num_files, n_T, 3))
    count_interact_msd = np.zeros((num_files, 2, 2))
    count_int_gvar_msd = np.zeros((num_files, 2, 2))
    
    # Perform calculations
    for f in range(num_files):
        n_male = 8 if f == 0 else 4  
        count_table[f, :, :] = np.sum(np.sum(count_interact[f, :, :n_male, :, :], axis=2), axis=1) / (Fs * n_male)
        count_table_gvar[f, :, :] = np.sum(np.sum(count_interact_gvar[f, :, :n_male, :, :], axis=2), axis=1) / (Fs * n_male)

        for sgn in range(2):
            count_interact_msd[f, sgn, 0] = np.mean(count_table[f, sgn, :])
            count_interact_msd[f, sgn, 1] = np.std(count_table[f, sgn, :]) / np.sqrt(n_T)
            count_int_gvar_msd[f, sgn, 0] = np.mean(count_table_gvar[f, sgn, :])
            count_int_gvar_msd[f, sgn, 1] = np.std(count_table_gvar[f, sgn, :]) / np.sqrt(n_T)

    count_table = np.transpose(count_table,[0,2,1])
    count_table_gvar = np.transpose(count_table_gvar,[0,2,1])

    # Statistical tests
    res = {}
    for sgn in range(2): 
        #res[sgn] = ttest_ind(count_table[0, sgn, :], count_table[1, sgn, :], equal_var=False)
        #res[sgn + 2] = ttest_ind(count_table_gvar[0, sgn, :], count_table_gvar[1, sgn, :], equal_var=False)
        stat, p = kruskal(count_table[0, sgn, :], count_table[1, sgn, :])
        res[sgn] = {'statistic': stat, 'p_value': p}
        stat_gvar, p_gvar = kruskal(count_table_gvar[0, sgn, :], count_table_gvar[1, sgn, :])
        res[sgn + 2] = {'statistic': stat_gvar, 'p_value': p_gvar}

# result figure
num_files = count_table.shape[0]  # Assuming count_table has the shape (num_files, conditions, measures)
n_T = count_table.shape[2]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), num=1000)

# Check if the filename contains 'mice' or 'flies'
if 'mice' in args.experiment or 'flies' in args.experiment:
    for m, ax in enumerate(axes.flatten(), start=1):
        data = count_table[:, :2, :].reshape(num_files * 2, n_T) if m == 1 else count_table_gvar[:, :2, :].reshape(num_files * 2, n_T)
        df = pd.DataFrame(data.T)
        
        # Boxplot without fliers (outliers)
        df.boxplot(ax=ax, showfliers=False)
        
        # Setting x-tick labels
        
        
        # Customizing boxplot colors
        for i, box in enumerate(ax.artists):
            box.set_edgecolor('black')
            plt.setp(ax.lines, color='black')
            if num_files in [2, 3]:
                color = 'blue' if i % 2 == 0 else 'red'
                plt.setp(box, edgecolor=color)
        
        ax.set_title('GVAR' if m == 2 else 'Our method')
        
        # Custom y-axis label based on file name
        ylabel = 'duration [sec]' if 'mice' in args.experiment else 'duration [sec/fly]'
        ax.set_ylabel(ylabel)

        # Legend
        if 'mice' in args.experiment and num_files in [2, 3]:
            labels = ['same cage', 'different cage'] # if num_files == 2 else ['VTA negative', 'same cage', 'different cage']
            ax.legend(ax.artists[::num_files], labels, loc='best')
            ax.set_xticklabels(['attraction \n (diff. cage)', 'repulsion \n (diff. cage)','attraction \n (same cage)', 'repulsion \n (same cage)'])

        elif 'flies' in args.experiment:
            ax.legend(ax.artists[::num_files], ['mixed group', 'male-only group'], loc='best')
            ax.set_xticklabels(['attraction \n (mixed grp.)', 'repulsion \n (mixed grp.)','attraction \n (male-only)', 'repulsion \n (male-only)'])
        
        plt.savefig(os.path.join(figure_dir, 'results.png'))

import pdb; pdb.set_trace()
