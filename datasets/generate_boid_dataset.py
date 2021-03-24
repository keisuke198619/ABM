import os
import json
import time
import numpy as np
import argparse

# from model import utils 
from boid import Boid

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation", type=str, default="boid", help="What simulation to generate."
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=50000,
        help="Number of training simulations to generate.",
    )
    parser.add_argument(
        "--num-valid",
        type=int,
        default=10000,
        help="Number of validation simulations to generate.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=10000,
        help="Number of test simulations to generate.",
    )
    parser.add_argument(
        "--length", type=int, default=5000, help="Length of trajectory."
    )
    parser.add_argument(
        "--length_test", type=int, default=10000, help="Length of test set trajectory."
    )
    parser.add_argument(
        "--sample_freq",
        type=int,
        default=100,
        help="How often to sample the trajectory.",
    )
    parser.add_argument(
        "--n_boids", type=int, default=5, help="Number of balls in the simulation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--datadir", type=str, default="data/boid", help="Name of directory to save data to."
    )

    parser.add_argument(
        "--avoid",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--bat",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Have symmetric connections (non-causal)",
    )

    parser.add_argument("--video", type=int, default=0)
    args = parser.parse_args()

    # args.length_test = args.length * 2

    print(args)
    return args


def generate_dataset(args, width, sim, num_sims, length, sample_freq, train=1):

    if train == 1:
        suffix = 'train'
    elif train == 0:
        suffix = 'val'
    elif train == -1:
        suffix = 'test'

    if args.bat:
        suffix += '_bat'

    suffix += "_" + args.simulation

    suffix += str(args.n_boids)

    if args.partial:
        suffix += "_partial" 

    if args.avoid:
        suffix += "_avoid" 

    if args.length != 10000:
        suffix += "_l" + str(args.length)

    if args.length != 1:
        suffix += "_Fs" + str(args.sample_freq)

    print(suffix)

    loc_all = list()
    vel_all = list()
    edges_all = list()
    edges_res_all = list()

    edges = None

    for i in range(num_sims):
        sim_i = sim
        t = time.time()
        loc, vel, edges_result, edges = sim_i.sample_trajectory(
            args,
            i,
            T=length,
            sample_freq=sample_freq,
            edges=edges
        )
        if True: # i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)
        edges_res_all.append(edges_result)


    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)
    edges_res_all = np.stack(edges_res_all)

    if train < 1: # args.video > 0:
        for s in range(args.video):
            vel_norm = np.sqrt((vel_all[s] ** 2).sum(axis=1))
            loc = loc_all[s]
            edge = edges_res_all[s]
            try: fig = plt.figure()
            except: import pdb; pdb.set_trace()
            axes = plt.gca()
            axes.set_xlim([-width, width])
            axes.set_ylim([-width, width])
            for i in range(loc.shape[-1]):
                plt.plot(loc[:, 0, i], loc[:, 1, i])
                plt.plot(loc[0, 0, i], loc[0, 1, i], "d")

            # plt.show()
            

            # Create animation.
            # Set up formatting for the movie files.
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)

            #try: fig = plt.figure()
            # except: import pdb; pdb.set_trace()
            ax = plt.axes(xlim=(-width, width), ylim=(-width, width))

            lines = [
                plt.plot([], [], marker="$" + "{:d}".format(i) + "$", alpha=1, markersize=10)[0]
                for i in range(loc.shape[-1])
            ]

            ani = FuncAnimation(fig, update, frames=loc.shape[0], fargs = (lines,loc,edge),interval=5, blit=True)

            ani.save(filename=os.path.join(args.datadir+'/video',suffix+"_"+str(s)+".mp4"), writer=writer)
            
            # fig.close()
            plt.close(fig)
            print('video '+str(s)+' is created')


    return loc_all, vel_all, edges_res_all, edges_all

def update(frame,lines,loc,edge):
    try: 
        plt.title('sum of edges: '+str(np.sum(edge[frame])))
        for j, line in enumerate(lines):
            line.set_data(loc[frame, 0, j], loc[frame, 1, j])
    except: import pdb; pdb.set_trace()
    return lines
    
if __name__ == "__main__":

    args = parse_args()
    np.random.seed(args.seed)

    if args.simulation == "boid":
        width = 30
        sim = Boid(args.n_boids, width, 1/args.sample_freq, noise_var=0.01)

    else:
        raise ValueError("Simulation {} not implemented".format(args.simulation))
    
    suffix = "_" + args.simulation

    suffix += str(args.n_boids)
    if args.bat:
        suffix += '_bat'
    if args.partial:
        suffix += "_partial" 
    if args.avoid:
        suffix += "_avoid" 

    if args.length != 10000:
        suffix += "_l" + str(args.length) + "_Fs" + str(args.sample_freq)

    print(suffix)

    print("Generating {} training simulations".format(args.num_train))

    if True:
        T = args.length*args.sample_freq # args.length
        args.train=1
        loc_train, vel_train, edges_res_train, edges_train  = generate_dataset(
            args, width, sim,
            args.num_train, 
            T,
            args.sample_freq, train=args.train
        )

        np.save(os.path.join(args.datadir, "loc_train" + suffix + ".npy"), loc_train)
        np.save(os.path.join(args.datadir, "vel_train" + suffix + ".npy"), vel_train)
        np.savez(os.path.join(args.datadir, "edges_train" + suffix + ".npz"), edges_res_train,edges_train)

    T = args.length_test*args.sample_freq # args.length_test
    print("Generating {} validation simulations".format(args.num_valid))
    args.train = 0
    loc_valid, vel_valid, edges_res_valid, edges_valid = generate_dataset(
        args, width, sim,
        args.num_valid,
        T,
        args.sample_freq, train=args.train, 
    )

    np.save(os.path.join(args.datadir, "loc_valid" + suffix + ".npy"), loc_valid)
    np.save(os.path.join(args.datadir, "vel_valid" + suffix + ".npy"), vel_valid)
    np.savez(os.path.join(args.datadir, "edges_valid" + suffix + ".npz"), edges_res_valid, edges_valid)

    print("Generating {} test simulations".format(args.num_test))
    args.train = -1
    loc_test, vel_test, edges_res_test, edges_test = generate_dataset(
        args, width, sim,
        args.num_test,
        T,
        args.sample_freq, train=args.train,
    )

    if not os.path.exists(args.datadir):
        os.makedirs(args.datadir)

    np.save(os.path.join(args.datadir, "loc_test" + suffix + ".npy"), loc_test)
    np.save(os.path.join(args.datadir, "vel_test" + suffix + ".npy"), vel_test)
    np.savez(os.path.join(args.datadir, "edges_test" + suffix + ".npz"), edges_res_test, edges_test)

