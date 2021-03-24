import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
# from p5 import Vector, stroke, circle
import warnings
warnings.simplefilter('error')

class Boid(object):
    def __init__(self, n_boids, width, delta_T, r_orientation= 10, noise_var=0.0):
        # self.max_force = 0.3
        self.speed = 1 # 1 4 
        self.r_repulsion = 1 # 2 
        self.r_repulsion2 = 10
        self.r_orientation = 2 # r_orientation # 10
        self.r_attraction = 8 # 10 (all) default:15 
        self.n_boids = n_boids
        self._delta_T = delta_T
        self.beta = 0.8727*0.6*self._delta_T # Maximum turning angle # 3*np.pi#

        self.width = width
        self.noise_var = noise_var
        # self.length = 1 # length of boids 0.5
        
    '''def __init__(
        self,
        
        box_size=5.0,
        loc_std=0.5,
        vel_norm=0.5,
        noise_var=0.0,
    ):
        self.n_boids = n_boids
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.noise_var = noise_var

        self._boid_types = np.array([0.0, 0.5, 1.0])
        
        # self._max_F = 0.1 / self._delta_T'''


    def _clamp(self, loc, vel):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hitting walls and returning after
            elastically colliding with walls
        """
        assert np.all(loc < self.width * 3)
        assert np.all(loc > -self.width * 3)

        over = loc > self.width
        loc[over] = 2 * self.width - loc[over]
        assert np.all(loc <= self.width)

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.width
        loc[under] = -2 * self.width - loc[under]
        # assert (np.all(vel[under] < 0))
        assert np.all(loc >= -self.width)
        vel[under] = np.abs(vel[under])

        return loc, vel
 
    def get_edges(self,args):
        if args.partial and args.avoid:
            edges = np.random.choice(
                np.array([-1., 0., 1.]), size=(self.n_boids, self.n_boids), p=np.array([1/5, 2/5, 2/5])
            )
        elif args.partial and not args.avoid:
            ratio = 3/4 ; value_neg = 0 
            edges = np.random.choice(
                np.array([value_neg, 1.0]), size=(self.n_boids, self.n_boids), p=np.array([1-ratio, ratio])
            )
        else:
            edges = np.ones((self.n_boids, self.n_boids))

        np.fill_diagonal(edges, 0)
        return edges

    def sample_trajectory(
        self,
        args, i_sim,
        T=10000,
        sample_freq=1,
        edges=None,
    ):
        n = self.n_boids
        assert T % sample_freq == 0
        # T_save = int(T / sample_freq )
        T_save = int(T / sample_freq - 1)
        counter = 0
        counter2 = 0

        if edges is None or args.bat:
            edges = self.get_edges(args)
        sd = 0.2
        speed = self.speed * sd*np.random.rand(n) 
        r_r = self.r_repulsion + sd*np.random.rand(n) 
        r_r2 = self.r_repulsion2 + sd*np.random.rand(n) 
        r_o = self.r_orientation + sd*np.random.rand(n) 
        r_a = self.r_attraction + sd*np.random.rand(n) 
        beta = self.beta

        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        edges_res = np.zeros((T_save, n, n))
        edges_all = np.zeros((T, n, n))

        if args.bat:
            rand_p = np.random.randn(1, n) * 2* np.pi
            loc_next = (15 + 10 * np.random.rand(2, n)) * np.array([np.cos(rand_p),np.sin(rand_p)]).squeeze() 
            loc_next[0] = loc_next[0] -20
            vel_next = np.array([np.abs(np.cos(rand_p/4)),np.sin(rand_p/4)]).squeeze() * self.speed
        else:
            rand_p = np.random.randn(1, n) * 2* np.pi
            loc_next = (6 + 10 * np.random.rand(2, n)) * np.array([np.cos(rand_p),np.sin(rand_p)]).squeeze() 
            vel_next = np.array([-np.sin(rand_p),np.cos(rand_p)]).squeeze() * self.speed 

        loc_next += vel_next
        # loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)
        res_prev = np.zeros((n,n))

        # run 
        for t in range(1, T):
            res = np.zeros((n,n))

            loc_next, vel_next = self._clamp(loc_next, vel_next)
            
            if t % sample_freq == 0:
                loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                counter += 1
                

            # apply_behaviour 
            di = np.zeros((2,n))
            vel_ = vel_next.copy()
            for i in range(n):
                di[:,i], repul_flag, res = self.repulsion(loc_next, vel_next, i, edges, r_r, r_r2, res, args)
                #if False: # repul_flag:
                #    vel_next[:,i] = di[:,i]  
                #if t == T -1 and i==4:
                #    import pdb; pdb.set_trace()
                if not repul_flag:
                    di[:,i], oa_flag, res = self.orient_attract(loc_next, vel_next, i, edges, r_r, r_o, r_a, res, res_prev, args)
                    #if False: # oa_flag > 0:
                    #    vel_next[:,i] = di[:,i]  
                

                # turn angle limitation
                signum = np.sign(vel_next[0,i]*di[1,i]-di[0,i]*vel_next[1,i]) # Compute direction of the angle
                dotprod = np.dot(di[:,i],vel_next[:,i])  # Dotproduct (needed for angle)
                if np.linalg.norm(di[:,i]) > 1e-10:
                    cos_theta = dotprod/np.linalg.norm(di[:,i])/np.linalg.norm(vel_next[:,i])
                    if np.abs(cos_theta) <= 1: 
                        try: phi = np.real(signum*np.arccos(cos_theta)) # Compute angle
                        except: import pdb; pdb.set_trace()
                    else: 
                        phi = 0.0
                else: 
                    phi = 0.0

                try: 
                    if abs(phi) <= beta:                                                    
                        vel_[:,i] = np.matmul(np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]),vel_next[:,i])           
                    elif phi < beta:                                                                  
                        vel_[:,i] = np.matmul(np.array([[np.cos(-beta),-np.sin(-beta)],[np.sin(-beta),np.cos(-beta)]]),vel_next[:,i])
                    else:                                                                                                
                        vel_[:,i] = np.matmul(np.array([[np.cos(beta),-np.sin(beta)],[np.sin(beta),np.cos(beta)]]),vel_next[:,i])
                except: import pdb; pdb.set_trace()

            #if args.train==0 and i_sim==1
            #    import pdb; pdb.set_trace()
            # update
            vel_next = vel_
            # vel_next += np.random.randn(2, self.n_boids) * self.noise_var
            
            loc_next += vel_next*self._delta_T 
            #if t-1 % sample_freq == 0:
            #    try: edges_res[counter] = res
            #    except: import pdb; pdb.set_trace()
            try: edges_res[counter2] += res/sample_freq
            except: import pdb; pdb.set_trace()
            edges_all[t] = res
            res_prev = res.copy()
            if t % sample_freq == sample_freq-1 and t > sample_freq:
                counter2 += 1

            #limit
            #if np.linalg.norm(vel_next) > self.max_speed:
            #    vel_next = vel_next / np.linalg.norm(vel_next) * self.max_speed
        # Add noise to observations
        # loc += np.random.randn(T_save, 2, self.n_boids) * self.noise_var
        # vel += np.random.randn(T_save, 2, self.n_boids) * self.noise_var

        if args.bat:
            idx_p = np.where(edges_res>1e-4)
            idx_n = np.where(edges_res<-1e-4)
            idx_0 = np.where((edges_res<1e-4) & (edges_res>-1e-4))
            edges_res[idx_p[0],idx_p[1],idx_p[2]] = 1
            edges_res[idx_n[0],idx_n[1],idx_n[2]] = -1
            edges_res[idx_0[0],idx_0[1],idx_0[2]] = 0
            edges_res[0,:,:] = 0
            edges_result = edges_res
        else:
            edges_result = edges
        #if args.train == 0:
        #    import pdb; pdb.set_trace()
        return loc, vel, edges_result, edges

    def repulsion(self, loc, vel, i, edges, r, r2, res, args):
        total = 0
        avg_vector = np.zeros(2)
        for j in range(self.n_boids):
            if args.avoid:
                flag = (edges[i,j] != 0)
                r_ = r if edges[i,j] == 1 else r2
            else:
                flag = (edges[i,j] == 1) # (i != j) if args.avoid_all else
                r_ = r
            if flag: 
                distance = np.linalg.norm(loc[:,j] - loc[:,i])
                if distance < r_[i]:
                    diff = loc[:,j] - loc[:,i]
                    diff /= distance
                    avg_vector += diff
                    total += 1
                    res[i,j] += -1

        if total > 0:
            steering = -avg_vector/total
            repul_flag = True
        else: 
            steering = avg_vector
            repul_flag = False

        return steering, repul_flag, res


    def orient_attract(self, loc, vel, i, edges, r_r, r_o, r_a, res, res_prev, args):
        
        total_o = 0
        avg_vector = np.zeros(2)
        total_a = 0
        center_of_mass = np.zeros(2)

        for j in range(self.n_boids):
            if edges[i,j] == 1: 
                dist = np.linalg.norm(loc[:,j] - loc[:,i])
                if dist >= r_r[i] and dist < r_o[i]: # orientation
                    avg_vector += vel[:,j]
                    total_o += 1
                    if res_prev[i,j]==1: # attraction -> orientation
                       res[i,j] += -0.5
                    elif res_prev[i,j]==-1: # repulsion -> orientation
                        res[i,j] += 0.5
                    else: # initial or continuing or unknown
                        res[i,j] += 0

                elif dist >= r_o[i] and dist < r_a[i]: # attraction
                    center_of_mass += loc[:,j]
                    total_a += 1
                    res[i,j] += 1


        if total_o > 0:
            steering_o = avg_vector/total_o
        else:
            steering_o = avg_vector

        if total_a > 0:
            center_of_mass /= total_a
            vec_to_com = center_of_mass - loc[:,i]
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) 
            steering_a = vec_to_com
        else:
            steering_a = np.zeros(2)

        if total_o > 0 and total_a > 0:
            steering = (steering_o + steering_a)/2
        else:
            steering = steering_o + steering_a

        return steering, (total_o + total_a), res

