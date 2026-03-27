#!/usr/bin/python3
import numpy as np
import sys
import time
np.set_printoptions(threshold=sys.maxsize,linewidth=300)

if __name__ == "__main__":
    def default_produce_FREYA_standalone_data(FREYA_data_path):
        #=================================================================================#
        #                                               LOAD FREYA DATA
        #================================================================================#
        # [0]=Z, [1]=A, [2]=E*, [3]=Ekin, [4]=TKE, [5]=#n, [6]=#g
        ff_data = np.loadtxt(fname=FREYA_data_path,comments='#',usecols=(1,2,3,4,5,7,8))
        # BECAUSE EACH FRAGMENT HAS A ROW OF ITS OWN (2 ROWS/FISSION EVENT)
        num_fission_events = len(ff_data[:,0])/2   
        unique_A, counts_unique_A = np.unique(ff_data[:,1],return_counts=True)
        yield_vec = np.zeros((len(unique_A),2))
        yield_vec[:,0] = unique_A
        yield_vec[:,1] = np.divide(counts_unique_A,num_fission_events)

        # NUBAR AND GUBAR
        nubar = np.sum(ff_data[:,5],axis=0)/num_fission_events
        gubar = np.sum(ff_data[:,6],axis=0)/num_fission_events
        print(f'Freya nubar = {nubar}, FREYA gubar = {gubar}')
        
        # NUBAR VS A
        An_vec = ff_data[:,[1,5]]
        N_vs_A_vec = np.zeros((len(unique_A),2))
        N_vs_A_vec[:,0] = unique_A
        for i,A in enumerate(unique_A):
            ind = np.where(An_vec[:,0] == A)[0]
            N_vs_A_vec[i,1] = np.mean(An_vec[ind,1])
        avg_nubar = np.average(N_vs_A_vec[:,1],weights=yield_vec[:,1])
        print(f'avg_nubar from N_vs_A:{avg_nubar}')

        # GUBAR VS A
        Ag_vec = ff_data[:,[1,6]]
        G_vs_A_vec = np.zeros((len(unique_A),2))
        G_vs_A_vec[:,0] = unique_A
        for i,A in enumerate(unique_A):
            ind = np.where(Ag_vec[:,0] == A)[0]
            G_vs_A_vec[i,1] = np.mean(Ag_vec[ind,1])
        avg_gubar = np.average(G_vs_A_vec[:,1],weights=yield_vec[:,1])

        print(f'avg_gubar from G_vs_A:{avg_gubar}')
        

        # PFNS & PFGS
        # CREATE ARRAYS FOR En AND Eg
        max_n = np.int16(np.max(ff_data[:,5]))
        max_g = np.int16(np.max(ff_data[:,6]))
        En_vec = np.zeros((len(ff_data[:,0]),max_n))
        Eg_vec = np.zeros((len(ff_data[:,0]),max_g))
        # READ AND TRUNCATE DATA
        with open(file=FREYA_data_path,mode='r') as f:
            event_data = f.readlines()[24:]
        n_g_data = [line.strip().split()[7:] for line in event_data]
        for i,line in enumerate(n_g_data):
            num_n = int(line[0])
            num_g = int(line[1])
            En_vec[i,0:num_n] = line[2:2+num_n]
            Eg_vec[i,0:num_g] = line[2+num_n:2+num_n+num_g]
        # PREPARE FOR PLOTTING
        En_vec = En_vec.flatten()
        En_vec = En_vec[np.nonzero(En_vec)]
        avg_neutron_E = np.round(np.sum(En_vec)/(num_fission_events*nubar),3)

        Eg_vec = Eg_vec.flatten()
        Eg_vec = Eg_vec[np.nonzero(Eg_vec)]
        avg_gamma_E = np.round(np.sum(Eg_vec)/(num_fission_events*gubar),3)

        # PRINT avg(En) AND avg(Eg)
        print(f'Freya avg(En) = {avg_neutron_E}, Freya avg(Eg) = {avg_gamma_E}')

        result_dict = {'Yield_data':(yield_vec,),'Nubar_data':(N_vs_A_vec,nubar),
                       'Gubar_data':(G_vs_A_vec,gubar),'En_data':(En_vec,avg_neutron_E),
                       'Eg_data':(Eg_vec,avg_gamma_E)}
        return result_dict


    #------------------------------------------- Opt Investigation --------------------------------------------------------------#
    data_path = "/home/Karlsson/Doktorandkurser/AdvancedPythonProgramming/Project/Project_APWP.out"
    #data_path = "/mnt/c/Users/petka376/PhD_Work/NewSimulationResults/TALYS/TALYS_FREYA/263_ExpTAL20FREYALLNL_2ndOpt_U236E253_1M/263_FREYA_event_by_event_file/263_FREYALLNL_Tuckey2ndSetOptParamU236E253_1M.out"

    #=============================== DEFAULT ===========================#
    
    start = time.perf_counter()
    default_produce_FREYA_standalone_data(FREYA_data_path=data_path)
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")
    
    #=============================== RESULTS DEFAULT ===========================#
    # Default 1e6 script: Elapsed: 10.2640 seconds
    # Default 1e7 script: Elapsed: 67.9062 seconds
    #===================================================================#