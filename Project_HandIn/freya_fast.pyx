#! /usr/bin/env python3
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

def cython_produce_FREYA_standalone_data(str FREYA_data_path):

    cdef:
        cnp.ndarray[double, ndim=2] ff_data
        Py_ssize_t n_rows
        int num_fission_events

    # LOAD DATA
    ff_data = np.loadtxt(fname=FREYA_data_path,
                         comments='#',
                         usecols=(1,2,3,4,5,7,8))

    n_rows = ff_data.shape[0]
    num_fission_events = n_rows // 2

    # =========================
    # UNIQUE A + YIELDS
    # =========================
    unique_A, counts_unique_A = np.unique(ff_data[:,1], return_counts=True)

    yield_vec = np.zeros((len(unique_A), 2), dtype=np.float64)
    yield_vec[:,0] = unique_A
    yield_vec[:,1] = counts_unique_A / num_fission_events

    # =========================
    # NUBAR / GUBAR (fast)
    # =========================
    cdef double nubar = np.sum(ff_data[:,5]) / num_fission_events
    cdef double gubar = np.sum(ff_data[:,6]) / num_fission_events

    print(f'Freya nubar = {nubar}, FREYA gubar = {gubar}')

    # =========================
    # NUBAR vs A (optimized loop)
    # =========================
    cdef:
        Py_ssize_t i, j, nA = len(unique_A)
        double A
        double[:, :] data_view = ff_data
        double[:, :] N_vs_A_vec = np.zeros((nA, 2), dtype=np.float64)
        double sum_val
        int count

    for i in range(nA):
        A = unique_A[i]
        N_vs_A_vec[i,0] = A
        sum_val = 0.0
        count = 0

        for j in range(n_rows):
            if data_view[j,1] == A:
                sum_val += data_view[j,5]
                count += 1

        if count > 0:
            N_vs_A_vec[i,1] = sum_val / count

    avg_nubar = np.average(N_vs_A_vec[:,1], weights=yield_vec[:,1])
    print(f'avg_nubar from N_vs_A:{avg_nubar}')

    # =========================
    # GUBAR vs A (same pattern)
    # =========================
    cdef double[:, :] G_vs_A_vec = np.zeros((nA, 2), dtype=np.float64)

    for i in range(nA):
        A = unique_A[i]
        G_vs_A_vec[i,0] = A
        sum_val = 0.0
        count = 0

        for j in range(n_rows):
            if data_view[j,1] == A:
                sum_val += data_view[j,6]
                count += 1

        if count > 0:
            G_vs_A_vec[i,1] = sum_val / count

    avg_gubar = np.average(G_vs_A_vec[:,1], weights=yield_vec[:,1])
    print(f'avg_gubar from G_vs_A:{avg_gubar}')

    # =========================
    # ENERGY PART (still Python-heavy)
    # =========================
    max_n = int(np.max(ff_data[:,5]))
    max_g = int(np.max(ff_data[:,6]))

    En_vec = np.zeros((n_rows, max_n))
    Eg_vec = np.zeros((n_rows, max_g))

    with open(FREYA_data_path, 'r') as f:
        event_data = f.readlines()[24:]

    cdef list parts
    cdef int num_n, num_g

    for i in range(len(event_data)):
        parts = event_data[i].split()[7:]
        num_n = int(parts[0])
        num_g = int(parts[1])

        En_vec[i, 0:num_n] = parts[2:2+num_n]
        Eg_vec[i, 0:num_g] = parts[2+num_n:2+num_n+num_g]

    En_vec = En_vec.ravel()
    En_vec = En_vec[En_vec != 0]

    Eg_vec = Eg_vec.ravel()
    Eg_vec = Eg_vec[Eg_vec != 0]

    avg_neutron_E = np.sum(En_vec) / (num_fission_events * nubar)
    avg_gamma_E = np.sum(Eg_vec) / (num_fission_events * gubar)

    print(f'Freya avg(En) = {avg_neutron_E}, Freya avg(Eg) = {avg_gamma_E}')

    return {
        'Yield_data': (yield_vec,),
        'Nubar_data': (np.asarray(N_vs_A_vec), nubar),
        'Gubar_data': (np.asarray(G_vs_A_vec), gubar),
        'En_data': (En_vec, avg_neutron_E),
        'Eg_data': (Eg_vec, avg_gamma_E)
    }