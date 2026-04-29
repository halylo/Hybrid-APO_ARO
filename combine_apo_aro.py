import numpy as np
import pandas as pd
import csv
import os
from scipy.special import gamma
from benchmark import benchmark_functions, set_bounds, N, T, dim, benchmark_names, SpaceBound, Levy

# Initialize populations
for idx in range(6):  # Chỉ chạy 6 hàm
    lb, ub = set_bounds(idx, dim)

PopPos_init = np.random.rand(N, dim) * (ub - lb) + lb
PopPos_init1 = PopPos_init.copy()  # For APO
PopPos_init2 = PopPos_init.copy()  # For PA1 (AOA)
PopPos_init3 = PopPos_init.copy()  # For PA2 (COA)
PopPos_init4 = PopPos_init.copy()  # For PA3 (EFO)
PopPos_init5 = PopPos_init.copy()  # For PA4 (ARO_APO)
PopPos_init6 = PopPos_init.copy()  # For PA5 (PSO)

# ARO Algorithm
def ARO(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='aro_results.csv'):
    PopPos2 = PopPos_init.copy()
    fitness_drawARO = []
    pop_fit = np.array([fobj(PopPos2[i, :]) for i in range(N)])

    best_f = float('inf')
    best_x = None
    for i in range(N):
        if pop_fit[i] <= best_f:
            best_f = pop_fit[i]
            best_x = PopPos2[i, :]

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for it in range(T):
        direct1 = np.zeros((N, dim))
        direct2 = np.zeros((N, dim))
        theta = 2 * (1 - (it + 1) / T)
        for i in range(N):
            L = (np.e - np.exp((((it + 1) - 1) / T) ** 2)) * np.sin(2 * np.pi * np.random.rand())
            rd = np.floor(np.random.rand() * dim)
            rand_dim = np.random.permutation(dim)
            direct1[i, rand_dim[:int(rd)]] = 1
            c = direct1[i, :]
            R = L * c
            A = 2 * np.log(1 / np.random.rand()) * theta
            if A > 1:
                K = np.r_[0:i, i + 1:N]
                RandInd = K[np.random.randint(0, N - 1)]
                newPopPos = PopPos2[RandInd, :] + R * (PopPos2[i, :] - PopPos2[RandInd, :]) + \
                            0.5 * (0.05 + np.random.rand()) * np.random.randn()
            else:
                ttt = int(np.floor(np.random.rand() * dim))
                direct2[i, ttt] = 1
                gr = direct2[i, :]
                H = ((T - (it + 1) + 1) / T) * np.random.randn()
                b = PopPos2[i, :] + H * gr * PopPos2[i, :]
                newPopPos = PopPos2[i, :] + R * (np.random.rand() * b - PopPos2[i, :])

            newPopPos = SpaceBound(newPopPos, ub, lb)
            newPopFit = fobj(newPopPos)
            if newPopFit < pop_fit[i]:
                pop_fit[i] = newPopFit
                PopPos2[i, :] = newPopPos

            if pop_fit[i] < best_f:
                best_f = pop_fit[i]
                best_x = PopPos2[i, :]

        fitness_drawARO.append(best_f)

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': it + 1,
                'Best_Fitness': best_f
            })

    return fitness_drawARO, PopPos2, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(fitness_drawARO)]

# APO Algorithm
def APO(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='apo_results.csv'):
    if PopPos_init is None:
        PopPos = np.random.rand(N, dim) * (ub - lb) + lb
    else:
        PopPos = PopPos_init.copy()
    PopPos1 = PopPos.copy()
    fitness_drawAPO = []
    PopFit = np.array([fobj(PopPos1[i, :]) for i in range(N)])

    BestF = float('inf')
    BestX = None
    for i in range(N):
        if PopFit[i] <= BestF:
            BestF = PopFit[i]
            BestX = PopPos1[i, :]

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for It in range(T):
        rand = np.random.rand()
        for i in range(N):
            theta1 = (1 - It / T)
            B = 2 * np.log(1 / rand) * theta1

            if B > 0.5:
                for _ in range(T):
                    K = [j for j in range(N) if j != i]
                    if not K:
                        break
                    RandInd = np.random.choice(K)
                    step1 = PopPos[i] - PopPos[RandInd]
                    if np.linalg.norm(step1) != 0:
                        break
                else:
                    continue
                R = 0.5 * (0.05 + rand) * np.random.normal(0, 1)
                Y = PopPos1[i, :] + 0.01 * Levy(dim) * step1 + R
                step2 = (rand - 0.5) * np.pi
                S = np.tan(step2)
                Z = Y * S

                Y = SpaceBound(Y, ub, lb)
                Z = SpaceBound(Z, ub, lb)
                NewPop = np.array([Y, Z])
                NewPopFit = np.array([fobj(Y), fobj(Z)])
                sorted_indexes = np.argsort(NewPopFit)
                newPopPos = NewPop[sorted_indexes[0], :]
            else:
                F = 0.5
                K = [j for j in range(N) if j != i]
                for _ in range(T):
                    available_indices = [j for j in range(N) if j != i]
                    if len(available_indices) < 3:
                        continue
                    RandInd = np.random.choice(available_indices, 3, replace=False)
                    step1 = PopPos[RandInd[1]] - PopPos[RandInd[2]]
                    if np.linalg.norm(step1) != 0:
                        break
                    else:
                        continue

                if rand < 0.5:
                    W = PopPos1[RandInd[0], :] + F * step1
                else:
                    W = PopPos1[RandInd[0], :] + F * 0.01 * Levy(dim) * step1
                f = 0.1 * (rand - 1) * ((T - It) / T)
                Y = (1 + f) * W
                for _ in range(T):
                    rand_leader_index1 = np.random.randint(0, N)
                    rand_leader_index2 = np.random.randint(0, N)
                    X_rand1 = PopPos1[rand_leader_index1, :]
                    X_rand2 = PopPos1[rand_leader_index2, :]
                    step2 = X_rand1 - X_rand2
                    if np.linalg.norm(step2) != 0 and not np.array_equal(X_rand1, X_rand2):
                        break
                    else:
                        continue
                Epsilon = np.random.uniform(0, 1)
                if rand < 0.5:
                    Z = PopPos1[i, :] + Epsilon * step2
                else:
                    Z = PopPos1[i, :] + F * 0.01 * Levy(dim) * step2

                W = SpaceBound(W, ub, lb)
                Y = SpaceBound(Y, ub, lb)
                Z = SpaceBound(Z, ub, lb)
                NewPop = np.array([W, Y, Z])
                NewPopFit = np.array([fobj(W), fobj(Y), fobj(Z)])
                sorted_indexes = np.argsort(NewPopFit)
                newPopPos = NewPop[sorted_indexes[0], :]

            newPopPos = SpaceBound(newPopPos, ub, lb)
            newPopFit = fobj(newPopPos)
            if newPopFit < PopFit[i]:
                PopFit[i] = newPopFit
                PopPos1[i, :] = newPopPos

        for i in range(N):
            if PopFit[i] < BestF:
                BestF = PopFit[i]
                BestX = PopPos1[i, :]

        fitness_drawAPO.append(BestF)

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': It + 1,
                'Best_Fitness': BestF
            })

    return fitness_drawAPO, PopPos1, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(fitness_drawAPO)]

# AOA Algorithm (PA1)
def initial_variables(size, min_values, max_values, target_function, dim, start_init=None):
    min_values = np.array([min_values] * dim) if np.isscalar(min_values) else np.array(min_values)
    max_values = np.array([max_values] * dim) if np.isscalar(max_values) else np.array(max_values)
    
    if start_init is not None:
        start_init = np.atleast_2d(start_init)
        n_rows = size - start_init.shape[0]
        if n_rows > 0:
            rows = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = np.array([target_function(ind) for ind in start_init])
        population = np.hstack((start_init, fitness_values[:, np.newaxis]))
    else:
        population = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = np.array([target_function(ind) for ind in population])
        population = np.hstack((population, fitness_values[:, np.newaxis]))
    return population

def update_population(population, elite, mu, moa, mop, min_values, max_values, target_function):
    e = 2.2204e-16
    dim = len(min_values)
    p = np.copy(population)
    r1 = np.random.rand(population.shape[0], dim)
    r2 = np.random.rand(population.shape[0], dim)
    r3 = np.random.rand(population.shape[0], dim)
    update_1 = np.where(r1 > moa, elite[:-1] / (mop + e) * ((max_values - min_values) * mu + min_values), elite[:-1])
    update_2 = np.where(r2 <= 0.5, update_1 * mop, update_1 - mop)
    update_3 = np.where(r3 > 0.5, update_2 - ((max_values - min_values) * mu + min_values), update_2 + ((max_values - min_values) * mu + min_values))
    up_pos = np.clip(update_3, min_values, max_values)
    for i in range(population.shape[0]):
        new_fitness = target_function(up_pos[i, :])
        if new_fitness < population[i, -1]:
            p[i, :-1] = up_pos[i, :]
            p[i, -1] = new_fitness
    return p

def AOA(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='PA1_aoa.csv'):
    population = initial_variables(N, lb, ub, fobj, dim, PopPos_init)
    best_f = float('inf')
    best_x = None
    elite = np.copy(population[population[:, -1].argsort()][0, :])
    if elite[-1] < best_f:
        best_f = elite[-1]
        best_x = elite[:-1].copy()

    curve = np.zeros(T)
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub)

    for it in range(T):
        moa = 0.2 + it * ((1 - 0.2) / T)
        mop = 1 - ((it ** (1 / 0.5)) / (T ** (1 / 0.5)))
        population = update_population(population, elite, 5, moa, mop, lb, ub, fobj)
        current_best = population[population[:, -1].argsort()][0, :]
        if current_best[-1] < best_f:
            best_f = current_best[-1]
            best_x = current_best[:-1].copy()
            elite = np.copy(current_best)

        curve[it] = best_f
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': it + 1,
                'Best_Fitness': best_f
            })

    return best_f, best_x, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]

# COA Algorithm (PA2)
def COA(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='PA2_coa.csv', n_coy=5, n_packs=20):
    VarMin = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb)
    VarMax = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub)

    if n_coy < 3:
        raise ValueError('At least 3 coyotes per pack!')
    p_leave = 0.005 * n_coy**2
    Ps = 1 / dim
    pop_total = n_packs * n_coy
    coyotes = PopPos_init if PopPos_init is not None and PopPos_init.shape == (pop_total, dim) else \
              np.tile(VarMin, (pop_total, 1)) + np.random.rand(pop_total, dim) * (np.tile(VarMax, (pop_total, 1)) - np.tile(VarMin, (pop_total, 1)))
    costs = np.zeros(pop_total)
    ages = np.zeros(pop_total)
    packs = np.arange(pop_total).reshape(n_packs, n_coy)

    for c in range(pop_total):
        costs[c] = fobj(coyotes[c, :])

    best_f = np.min(costs)
    best_idx = np.argmin(costs)
    best_x = coyotes[best_idx, :].copy()

    curve = np.zeros(T)
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for it in range(T):
        for p in range(n_packs):
            pack_indices = packs[p, :]
            coyotes_aux = coyotes[pack_indices, :]
            costs_aux = costs[pack_indices]
            ages_aux = ages[pack_indices]
            inds = np.argsort(costs_aux)
            coyotes_aux = coyotes_aux[inds, :]
            costs_aux = costs_aux[inds]
            ages_aux = ages_aux[inds]
            c_alpha = coyotes_aux[0, :]
            tendency = np.median(coyotes_aux, axis=0)
            new_coyotes = np.zeros((n_coy, dim))
            for c in range(n_coy):
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(0, n_coy)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(0, n_coy)
                new_c = coyotes_aux[c, :] + np.random.rand() * (c_alpha - coyotes_aux[rc1, :]) + \
                        np.random.rand() * (tendency - coyotes_aux[rc2, :])
                new_coyotes[c, :] = np.minimum(np.maximum(new_c, VarMin), VarMax)
                new_cost = fobj(new_coyotes[c, :])
                if new_cost < costs_aux[c]:
                    costs_aux[c] = new_cost
                    coyotes_aux[c, :] = new_coyotes[c, :]

            parents = np.random.permutation(n_coy)[:2]
            prob1 = (1 - Ps) / 2
            prob2 = prob1
            pdr = np.random.permutation(dim)
            p1 = np.zeros(dim)
            p2 = np.zeros(dim)
            p1[pdr[0]] = 1
            p2[pdr[1]] = 1
            r = np.random.rand(dim - 2)
            p1[pdr[2:]] = r < prob1
            p2[pdr[2:]] = r > 1 - prob2
            p1 = p1.astype(bool)
            p2 = p2.astype(bool)
            n = ~(p1 | p2)
            pup = p1 * coyotes_aux[parents[0], :] + p2 * coyotes_aux[parents[1], :] + \
                  n * (VarMin + np.random.rand(dim) * (VarMax - VarMin))
            pup_cost = fobj(pup)
            worst = np.where(pup_cost < costs_aux)[0]
            if len(worst) > 0:
                older = np.argsort(ages_aux[worst])[::-1]
                which = worst[older[0]]
                coyotes_aux[which, :] = pup
                costs_aux[which] = pup_cost
                ages_aux[which] = 0

            coyotes[pack_indices, :] = coyotes_aux
            costs[pack_indices] = costs_aux
            ages[pack_indices] = ages_aux

        if n_packs > 1 and np.random.rand() < p_leave:
            rp = np.random.permutation(n_packs)[:2]
            rc = np.random.randint(0, n_coy, size=2)
            aux = packs[rp[0], rc[0]]
            packs[rp[0], rc[0]] = packs[rp[1], rc[1]]
            packs[rp[1], rc[1]] = aux

        ages += 1
        current_best_idx = np.argmin(costs)
        current_best_f = costs[current_best_idx]
        if current_best_f < best_f:
            best_f = current_best_f
            best_x = coyotes[current_best_idx, :].copy()

        curve[it] = best_f
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': it + 1,
                'Best_Fitness': best_f
            })

    return best_f, best_x, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]

# EFO Algorithm (PA3)
def insert_in_pop(cpopulation, nparticle, position):
    newpop = np.vstack((cpopulation[:position-1, :], nparticle, cpopulation[position-1:-1, :]))
    return newpop

def EFO(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='PA3_efo.csv', R_rate=0.3, Ps_rate=0.85, P_field=0.1, N_field=0.45):
    phi = (1 + np.sqrt(5)) / 2
    em_pop = PopPos_init if PopPos_init is not None and PopPos_init.shape == (N, dim) else \
             np.tile(lb, (N, 1)) + np.random.rand(N, dim) * (np.tile(ub, (N, 1)) - np.tile(lb, (N, 1)))
    fit = np.array([fobj(ind) for ind in em_pop])
    em_pop = np.hstack((em_pop, fit[:, np.newaxis]))
    em_pop = em_pop[np.argsort(em_pop[:, dim]), :]

    best_f = em_pop[0, dim]
    best_x = em_pop[0, :dim].copy()

    curve = np.zeros(T)
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    r_index1 = np.random.randint(0, int(np.round(N * P_field)), (dim, T))
    r_index2 = np.random.randint(int(np.round(N * (1 - N_field))), N, (dim, T))
    r_index3 = np.random.randint(int(np.round(N * P_field)) + 1, int(np.round(N * (1 - N_field))) - 1, (dim, T))
    ps = np.random.rand(dim, T)
    r_force = np.random.rand(T)
    rp = np.random.rand(T)
    randomization = np.random.rand(T)

    RI = 0
    new_emp = np.zeros(dim + 1)
    for it in range(T):
        r = r_force[it]
        for i in range(dim):
            if ps[i, it] > Ps_rate:
                new_emp[i] = em_pop[r_index3[i, it], i] + phi * r * (em_pop[r_index1[i, it], i] - em_pop[r_index3[i, it], i]) + \
                             r * (em_pop[r_index3[i, it], i] - em_pop[r_index2[i, it], i])
            else:
                new_emp[i] = em_pop[r_index1[i, it], i]
            lb_i = lb[i] if isinstance(lb, np.ndarray) else lb
            ub_i = ub[i] if isinstance(ub, np.ndarray) else ub
            if new_emp[i] >= ub_i or new_emp[i] <= lb_i:
                new_emp[i] = lb_i + (ub_i - lb_i) * randomization[it]

        if rp[it] < R_rate:
            lb_i = lb[RI] if isinstance(lb, np.ndarray) else lb
            ub_i = ub[RI] if isinstance(ub, np.ndarray) else ub
            new_emp[RI] = lb_i + (ub_i - lb_i) * randomization[it]
            RI = (RI + 1) % dim

        new_emp[dim] = fobj(new_emp[:dim])
        if new_emp[dim] < em_pop[N-1, dim]:
            position = np.where(em_pop[:, dim] > new_emp[dim])[0][0]
            em_pop = insert_in_pop(em_pop, new_emp, position + 1)

        if em_pop[0, dim] < best_f:
            best_f = em_pop[0, dim]
            best_x = em_pop[0, :dim].copy()

        curve[it] = best_f
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': it + 1,
                'Best_Fitness': best_f
            })

    return best_f, best_x, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]

# ARO_APO Algorithm (PA4)
def ARO_APO(N, T, lb, ub, dim, fobj, function_name, PopPos_ARO=None, csv_filename='PA4_aro_apo.csv'):
    if PopPos_ARO is None:
        PopPos_ARO = np.random.rand(N, dim) * (ub - lb) + lb
    PopFit_ARO = np.array([fobj(ind) for ind in PopPos_ARO])
    global_best_fitness = float('inf')
    global_best_position = None
    for i in range(N):
        if PopFit_ARO[i] <= global_best_fitness:
            global_best_fitness = PopFit_ARO[i]
            global_best_position = PopPos_ARO[i].copy()

    fitness_log = []
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness', 'Algorithm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for i in range(T):
        if i % 2 == 0:
            _, PopPos_ARO, curve = APO(N, 1, lb, ub, dim, fobj, function_name, PopPos_ARO, csv_filename)
            algorithm = 'APO'
        else:
            _, PopPos_ARO, curve = ARO(N, 1, lb, ub, dim, fobj, function_name, PopPos_ARO, csv_filename)
            algorithm = 'ARO'

        PopFit_ARO = np.array([fobj(ind) for ind in PopPos_ARO])
        best_idx = np.argmin(PopFit_ARO)
        best_fitness = PopFit_ARO[best_idx]
        if best_fitness < global_best_fitness:
            global_best_fitness = best_fitness
            global_best_position = PopPos_ARO[best_idx].copy()

        if i % 5 == 0:
            fitness_log.append({
                'Iteration': i + 1,
                'Best_Fitness': global_best_fitness
            })

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': i + 1,
                'Best_Fitness': global_best_fitness,
                'Algorithm': algorithm
            })

    return global_best_fitness, global_best_position, fitness_log

# PSO Algorithm (PA5)
def PSO(N, T, lb, ub, dim, fobj, function_name, PopPos_init, csv_filename='PA5_pso.csv'):
    PopPos = PopPos_init if PopPos_init is not None and PopPos_init.shape == (N, dim) else \
             np.random.rand(N, dim) * (ub - lb) + lb
    PopVel = np.random.rand(N, dim) * (ub - lb)
    PopFit = np.array([fobj(ind) for ind in PopPos])

    PBestPos = PopPos.copy()
    PBestFit = PopFit.copy()
    GBestFit = float('inf')
    GBestPos = None
    for i in range(N):
        if PopFit[i] <= GBestFit:
            GBestFit = PopFit[i]
            GBestPos = PopPos[i].copy()

    w = 0.9 - 0.5 * (np.arange(T) / T)
    c1 = 2
    c2 = 2
    curve = np.zeros(T)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ['Function', 'Iteration', 'Best_Fitness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for it in range(T):
        for i in range(N):
            r1, r2 = np.random.rand(), np.random.rand()
            PopVel[i] = (w[it] * PopVel[i] + 
                         c1 * r1 * (PBestPos[i] - PopPos[i]) + 
                         c2 * r2 * (GBestPos - PopPos[i]))
            newPopPos = PopPos[i] + PopVel[i]
            newPopPos = SpaceBound(newPopPos, ub, lb)
            newPopFit = fobj(newPopPos)
            if newPopFit < PBestFit[i]:
                PBestFit[i] = newPopFit
                PBestPos[i] = newPopPos.copy()
            PopPos[i] = newPopPos
            PopFit[i] = newPopFit

        for i in range(N):
            if PopFit[i] < GBestFit:
                GBestFit = PopFit[i]
                GBestPos = PopPos[i].copy()

        curve[it] = GBestFit
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Function': function_name,
                'Iteration': it + 1,
                'Best_Fitness': GBestFit
            })

    return GBestFit, GBestPos, [{'Iteration': i + 1, 'Best_Fitness': f} for i, f in enumerate(curve)]

# Run functions
def run_apo_adjusted():
    all_results = []
    for idx in range(6):
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = APO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init1)
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('apo.csv', index=False)

def run_aro_adjusted():
    all_results = []
    for idx in range(6):
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = ARO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init)
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('aro.csv', index=False)

def run_aoa():  # PA1
    all_results = []
    for idx in range(6):
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = AOA(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init2)
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA1_aoa.csv', index=False)

def run_coa():  # PA2
    all_results = []
    for idx in range(6):
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = COA(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init3)
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA2_coa.csv', index=False)

def run_efo():  # PA3
    all_results = []
    for idx in range(6):
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = EFO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init4)
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA3_efo.csv', index=False)

def run_aro_apo():  # PA4
    all_results = []
    for idx in range(6):
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = ARO_APO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init5)
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA4_aro_apo.csv', index=False)

def run_pso():  # PA5
    all_results = []
    for idx in range(6):
        lb, ub = set_bounds(idx, dim)
        fobj = lambda x: benchmark_functions(x, idx)
        _, _, fitness_log = PSO(N, T, lb, ub, dim, fobj, benchmark_names[idx], PopPos_init6)
        df = pd.DataFrame(fitness_log)
        df['Function'] = benchmark_names[idx]
        all_results.append(df)
    pd.concat(all_results).to_csv('PA5_pso.csv', index=False)

if __name__ == "__main__":
    if os.path.exists('apo.csv'):
        os.remove('apo.csv')
    if os.path.exists('aro.csv'):
        os.remove('aro.csv')
    if os.path.exists('PA1_aoa.csv'):
        os.remove('PA1_aoa.csv')
    if os.path.exists('PA2_coa.csv'):
        os.remove('PA2_coa.csv')
    if os.path.exists('PA3_efo.csv'):
        os.remove('PA3_efo.csv')
    if os.path.exists('PA4_aro_apo.csv'):
        os.remove('PA4_aro_apo.csv')
    if os.path.exists('PA5_pso.csv'):
        os.remove('PA5_pso.csv')

    run_apo_adjusted()
    run_aro_adjusted()
    run_aoa()  # PA1
    run_coa()  # PA2
    run_efo()  # PA3
    run_aro_apo()  # PA4
    run_pso()  # PA5