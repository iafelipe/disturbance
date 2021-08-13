# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import default_rng
import pickle
import events

# informações dos vizinhos (L = 50, R = 3)
with open('neighbors_L=50_R=3.txt', 'rb') as handle:
    neighbors_info = pickle.loads(handle.read())


def scenario_4(p, pr, rec_time, total_dist, native_migration_rate, exotic_migration_rate,
               inicial_disturbance_clustered=False, q00=None,
               matrix_size=(50, 50), total_num_generations=100, alfa=0.8, beta=0.8, r_n=1, r_e=1,
               inicial_native_population=500, inicial_patch_quality=2,
               exotic_individuals_to_introduce=1000, seed=12456789):
    """
    Cenário 4: Invasão Biológica em Paisagens Pós-Distúrbio que Passam por Eventos de Restauração e
    Eventos de Distúrbio (eventos de distúrbio aleatórios)

    Eventos específicos:
    --------------------
    t = 1: distúrbio inicial

    t = 1: invasão da sp. exótica

    a cada {rec_time}: evento de restauração

    em {total_dist} t aleatórios: evento de distúrbio

    Parameters
    ----------
    p : float
        Intensidade do distúrbio.
    
    pr : float
        Intensidade da restauração.
    
    rec_time : int
        Tempo entre os eventos de restauração.
    
    total_dist : int
        Número total de eventos de distúrbio.
    
    native_migration_rate : float
        Taxa de migração da sp. nativa.
    
    exotic_migration_rate : float
        Taxa de migração da sp. exótica.
    
    inicial_disturbance_clustered : bool, optional
        Se o distúrbio inicial acontece de acordo com o padrão agregado. The default is False.
    
    q00 : float, optional
        Nível de correlação entre os patches disturbados. Necessário caso inicial_disturbance_clustered=True.
    
    matrix_size : (int, int)
        Tamanho da paisagem. Por enquanto só funciona com (50, 50).
    
    total_num_generations : int, optional
        Número total de gerações. The default is 100.
    
    alfa : float, optional
        Efeito de um indivíduo da sp. exótica sobre um indivíduo da sp. nativa. The default is 0.8.
    
    beta : float, optional
        Efeito de um indivíduo da sp. nativa sobre um indivíduo da sp. exótica. The default is 0.8.
    
    r_n : integer or real, optional
        Taxa de crescimento intrínseca da sp. nativa. The default is 1.
    
    r_e : integer or real, optional
        Taxa de crescimento intrínseca da sp. exótica. The default is 1.
    
    inicial_native_population : integer or real, optional
        Número de indivíduos da sp. nativa inicial para todos os patches. The default is 500.
    
    inicial_patch_quality : int, optional
        ualidade inicial de todos os patches. The default is 2.
    
    exotic_individuals_to_introduce : integer or real, optional
        Número de indivíduos da sp. exótica que serão introduzidos na invasão biológica. The default is 1000.
    
    seed : int, optional
        Seed para gerar números pseudo-aleatórios. The default is 12456789.

    Returns
    -------
    stored_natpop : numpy array of shape (total_num_generations, matrix_size)
        Array com todos os valores da população nativa.
    
    stored_exopop : numpy array of shape (total_num_generations, matrix_size)
        Array com todos os valores da população exótica.
    
    stored_landscape : numpy array of shape (total_num_generations, matrix_size)
        Array com todos os registros da qualidade da paisagem.
    
    stored_mean_nat : numpy array of shape (total_num_generations, )
        Vetor com todas as médias de indivíduos da sp. nativa na paisagem.
    
    stored_mean_exo : numpy array of shape (total_num_generations, )
        Vetor com todas as médias de indivíduos da sp. exótica na paisagem.
    
    stored_generations : numpy array of shape (total_num_generations, )
        Vetor com todas as gerações

    """
    rng = rng = default_rng(seed)

    # t = 0    
    restoration_counter = 0
    disturbance_gens = rng.integers(2, 100, size=total_dist)
    landscape = np.full(matrix_size, inicial_patch_quality, dtype=int)
    native_population = np.full(matrix_size, inicial_native_population, dtype=float)
    exotic_population = np.zeros(matrix_size, dtype=float)
    
    # store
    stored_mean_nat = np.array([np.mean(native_population)])
    stored_mean_exo = np.array([np.mean(exotic_population)])
    stored_natpop = np.array([native_population])
    stored_exopop = np.array([exotic_population])
    stored_landscape = np.array([landscape])
    stored_generations = np.array([0])

    for gen in range(1, total_num_generations):
        # counters
        restoration_counter += 1
        
        # atualizar capacidade de suporte
        kn_array = events.kn_update(landscape)
        ke_array = events.ke_update(landscape)

        # lotka
        native_population = events.lotka_volterra(
            native_population, exotic_population, r_n, alfa, kn_array)
        exotic_population = events.lotka_volterra(
            exotic_population, native_population, r_e, beta, ke_array)
        
        # breque
        native_population = events.breque(native_population)
        exotic_population = events.breque(exotic_population)
        
        # calcular migrantes
        nat_migrantes = events.calc_migrantes(native_population, native_migration_rate)
        exo_migrantes = events.calc_migrantes(exotic_population, exotic_migration_rate)
               
        # migração
        native_population = events.migracao(rng, nat_migrantes, native_population, neighbors_info)
        exotic_population = events.migracao(rng, exo_migrantes, exotic_population, neighbors_info)
        
        # remover migrantes
        native_population = events.remove_migrantes(native_population, nat_migrantes)
        exotic_population = events.remove_migrantes(exotic_population, exo_migrantes)
        
        # t = 1
        if gen == 1:
            # distúrbio inicial
            if inicial_disturbance_clustered:
                landscape = events.clustered_disturbance(rng, landscape, p, q00, neighbors_info)
            else:
                landscape = events.random_disturbance(rng, landscape, p)
            
            # invasão
            exotic_population = events.invasion(rng, landscape, 
                                                exotic_population, exotic_individuals_to_introduce)
         
        # restoration event
        if restoration_counter == rec_time:
            landscape = events.restoration(rng, landscape, pr)
            restoration_counter = 0
            
        # disturbance event
        if np.any(gen == disturbance_gens):
            landscape = events.random_disturbance(rng, landscape, p)
        
        # campo médio
        nat_cm, native_population = events.campo_medio(native_population)
        exo_cm, exotic_population = events.campo_medio(exotic_population)
        
        # store
        stored_mean_nat = np.append(stored_mean_nat, [nat_cm], axis=0)
        stored_mean_exo = np.append(stored_mean_exo, [exo_cm], axis=0)
        stored_natpop = np.append(stored_natpop, [native_population], axis=0)
        stored_exopop = np.append(stored_exopop, [exotic_population], axis=0)
        stored_landscape = np.append(stored_landscape, [landscape], axis=0)
        stored_generations = np.append(stored_generations, [gen], axis=0)
    
    np.savez('output_scenario4.npz', stored_mean_nat, stored_mean_exo,
             stored_natpop, stored_exopop, stored_landscape, stored_generations)

# Exemplos:
scenario_4(p=0.5, pr=0.2, rec_time=5, total_dist=5, native_migration_rate=0.2, exotic_migration_rate=0.2)
scenario_4(p=0.5, pr=0.2, rec_time=5, total_dist=5, native_migration_rate=0.2, exotic_migration_rate=0.2,
           q00=1.0, inicial_disturbance_clustered=True)
