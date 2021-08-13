# -*- coding: utf-8 -*-

import numpy as np


@np.vectorize
def kn_update(landscape):
    """
    retorna a capacidade de suporte da sp. nativa de acordo com a qualidade dos patches

    Parameters
    ----------
    landscape : numpy array
        array contendo a qualidade da paisagem

    Returns
    -------
    kn : numpy array
        array contendo a qualidade de suporte da sp. nativa de todos os patches

    """

    if landscape == 2 or landscape == 1:
        kn = 1000

    elif landscape == 0:
        kn = 500

    return kn


@np.vectorize
def ke_update(landscape):
    """
    retorna a capacidade de suporte da espécie exótica de acordo com a qualidade do patch

    Parameters
    ----------
    landscape : numpy array
        array contendo a qualidade da paisagem

    Returns
    -------
    ke : numpy array
        array contendo a qualidade de suporte da sp. exótica de todos os patches

    """
    
    if landscape == 2:
        ke = 500

    elif landscape == 1 or landscape == 0:
        ke = 1000

    return ke


@np.vectorize
def lotka_volterra(pop1, pop2, r, alfa_or_beta, k):
    """

    Parameters
    ----------
    pop1 : numpy array
        array contendo a população 1 de todos os patches

    pop2 : numpy array
        array contendo a população 2 de todos os patches

    r : integer or real
        taxa de crescimento intrínseca da sp. 1

    alfa_or_beta : float
        efeito de um indivíduo da sp. 2 sobre um indivíduo da sp. 1

    k : numpy array
        array contendo a qualidade de suporte da sp. 1 de todos os patches
        
    Returns
    -------
    updated_pop1 : numpy array
        array contendo a população 1 atualizada

    """
       
    updated_pop1 = pop1 * (1 + r * (1 - ((pop1 + alfa_or_beta * pop2) / k)))

    return updated_pop1


@np.vectorize
def breque(x):
    """ verifica valores próximos de 0 """
    
    if x < 0.001:
        return 0
    else:
        return x


@np.vectorize
def random_disturbance(rng, landscape, p):
    """
    distúrbio do padrão aleatório

    Parameters
    ----------
    rng : Generator
        gerador de números pseudo-aleatórios
        
    landscape : numpy array
        array contendo a qualidade da paisagem
        
    p : float
        intensidade do distúrbio

    Returns
    -------
    landscape : numpy array
        paisagem pós-distúrbio

    """
    
    if landscape > 0:
        generated_p = rng.random()
    
        if generated_p <= p:
            landscape = 0

    return landscape


@np.vectorize
def clustered_disturbance(rng, landscape, p, q00, neighbors_info, iterations=1000000):
    """
    distúrbio do padrão agregado
    versão em Python do algoritmo de Hiebeler (2000):
        Hiebeler, D. (2000). Populations on Fragmented Landscapes with Spatially Structured Heterogeneities: Landscape Generation and Local Dispersal. Ecology, 81(6), 1629-1641.

    Parameters
    ----------
    rng : Generator
        gerador de números pseudo-aleatórios
        
    landscape : numpy array
        array contendo a qualidade da paisagem
        
    p : float
        intensidade do distúrbio
        
    q00 : float
        nível de correlação entre os patches disturbados
        
    neighbors_info : dict
        dicionário contendo os vizinhos de cada patch
        
    iterations : int
        número de iterações desejada. padrão é 1000000.

    Returns
    -------
    landscape : numpy array
        paisagem pós-distúrbio

    """

    def desired_blocks(p, q00):
        """ calcula o número de blocos desejados """

        block_00 = int((p * q00) * (50 * 50) * 4)
        block_02 = int((p - block_00) * (50 * 50) * 4)
        block_22 = int((1 - (block_00 - (2 * block_02))) * (50 * 50) * 4)
        
        return block_00, block_02, block_22
    
    def count_blocks(landscape, neighbors_info):
        """ conta os blocos 2x1 da paisagem """

        count_00 = 0
        count_02 = 0
        count_22 = 0
        
        for i in range(50):
            for j in range(50):
                
                neighborhood = neighbors_info[i * 50 + j]
                neighborhood = neighborhood[neighborhood['euclid_dist'] == 1]
                
                for neighbor in range(np.size(neighborhood)):
                    i2 = neighborhood[neighbor]['xviz']
                    j2 = neighborhood[neighbor]['yviz']
                    
                    if landscape[i, j] == 0:
                        if landscape[i2, j2] == 0:
                            count_00 += 1
                        else:
                            count_02 += 1
                    else:
                        if landscape[i2, j2] == 0:
                            count_02 += 1
                        else:
                            count_22 += 1
        
        return count_00, count_02, count_22
     
    def d_value(desired_00, desired_02, desired_22, count_00, count_02, count_22):
        """ calcula o valor de d """
        
        d = abs(desired_00 - count_00) + 2 * abs(desired_02 - count_02) + abs(desired_22 - count_22)
        
        return d
    
    target = desired_blocks(p, q00)
    landscape = random_disturbance(rng, landscape, p)
    
    for _ in range(iterations):
        count = count_blocks(landscape, neighbors_info)
        d = d_value(*target, *count)
        random_i, random_j = rng.integers(50), rng.integers(50)
    
        temp_landscape = np.copy(landscape)
        if temp_landscape[random_i, random_j] == 0:
            temp_landscape[random_i, random_j] = 1
        else:
            temp_landscape[random_i, random_j] = 0
        
        temp_count = count_blocks(temp_landscape, neighbors_info)
        temp_d = d_value(*target, *temp_count)
        
        if temp_d < d:
            landscape[random_i, random_j] = temp_landscape[random_i, random_j]
        
    return landscape


@np.vectorize
def restoration(rng, landscape, pr):
    """
    evento de restauração da paisagem

    Parameters
    ----------
    rng : Generator
        gerador de números pseudo-aleatórios
        
    landscape : numpy array
        array contendo a qualidade da paisagem
        
    pr : float
        intensidade da restauração

    Returns
    -------
    landscape : numpy array
        paisagem após a restauração dos patches

    """
    if landscape < 2:
        generated_pr = rng.random()
        
        if generated_pr <= pr:
            landscape += 1
            
    return landscape
        

def invasion(rng, landscape, exopop, exotic_individuals_to_introduce):
    """
    invasão biológica da espécie exótica

    Parameters
    ----------
    rng : Generator
        gerador de números pseudo-aleatórios
        
    landscape : numpy array
        array contendo a qualidade da paisagem
        
    exopop : numpy array
        array contendo a população exótica de todos os patches
        
    exotic_individuals_to_introduce : integer or real
        número de indivíduos da sp. exótica que serão introduzidos na invasão biológica

    Returns
    -------
    exopop : numpy array
        array contendo a população exótica atualizada
        
    """
    
    disturbed_patches = []
    it = np.nditer(landscape, flags=['multi_index'])
    for x in it:
        if x == 0:
            disturbed_patches.append(it.multi_index)

    if len(disturbed_patches) >= 1:
        while exotic_individuals_to_introduce > 0:

            next_x, next_y = disturbed_patches[rng.integers(len(disturbed_patches))]
            exopop[next_x, next_y] += 1
            exotic_individuals_to_introduce -= 1

    return exopop


def campo_medio(pop):
    """
    calcula o campo médio das espécies além de rodar o breque

    Parameters
    ----------
    pop : numpy array
        array contendo a população

    Returns
    -------
    cm: float
        campo médio
        
    pop: numpy array
        array contendo a população

    """
    cm = breque(np.mean(pop))

    if cm == 0:
        pop = np.zeros((50, 50), dtype=float)

    return cm, pop


@np.vectorize
def calc_migrantes(pop, migration_rate):
    """
    calcula os migrantes

    Parameters
    ----------
    pop : numpy array
        array contendo a população
        
    migration_rate : float
        porcentagem de migrantes da população

    Returns
    -------
    migrantes: numpy array
        array contendo os migrantes
        
    """
    
    return pop * migration_rate


def migracao(rng, migrantes, pop, neighbors_info):
    """
    migração das espécies entre os patches

    Parameters
    ----------
    rng : Generator
        gerador de números pseudo-aleatórios
        
    migrantes: numpy array
        array contendo os migrantes
        
    pop : numpy array
        array contendo a população
        
    neighbors_info : dict
        dicionário contendo os vizinhos de cada patch

    Returns
    -------
    pop : numpy array
        array contendo a população atualizada após a migração

    """

    it = np.nditer(migrantes, flags=['multi_index'])
    for x in it:
        neighbors = neighbors_info[it.multi_index[0] * 50 + it.multi_index[1]]
        current_migrantes = np.copy(x)
           
        while current_migrantes >= 1:
            current_neighbor = neighbors[rng.integers(neighbors.shape[0])]
            pop[current_neighbor['xviz'], current_neighbor['yviz']] += 10
            current_migrantes -= 10

    return pop


@np.vectorize
def remove_migrantes(pop, migrantes):
    """
    retira os migrantes de seus patches antigos

    Parameters
    ----------
    pop : numpy array
        array contendo a população
        
    migrantes: numpy array
        array contendo os migrantes

    Returns
    -------
    pop : numpy array
        array contendo a população atualizada sem os migrantes
        
    """
    
    return pop - migrantes
