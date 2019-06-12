import numpy as np
import matplotlib.pyplot as plt


def plotaKmeans(dataset, r, k):
    coresAleatorias = np.random.random((k, 3))
    cores = r.dot(coresAleatorias)
    plt.scatter(dataset[:,0], dataset[:,1], c=cores)
    plt.show()


def inicializaCentroides(dataset, k):
    numAmostras, numAtributos = dataset.shape
    centroides = np.zeros((k, numAtributos))
    listaIndices = []
    for i in range(k):
        indice = np.random.choice(numAmostras)
        while indice in listaIndices:
            indice = np.random.choice(numAmostras)
        listaIndices.append(indice)
        centroides[i] = dataset[indice]
    return centroides

def atualizaCentroides(dataset, r, k):
    N, D = dataset.shape
    centroides = np.zeros((k, D))
    for i in range(k):
        centroides[i] = r[:, i].dot(dataset) / r[:, i].sum()
    return centroides


def custo_func(dataset, r, centroides, k):
    custo = 0
    for i in range(k):
        norm = np.linalg.norm(dataset - centroides[i], 2)
        custo += (norm * np.expand_dims(r[:, i], axis=1) ).sum()
    return custo


def cluster_responsibilities(centroides, dataset, beta):
    N = dataset.shape[0]
    K, D = centroides.shape
    R = np.zeros((N, K))

    for i in range(N):        
        R[i] = np.exp(-beta * np.linalg.norm(centroides - dataset[i], 2, axis=1)) 
    R /= R.sum(axis=1, keepdims=True)

    return R

def Kmeans(dataset, k, max_iters=1000, beta=1.):
    centroides = inicializaCentroides(dataset, k)
    custoAnt = 0
    for _ in range(max_iters):
        r = cluster_responsibilities(centroides, dataset, beta)
        centroides = atualizaCentroides(dataset, r, k)
        custo = custo_func(dataset, r, centroides, k)
        if np.abs(custo - custoAnt) < 1e-5:
            break
        custoAnt = custo
        
    plotaKmeans(dataset, r, k)


def geraAmostras(std=1, dim=2, dist=4):
    mu0 = np.array([0,0])
    mu1 = np.array([dist, dist])
    mu2 = np.array([0, dist])
    Nc = 300
    x0 = np.random.randn(Nc, dim) * std + mu0
    x1 = np.random.randn(Nc, dim) * std + mu1
    x2 = np.random.randn(Nc, dim) * std + mu2
    dataset = np.concatenate((x0, x1, x2), axis=0)
    return dataset
    

dataset = geraAmostras()
k = 3
Kmeans(dataset, k)
