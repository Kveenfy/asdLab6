import numpy as np

def floydWarshall(graph):
    N = len(graph)
    dist = np.copy(graph)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def tsp(dist):
    N = len(dist)
    visitMask = 2 ** N
    dp = np.ones((N, visitMask)) * np.inf
    for i in range(N):
        dp[i][0] = 0
    prev = np.zeros((N, visitMask), dtype=int)
    for mask in range(visitMask):
        for node in range(N):
            if dp[node][mask] == np.inf: continue
            for i in range(N):
                if int(mask / (2 ** i)) % 2: continue
                newDist = dp[node][mask] + dist[node][i]
                if newDist < dp[i][mask + 2 ** i]:
                    dp[i][mask + 2 ** i] = newDist
                    prev[i][mask + 2 ** i] = node

    minDist = min(dp[i][2 ** N - 1] for i in range(N))
    end = np.argmin(dp[:, 2 ** N - 1])

    path = []
    mask = 2 ** N - 1
    while mask:
        path.append(end)
        mask ^= 2 ** end
        end = prev[end][mask + 2 ** end]
    path.append(end)

    path.pop()
    for i in range(len(path)):
        path[i] += 1

    path = path[::-1]
    pathStr = ' -> '.join(str(node) for node in path)

    return minDist, pathStr

graph = np.array([
    [0, 4, 3, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
    [4, 0, np.inf, np.inf, 6, 3, np.inf, np.inf, np.inf, np.inf],
    [3, np.inf, 0, np.inf, 4, 7, np.inf, np.inf, np.inf, np.inf],
    [1, np.inf, np.inf, 0, 5, 6, np.inf, np.inf, np.inf, np.inf],
    [np.inf, 6, 4, 5, 0, np.inf, 11, 4, 3, np.inf],
    [np.inf, 3, 7, 6, np.inf, 0, 9, 6, 8, np.inf],
    [np.inf, np.inf, np.inf, np.inf, 11, 9, 0, np.inf, np.inf, 6],
    [np.inf, np.inf, np.inf, np.inf, 4, 6, np.inf, 0, np.inf, 4],
    [np.inf, np.inf, np.inf, np.inf, 3, 8, np.inf, np.inf, 0, 5],
    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 6, 4, 5, 0]
])
dist = floydWarshall(graph)
print(tsp(dist))
