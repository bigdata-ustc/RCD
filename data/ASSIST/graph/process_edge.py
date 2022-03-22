K_Directed = ''
K_Undirected = ''
edge = []
with open('knowledgeGraph.txt', 'r') as f:
    for i in f.readlines():
        i = i.replace('\n', '').split('\t')
        src = i[0]
        tar = i[1]
        edge.append((src, tar))
visit = []
for e in edge:
    if e not in visit:
        if (e[1],e[0]) in edge:
            K_Undirected += str(e[0] + '\t' + e[1] + '\n')
            visit.append(e)
            visit.append((e[1],e[0]))
        else:
            K_Directed += str(e[0] + '\t' + e[1] + '\n')
            visit.append(e)

with open('K_Directed.txt', 'w') as f:
    f.write(K_Directed)
with open('K_Undirected.txt', 'w') as f:
    f.write(K_Undirected)
all = len(visit)
print(all)
