import json
import numpy as np

exer_n = 17746
knowledge_n = 123
student_n = 4163
edge_dic_deno = {}

def constructDependencyMatrix():
    data_file = '../log_data_all.json'
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)

    # Calculate correct matrix
    knowledgeCorrect = np.zeros([knowledge_n, knowledge_n])
    for student in data:
        if student['log_num'] < 2:
            continue
        log = student['logs']
        for log_i in range(student['log_num'] - 1):
            if log[log_i]['score'] * log[log_i + 1]['score'] == 1:
                for ki in log[log_i]['knowledge_code']:
                    for kj in log[log_i + 1]['knowledge_code']:
                        if ki != kj:
                            # n_{ij}
                            knowledgeCorrect[ki - 1][kj - 1] += 1.0
                            # n_{i*}, calculate the number of correctly answering i
                            if ki - 1 in edge_dic_deno.keys():
                                edge_dic_deno[ki - 1] += 1
                            else:
                                edge_dic_deno[ki - 1] = 1

    s = 0
    c = 0
    # Calculate transition matrix
    knowledgeDirected = np.zeros([knowledge_n, knowledge_n])
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if i != j and knowledgeCorrect[i][j] > 0:
                    knowledgeDirected[i][j] = float(knowledgeCorrect[i][j]) / edge_dic_deno[i]
                    s += knowledgeDirected[i][j]
                    c += 1
    o = np.zeros([knowledge_n, knowledge_n])
    min_c = 100000
    max_c = 0
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if knowledgeCorrect[i][j] > 0 and i != j:
                min_c = min(min_c, knowledgeDirected[i][j])
                max_c = max(max_c, knowledgeDirected[i][j])
    s_o = 0
    l_o = 0
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if knowledgeCorrect[i][j] > 0 and i != j:
                o[i][j] = (knowledgeDirected[i][j] - min_c) / (max_c - min_c)
                l_o += 1
                s_o += o[i][j]
    avg = s_o / l_o #total / count
    # avg = 0.02
    avg *= avg
    avg *= avg
    # avg is threshold
    graph = ''
    edge = np.zeros([knowledge_n, knowledge_n])
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if o[i][j] >= avg:
                graph += str(i) + '\t' + str(j) + '\n'
                edge[i][j] = 1
    e_l = []
    co = 0
    tr = 0
    all = 0
    # Calculate concept dependency relation
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if (i,j) not in e_l:
                e_l.append((i,j))
                if edge[i][j] == 1:
                    if edge[j][i] == 1:
                        co += 1
                        all += 1
                        e_l.append((j, i))
                    else:
                        tr += 1
                        all += 1
    with open('knowledgeGraph.txt', 'w') as f:
        f.write(graph)

if __name__ == '__main__':
    constructDependencyMatrix()