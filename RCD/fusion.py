import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphLayer import GraphLayer

class Fusion(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        # graph structure
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.u_from_e = local_map['u_from_e'].to(self.device)
        self.e_from_u = local_map['e_from_u'].to(self.device)

        super(Fusion, self).__init__()

        self.directed_gat = GraphLayer(self.directed_g, args.knowledge_n, args.knowledge_n)
        self.undirected_gat = GraphLayer(self.undirected_g, args.knowledge_n, args.knowledge_n)

        self.k_from_e = GraphLayer(self.k_from_e, args.knowledge_n, args.knowledge_n)  # src: e
        self.e_from_k = GraphLayer(self.e_from_k, args.knowledge_n, args.knowledge_n)  # src: k

        self.u_from_e = GraphLayer(self.u_from_e, args.knowledge_n, args.knowledge_n)  # src: e
        self.e_from_u = GraphLayer(self.e_from_u, args.knowledge_n, args.knowledge_n)  # src: u

        self.k_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        k_directed = self.directed_gat(kn_emb)
        k_undirected = self.undirected_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        k_from_e_graph = self.k_from_e(e_k_graph)
        e_from_k_graph = self.e_from_k(e_k_graph)

        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_graph = self.u_from_e(e_u_graph)
        e_from_u_graph = self.e_from_u(e_u_graph)

        # update concepts
        A = kn_emb
        B = k_directed
        C = k_undirected
        D = k_from_e_graph[self.exer_n:]
        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        concat_c_3 = torch.cat([A, D], dim=1)
        score1 = self.k_attn_fc1(concat_c_1)
        score2 = self.k_attn_fc2(concat_c_2)
        score3 = self.k_attn_fc3(concat_c_3)
        score = F.softmax(torch.cat([torch.cat([score1, score2], dim=1), score3], dim=1),
                          dim=1)  # dim = 1, 按行SoftMax, 行和为1
        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C + score[:, 2].unsqueeze(1) * D

        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0: self.exer_n]
        C = e_from_u_graph[0: self.exer_n]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)  # dim = 1, 按行SoftMax, 行和为1
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # updated students
        all_stu_emb = all_stu_emb + u_from_e_graph[self.exer_n:]

        return kn_emb, exer_emb, all_stu_emb
