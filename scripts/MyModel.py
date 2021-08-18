"""
some codes in this script was based on
https:https://github.com/awslabs/dgl-lifesci
"""

import torch.nn as nn
from dgllife.model.gnn import GAT
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax


class AttentiveGRU1(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))


class ModifiedChargeModelNNV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedChargeModelNNV2, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats


class ModifiedChargeModelV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedChargeModelV2, self).__init__()

        self.gnn = ModifiedChargeModelNNV2(node_feat_size=node_feat_size,
                                           edge_feat_size=edge_feat_size,
                                           num_layers=num_layers,
                                           graph_feat_size=graph_feat_size,
                                           dropout=dropout)
        self.predict = nn.Sequential(
            nn.Linear(graph_feat_size, graph_feat_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return self.predict(sum_node_feats)


class ModifiedChargeModelV2New(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0., n_tasks=1):
        super(ModifiedChargeModelV2New, self).__init__()

        self.gnn = ModifiedChargeModelNNV2(node_feat_size=node_feat_size,
                                           edge_feat_size=edge_feat_size,
                                           num_layers=num_layers,
                                           graph_feat_size=graph_feat_size,
                                           dropout=dropout)
        self.predict = nn.Sequential(
            nn.Linear(graph_feat_size, graph_feat_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return self.predict(sum_node_feats)


class ModifiedChargeModelNNV3(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedChargeModelNNV3, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )
        self.sum_predictions = 0
        self.num_layers = num_layers

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_predictions = self.sum_predictions + self.predict(node_feats)
        return self.sum_predictions / (self.num_layers - 1)


class ModifiedChargeModelV3(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedChargeModelV3, self).__init__()

        self.gnn = ModifiedChargeModelNNV3(node_feat_size=node_feat_size,
                                            edge_feat_size=edge_feat_size,
                                            num_layers=num_layers,
                                            graph_feat_size=graph_feat_size,
                                            dropout=dropout)

    def forward(self, g, node_feats, edge_feats):
        predictions = self.gnn(g, node_feats, edge_feats)
        return predictions


class ModifiedGATPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None):
        super(ModifiedGATPredictor, self).__init__()

        self.gnn = GAT(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.predict = nn.Sequential(nn.Linear(gnn_out_feats, 1))

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        return self.predict(node_feats)


# class ModifiedChargeModel(nn.Module):
#     def __init__(self,
#                  node_feat_size,
#                  edge_feat_size,
#                  num_layers=2,
#                  graph_feat_size=200,
#                  dropout=0.):
#         super(ModifiedChargeModel, self).__init__()
#
#         self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
#                                   edge_feat_size=edge_feat_size,
#                                   num_layers=num_layers,
#                                   graph_feat_size=graph_feat_size,
#                                   dropout=dropout)
#         self.predict = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(graph_feat_size, 1)
#         )
#
#     def forward(self, g, node_feats, edge_feats):
#         node_feats = self.gnn(g, node_feats, edge_feats)
#         return self.predict(node_feats)


# incorporate both the node and edge features using Multilayer Perception
class AttentiveMLP1(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveMLP1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        # self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)
        self.MPL = nn.Sequential(
            nn.Linear(edge_hidden_size + node_feat_size, node_feat_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(node_feat_size, node_feat_size),
            nn.Dropout(dropout)
        )

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        # return F.relu(self.gru(context, node_feats))
        return F.relu(self.MPL(torch.cat([context, node_feats], dim=1)))


class AttentiveMLP2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveMLP2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        # self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)
        self.MPL = nn.Sequential(
            nn.Linear(edge_hidden_size + node_feat_size, node_feat_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(node_feat_size, node_feat_size),
            nn.Dropout(dropout)
        )

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        # return F.relu(self.gru(context, node_feats))
        return F.relu(self.MPL(torch.cat([context, node_feats], dim=1)))


class GetMLPContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetMLPContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        # self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
        #                                    graph_feat_size, dropout)
        self.attentive_mlp = AttentiveMLP1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_mlp(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNMLPLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNMLPLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        # self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.attentive_mlp = AttentiveMLP2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        # return self.bn_layer(self.attentive_gru(g, logits, node_feats))
        return self.bn_layer(self.attentive_mlp(g, logits, node_feats))


class GNNMLP(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(GNNMLP, self).__init__()

        self.init_context = GetMLPContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNMLPLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats


class GNNMLPPredictor(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(GNNMLPPredictor, self).__init__()

        self.gnn = GNNMLP(node_feat_size=node_feat_size,
                          edge_feat_size=edge_feat_size,
                          num_layers=num_layers,
                          graph_feat_size=graph_feat_size,
                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Linear(graph_feat_size, graph_feat_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return self.predict(sum_node_feats)
