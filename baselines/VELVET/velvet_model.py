import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.Dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, 1)
        
        self.rnn_pool = nn.GRU(input_size=768,
                                hidden_size=768,
                                num_layers=1,
                                batch_first=True)
        self.func_dense = nn.Linear(hidden_dim, hidden_dim)
        self.func_out_proj = nn.Linear(hidden_dim, 2)
        
    def forward(self, hidden):        
        x = self.Dropout(hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.Dropout(x)
        x = self.out_proj(x)
        
        out, func_x = self.rnn_pool(hidden)
        func_x = func_x.squeeze(0)
        func_x = self.Dropout(func_x)
        func_x = self.func_dense(func_x)
        func_x = torch.tanh(func_x)
        func_x = self.Dropout(func_x)
        func_x = self.func_out_proj(func_x)
        return x.squeeze(-1), func_x

class DenseGATConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        # TODO Add support for edge features.
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, 1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, 1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj, mask=None,add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]

        H, C = self.heads, self.out_channels
        B, N, _ = x.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1.0

        x = self.lin(x).view(B, N, H, C)  # [B, N, H, C]

        alpha_src = torch.sum(x * self.att_src, dim=-1)  # [B, N, H]
        alpha_dst = torch.sum(x * self.att_dst, dim=-1)  # [B, N, H]

        alpha = alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2)  # [B, N, N, H]

        # Weighted and masked softmax:
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float('-inf'))
        alpha = alpha.softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1))
        out = out.movedim(1, 2)  # [B,N,H,C]

        if self.concat:
            out = out.reshape(B, N, H * C)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

class Model(nn.Module):   
    def __init__(self, xfmr, tokenizer, args, hidden_dim=768, num_labels=155):
        super(Model, self).__init__()
        self.word_embedding = xfmr.embeddings.word_embeddings
        self.xfmr = xfmr
        self.gnn = DenseGATConv(768, 768)
        self.gnn_2 = DenseGATConv(768, 768)
        self.fc = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.tokenizer = tokenizer
        self.args = args
        # CLS head 
        self.classifier = ClassificationHead(hidden_dim=hidden_dim)

    def forward(self, input_ids_with_pattern, statement_mask, labels=None, func_labels=None, adj=None):
        statement_mask = statement_mask[:, :self.args.num_labels]
        if self.training:
            embed = self.word_embedding(input_ids_with_pattern)
            inputs_embeds = torch.mean(embed, dim=2)
            inputs_embeds = inputs_embeds[:, :self.args.num_labels]
            # transformer
            rep_xfmr = self.xfmr(inputs_embeds=inputs_embeds, attention_mask=statement_mask).last_hidden_state
            
            g_rep = self.gnn(inputs_embeds, adj)
            g_rep = self.gnn_2(g_rep, adj)
            
            logits, func_logits = self.classifier(rep_xfmr)
            logits_gnn, func_logits_gnn = self.classifier(g_rep)
            
            loss_fct = nn.CrossEntropyLoss()
            statement_loss = loss_fct(logits, labels)
            func_loss = loss_fct(func_logits, func_labels)
            statement_loss_gnn = loss_fct(logits_gnn, labels)
            func_loss_gnn = loss_fct(func_logits_gnn, func_labels)
            statement_loss = statement_loss + statement_loss_gnn
            func_loss = func_loss+func_loss_gnn
            return statement_loss, func_loss
        else:
            embed = self.word_embedding(input_ids_with_pattern)
            inputs_embeds = torch.mean(embed, dim=2)
            
            rep_xfmr = self.xfmr(inputs_embeds=inputs_embeds, attention_mask=statement_mask).last_hidden_state
            
            g_rep = self.gnn(inputs_embeds, adj)
            g_rep = self.gnn_2(g_rep, adj)
            
            logits, func_logits = self.classifier(rep_xfmr)
            logits_gnn, func_logits_gnn = self.classifier(g_rep)
            
            probs = torch.sigmoid(logits)
            func_probs = torch.softmax(func_logits, dim=-1)
            
            probs_gnn = torch.sigmoid(logits)
            func_probs_gnn = torch.softmax(func_logits, dim=-1)
            
            probs = probs+probs_gnn
            func_probs = func_probs+func_probs_gnn
            return probs, func_probs