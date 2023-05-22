import torch
import torch.nn as nn
from kmeans_pytorch import kmeans
import geomloss

class CrossAttention(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.k = nn.Linear(self.dim, self.dim, bias=False)
        self.v = nn.Linear(self.dim, self.dim, bias=False)
        self.o = nn.Linear(self.dim, self.dim, bias=False)        
    
    def forward(self, hidden_states, key_value_states):
        query_states = self.q(hidden_states)
        key_states = self.k(key_value_states)
        value_states = self.v(key_value_states)
        scores = torch.matmul(query_states, key_states.transpose(1, 0))
        # use minus attention score to align with argmin selection
        scores = -scores
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=0.1, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self.o(attn_output)
        return attn_output, attn_weights 

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_dim, args):
        super().__init__()
        self.args = args
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
        # statement prediction
        x = self.Dropout(hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.Dropout(x)
        x = self.out_proj(x)
        # function prediction
        out, func_x = self.rnn_pool(hidden)
        func_x = func_x.squeeze(0)
        func_x = self.Dropout(func_x)
        func_x = self.func_dense(func_x)
        func_x = torch.tanh(func_x)
        func_x = self.Dropout(func_x)
        func_x = self.func_out_proj(func_x)
        return x.squeeze(-1), func_x

class Model(nn.Module):   
    def __init__(self, t5, tokenizer, args, hidden_dim=768, codebook_hidden=192, num_clusters=100, codebook_initialized=False):
        super(Model, self).__init__()
        self.codebook_hidden = codebook_hidden
        self.word_embedding = t5.shared
        self.rnn_statement_embedding = nn.GRU(input_size=hidden_dim,
                                              hidden_size=hidden_dim,
                                              num_layers=1,
                                              batch_first=True)
        self.t5 = t5
        self.tokenizer = tokenizer
        self.args = args
        # CLS head
        self.classifier = ClassificationHead(hidden_dim=hidden_dim, args=args)
        # patterns related
        self.vul_pattern_representations = torch.zeros(7000, codebook_hidden)
        self.accumulate_idx = 0
        self.rnn_patterns_pooling = nn.GRU(input_size=hidden_dim,
                                            hidden_size=hidden_dim,
                                            num_layers=1,
                                            batch_first=True)
        self.cross_att = CrossAttention(codebook_hidden)
        self.num_clusters = num_clusters
        if args.phase_two_training:
            self.vp_codebook = nn.Embedding(100, codebook_hidden)
        else:
            self.vp_codebook = nn.Embedding(num_clusters, codebook_hidden)
        self.codebook_initialized = codebook_initialized
        self.low = nn.Linear(hidden_dim, codebook_hidden)
        self.high = nn.Linear(codebook_hidden, hidden_dim)
        self.layer_norm = nn.LayerNorm(codebook_hidden)

    def forward(self, input_ids_with_pattern, statement_mask, labels=None, func_labels=None, phase_one_training=False):
        if self.training:
            d_loss = None
            embed = self.word_embedding(input_ids_with_pattern)
            inputs_embeds = torch.amax(embed, dim=2)
            pat = inputs_embeds[:, self.args.max_num_statements:, :]
            out, pat = self.rnn_patterns_pooling(pat)
            pat = pat.permute(1, 0, 2)
            if phase_one_training:
                inputs_embeds = inputs_embeds[:, :self.args.max_num_statements, :]
                inputs_embeds = torch.cat((inputs_embeds, pat), dim=1)
                rep = self.t5(inputs_embeds=inputs_embeds, attention_mask=statement_mask).last_hidden_state
                h_function = rep[:, :self.args.max_num_statements, :]
                logits, func_logits = self.classifier(h_function)
                loss_fct = nn.CrossEntropyLoss()
                statement_loss = loss_fct(logits, labels)
                loss_fct_func = nn.CrossEntropyLoss()
                func_loss = loss_fct_func(func_logits, func_labels)            
                return statement_loss, func_loss, d_loss
            else:
                pat = self.low(pat)
                pat = self.layer_norm(pat)
                if not self.codebook_initialized:
                    # keep accumulating vul patterns before the first evaluation
                    for n in range(len(pat)):
                        # only accumulate vulnerable patterns
                        lab = labels[n]
                        if 1 in lab:
                            vul_pattern_rep = pat[n]
                            self.vul_pattern_representations[self.accumulate_idx, :] = vul_pattern_rep
                            self.accumulate_idx += 1
                    pat = self.high(pat)
                else:
                    ### Cross Attention ###
                    pat = pat.squeeze(1)
                    
                    ### selection ###
                    pat, att_scores = self.cross_att(pat, self.vp_codebook.weight)                    
                    selected_indices = torch.argmin(att_scores, dim=-1) 
                    ###
                      
                    # get quantized latent vectors
                    z_q = self.vp_codebook(selected_indices)
                            
                    # sinkhorn loss
                    d_loss_fct = geomloss.SamplesLoss(loss="sinkhorn")
                    d_loss = d_loss_fct(pat, self.vp_codebook.weight)

                    pat = self.high(pat)
                    z_q = self.high(z_q)
                    
                inputs_embeds = inputs_embeds[:, :self.args.max_num_statements, :]
                
                if not self.codebook_initialized:
                    inputs_embeds = torch.cat((inputs_embeds, pat), dim=1)
                else:
                    pat = pat.unsqueeze(1)
                    z_q = z_q.unsqueeze(1)
                    # preserve gradients
                    z_q = pat + (z_q - pat).detach()
                    inputs_embeds = torch.cat((inputs_embeds, z_q), dim=1)
    
                rep = self.t5(inputs_embeds=inputs_embeds, attention_mask=statement_mask).last_hidden_state
                            
                h_function = rep[:, :self.args.max_num_statements, :]
                
                logits, func_logits = self.classifier(h_function)
                loss_fct = nn.CrossEntropyLoss()
                statement_loss = loss_fct(logits, labels)
                loss_fct_func = nn.CrossEntropyLoss()
                func_loss = loss_fct_func(func_logits, func_labels)            
                return statement_loss, func_loss, d_loss
        else:
            embed = self.word_embedding(input_ids_with_pattern)
            inputs_embeds = torch.amax(embed, dim=2)
            
            rep = self.t5(inputs_embeds=inputs_embeds, attention_mask=statement_mask[:, :inputs_embeds.shape[1]]).last_hidden_state
            logits, func_logits = self.classifier(rep)
            if phase_one_training:
                probs = torch.sigmoid(logits)
                func_probs = torch.softmax(func_logits, dim=-1)
                return probs, func_probs
            else:
                if not self.codebook_initialized:
                    self.vp_codebook = nn.Embedding(self.num_clusters, self.codebook_hidden).to(self.args.device)
                self.codebook_initialized = True 
                # inference by taking average - statement-level
                all_prob = None
                all_func_prob = None
                            
                for z in range(len(self.vp_codebook.weight)):
                    vul_pattern = self.vp_codebook.weight[z]
                    vul_pattern = vul_pattern.unsqueeze(0).unsqueeze(0).expand(inputs_embeds.shape[0], 1, vul_pattern.shape[0]).to(self.args.device)
                    
                    vul_pattern = self.high(vul_pattern)
                    
                    input_embeddings = torch.cat((inputs_embeds, vul_pattern), dim=1)
                    rep = self.t5(inputs_embeds=input_embeddings, attention_mask=statement_mask).last_hidden_state
                    
                    h_function = rep[:, :self.args.max_num_statements, :]
    
                    logits, func_logits = self.classifier(h_function)
                    prob = torch.sigmoid(logits).detach().cpu()
                    prob = prob.unsqueeze(0)
                    func_prob = torch.softmax(func_logits, dim=-1)
                    func_prob = func_prob.unsqueeze(0)
                    if all_prob is None:
                        all_prob = prob
                        all_func_prob = func_prob
                    else:
                        all_prob = torch.cat((all_prob, prob), dim=0)
                        all_func_prob = torch.cat((all_func_prob, func_prob), dim=0)
                all_prob = torch.mean(all_prob, dim=0)
                all_func_prob = torch.amax(all_func_prob, dim=0)
                return all_prob, all_func_prob