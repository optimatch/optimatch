import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, emb_dim, dim_channel, kernel_wins, dropout_rate, args):
        super(TextCNN, self).__init__()
        self.args = args
        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, 
               emb_x,
               attention_mask):
        attention_mask = attention_mask.unsqueeze(-1).expand(emb_x.shape[0], emb_x.shape[1], emb_x.shape[2])
        emb_x = emb_x * attention_mask                 
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        return fc_x

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.Dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, 155)
        
        self.func_dense = nn.Linear(hidden_dim, hidden_dim)
        self.func_out_proj = nn.Linear(hidden_dim, 2)
        
    def forward(self, hidden):
        x = self.Dropout(hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.Dropout(x)
        x = self.out_proj(x)
        
        func_x = self.Dropout(hidden)
        func_x = self.func_dense(func_x)
        func_x = torch.tanh(func_x)
        func_x = self.Dropout(func_x)
        func_x = self.func_out_proj(func_x)
        return x.squeeze(-1), func_x

class Model(nn.Module):   
    def __init__(self, roberta, tokenizer, args, hidden_dim=768, num_labels=155):
        super(Model, self).__init__()
        self.word_embedding = roberta.embeddings.word_embeddings
        self.birnn = nn.LSTM(768, 768, num_layers=2, batch_first=True, bidirectional=True)
        self.tokenizer = tokenizer
        self.args = args
        # CLS head
        self.classifier = ClassificationHead(hidden_dim=768)

    def forward(self, input_ids_with_pattern, statement_mask, labels=None, func_labels=None):
        statement_mask = statement_mask[:, :self.args.num_labels]
        if self.training:
            embed = self.word_embedding(input_ids_with_pattern)
            inputs_embeds = torch.amax(embed, dim=2)
            inputs_embeds = inputs_embeds[:, :self.args.num_labels, :]        
            out, (hn, cn) = self.birnn(inputs_embeds)
            rep = hn[-1]  
            logits, func_logits = self.classifier(rep)
            loss_fct = nn.CrossEntropyLoss()
            statement_loss = loss_fct(logits, labels)
            loss_fct_2 = nn.CrossEntropyLoss()
            func_loss = loss_fct_2(func_logits, func_labels)
            return statement_loss, func_loss
        else:
            embed = self.word_embedding(input_ids_with_pattern)
            inputs_embeds = torch.amax(embed, dim=2)
            inputs_embeds = inputs_embeds[:, :self.args.num_labels, :]
            out, (hn, cn) = self.birnn(inputs_embeds)
            rep = hn[-1]
            logits, func_logits = self.classifier(rep)
            probs = torch.sigmoid(logits)
            func_probs = torch.softmax(func_logits, dim=-1)
            return probs, func_probs