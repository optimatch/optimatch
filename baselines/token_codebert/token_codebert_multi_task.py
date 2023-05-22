import torch
import torch.nn as nn


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
        hidden = hidden[:, 0, :]
        hidden = self.Dropout(hidden)
        x = self.dense(hidden)
        x = torch.tanh(x)
        x = self.Dropout(x)
        x = self.out_proj(x)
        
        func_x = self.func_dense(hidden)
        func_x = torch.tanh(func_x)
        func_x = self.Dropout(func_x)
        func_x = self.func_out_proj(func_x)
        return x.squeeze(-1), func_x

class Model(nn.Module):   
    def __init__(self, roberta, tokenizer, args, hidden_dim=768):
        super(Model, self).__init__()
        self.word_embedding = roberta.embeddings.word_embeddings
        self.roberta = roberta
        self.tokenizer = tokenizer
        self.args = args
        # CLS head
        self.classifier = ClassificationHead(hidden_dim=hidden_dim)

    def forward(self, input_ids, labels=None, func_labels=None):
        rep = self.roberta(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        if self.training:
            logits, func_logits = self.classifier(rep)
            loss_fct = nn.CrossEntropyLoss()
            statement_loss = loss_fct(logits, labels)
            loss_fct_2 = nn.CrossEntropyLoss()
            func_loss = loss_fct_2(func_logits, func_labels)
            return statement_loss, func_loss
        else:
            logits, func_logits = self.classifier(rep)
            probs = torch.sigmoid(logits)
            func_probs = torch.softmax(func_logits, dim=-1)
            return probs, func_probs