import torch
import torch.nn as nn

from layers.retention import MultiScaleRetention

class Model(nn.Module):
    def __init__(self, configs, double_v_dim=False):
        super(Model, self).__init__()
        self.configs = configs
        self.layers = configs.e_layers
        self.hidden_dim = configs.d_model
        self.ffn_size = configs.d_ff
        self.heads = configs.n_heads
        self.v_dim = configs.d_model * 2 if double_v_dim else configs.d_model
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.retentions = nn.ModuleList([
            MultiScaleRetention(configs.d_model, configs.n_heads, double_v_dim)
            for _ in range(configs.e_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.d_model)
            )
            for _ in range(configs.e_layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(configs.d_model)
            for _ in range(configs.e_layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(configs.d_model)
            for _ in range(configs.e_layers)
        ])
        self.channel_proj = nn.Sequential(
                nn.Linear(configs.enc_in, configs.d_model),
                nn.Dropout(configs.dropout),
                nn.GELU(),
            )
        self.channel_proj_invers = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, configs.enc_in)
            )
        self.temp_proj = nn.Sequential(
                nn.Linear(configs.seq_len, configs.d_ff),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.pred_len)
            )    
        # Revin
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

    def forward(self, X, batch_x_mark, dec_inp, batch_y_mark):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = self.channel_proj(X)

        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        X = self.channel_proj_invers(X)

        Y = self.temp_proj(X.transpose(2, 1)).transpose(2, 1)
        
        return Y