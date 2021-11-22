import torch as t


class GRUAttnNet(t.nn.Module):
    def __init__(self, embed_dim, hidden_dim, hidden_layers, dropout=0, device: t.device = 'cpu'):
        super(GRUAttnNet, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.device = device

        self.build_model(dropout)
        self.out_dim = hidden_dim

    def build_model(self, dropout):
        self.attn_layer = t.nn.Sequential(
            t.nn.Linear(self.hidden_dim, self.hidden_dim),
            t.nn.ReLU(inplace=True)
        ).to(self.device)
        self.gru_layer = t.nn.GRU(self.embed_dim, self.hidden_dim, self.hidden_layers, bidirectional=True, batch_first=True, dropout=dropout).to(self.device)
        self.gru_layer_attn_w = t.nn.GRU(self.embed_dim, self.hidden_dim, self.hidden_layers, bidirectional=True, batch_first=True, dropout=dropout).to(self.device)
        # self.gru_layer = t.nn.LSTM(self.embed_dim, self.hidden_dim, self.hidden_layers, bidirectional=True, batch_first=True, dropout=dropout).to(self.device)

    def attn_net_with_w(self, rnn_out, rnn_hn, neighbor_mask: t.Tensor, x):
        """
        :param rnn_out: [batch_size, seq_len, n_hidden * 2]
        :param rnn_hn: [batch_size, num_layers * num_directions, n_hidden]
        :return:
        """
        neighbor_mask_dim = neighbor_mask.unsqueeze(2)
        neighbor_mask_dim = neighbor_mask_dim.repeat(1, 1, self.hidden_dim)
        neighbor_mask_dim = neighbor_mask_dim.cuda()
        lstm_tmp_out = t.chunk(rnn_out, 2, -1)  # 把最后一维度分成两份
        # h [batch_size, time_step(seq_len), hidden_dims] 把两层的结果叠加？
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        assert h.shape == neighbor_mask_dim.shape
        h = t.where(neighbor_mask_dim == 1, h, neighbor_mask_dim)
        # 计算权重
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = t.sum(rnn_hn, dim=1, keepdim=True) # 按维度求和
        # atten_w [batch_size, 1, hidden_dims] 算出各个隐藏状态的权重？
        atten_w = self.attn_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = t.nn.Tanh()(h)
        # atten_w       [batch_size, 1, hidden_dims]
        # m.t(1,2)      [batch_size, hidden_dims, time_step(seq_len)]
        # atten_context [batch_size, 1, time_step(seq_len)]
        atten_context = t.bmm(atten_w, m.transpose(1, 2)) # bmm批次中每一个step的矩阵乘法， transpose交换两个维度
        # softmax_w [batch_size, 1, time_step]
        softmax_w = t.nn.functional.softmax(atten_context, dim=-1)  # 把最后一维度映射到[0,1]
        # 序列结果加权
        # context [batch_size, 1, hidden_dims]
        context = t.bmm(softmax_w, h)
        # context = t.bmm(softmax_w, x)
        # context = t.bmm(softmax_w, t.cat((h, x), dim=-1))
        # result [batch_size, hidden_dims]
        result = context.squeeze_(1)     #squeeze(arg)表示若第arg维的维度值为1，则去掉该维度。否则tensor不变。
        return result, softmax_w

    def forward(self, x, neighbor_mask):
        rnn_out, _ = self.gru_layer(x)
        # attention
        _, hn = self.gru_layer_attn_w(x)
        # rnn_out, (hn, _) = self.gru_layer(x)
        hn: t.Tensor
        hn = hn.permute(1, 0, 2)
        out, weights = self.attn_net_with_w(rnn_out, hn, neighbor_mask, x)

        # gru
        # lstm_tmp_out = t.chunk(rnn_out, 2, -1)
        # h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # out = h[:, -1, :].squeeze()
        # # out = rnn_out[:, -1, :].squeeze()
        #
        # # weights = t.zeros((rnn_out.shape[0], 1, rnn_out.shape[1]))
        # weights = None
        return out, weights
