import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import STEncoder, LTEncoder
from recbole.model.loss import BPRLoss, EmbLoss


class FASRec(SequentialRecommender):
        def __init__(self, config, dataset):
            super(FASRec, self).__init__(config, dataset)

            # load parameters info
            self.config = config
            self.n_layers = config['n_layers']
            self.n_heads = config['n_heads']
            self.hidden_size = config['hidden_size']  # same as embedding_size
            self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
            self.hidden_dropout_prob = config['hidden_dropout_prob']
            self.attn_dropout_prob = config['attn_dropout_prob']
            self.hidden_act = config['hidden_act']
            self.layer_norm_eps = config['layer_norm_eps']
            self.GRU_dropout_prob = config['gru_dropout_prob']
            self.ssl_sem = config['ssl_sem']
            self.initializer_range = config['initializer_range']
            self.loss_type = config['loss_type']
            self.batch_size = config['train_batch_size']

            # define layers and loss
            self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
            self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
            self.stitem_encoder = STEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                config=self.config,
            )

            self.ltitem_encoder = LTEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                config=self.config,
            )

            self.gru_layers = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=False,
                batch_first=True,
            )
            self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
            self.dropout = nn.Dropout(self.hidden_dropout_prob)
            self.emb_dropout = nn.Dropout(self.GRU_dropout_prob)

            if self.loss_type == 'BPR':
                self.loss_fct = BPRLoss()
            elif self.loss_type == 'CE':
                self.loss_fct = nn.CrossEntropyLoss()
            else:
                raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")


            self.aug_nce_fct = nn.CrossEntropyLoss()
            self.sem_aug_nce_fct = nn.CrossEntropyLoss()

            # parameters initialization
            self.apply(self._init_weights)

        def _init_weights(self, module):
            """ Initialize the weights """
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
                # module.weight.data = self.truncated_normal_(tensor=module.weight.data, mean=0, std=self.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        def get_attention_mask(self, item_seq):
            """Generate left-to-right uni-directional attention mask for multi-head attention."""
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
            # mask for left-to-right unidirectional
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(item_seq.device)

            extended_attention_mask = extended_attention_mask * subsequent_mask
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask

        def forwardGRU(self, item_seq):
            item_seq_emb = self.item_embedding(item_seq)
            item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
            gru_output, _ = self.gru_layers(item_seq_emb_dropout)

            return gru_output

        def mlp(self, model_output, layer_sizes, activation_functions, device):
            layers = nn.ModuleList()

            # 构建MLP层
            last_layer_size = model_output.shape[-1]
            for layer_size in layer_sizes:
                linear_layer = nn.Linear(last_layer_size, layer_size).to(device)
                layers.append(linear_layer)

                last_layer_size = layer_size

            hidden_nn_layers = [model_output]
            for idx, layer in enumerate(layers):
                curr_hidden_nn_layer = layer(hidden_nn_layers[-1])
                activation = getattr(F, activation_functions[idx])
                curr_hidden_nn_layer = activation(curr_hidden_nn_layer)
                hidden_nn_layers.append(curr_hidden_nn_layer)

            nn_output = hidden_nn_layers[-1]
            return nn_output

        def forward(self, item_seq, item_seq_len):
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)

            item_emb = self.item_embedding(item_seq)
            input_emb = item_emb + position_embedding
            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)
            gru_output = self.forwardGRU(item_seq)

            extended_attention_mask = self.get_attention_mask(item_seq)
            st_output = self.stitem_encoder(gru_output, extended_attention_mask, output_all_encoded_layers=True)
            lt_output = self.ltitem_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
            st_output = st_output[-1]
            lt_output = lt_output[-1]
            outputls = torch.concat([st_output, lt_output], -1)
            output = self.mlp(outputls, [self.inner_size, self.hidden_size], ['relu', 'tanh'], device=item_seq.device)
            self.alpha_output = torch.sigmoid(output)
            outputfi = lt_output * self.alpha_output + st_output * (1 - self.alpha_output)
            output = self.gather_indexes(outputfi, item_seq_len - 1)
            return output,  st_output, lt_output    # [B H]



        def calculate_loss(self, interaction):
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            seq_output, _, _ = self.forward(item_seq, item_seq_len)
            pos_items = interaction[self.POS_ITEM_ID]
            if self.loss_type == 'BPR':
                neg_items = interaction[self.NEG_ITEM_ID]
                pos_items_emb = self.item_embedding(pos_items)
                neg_items_emb = self.item_embedding(neg_items)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
                loss = self.loss_fct(pos_score, neg_score)
            else:  # self.loss_type = 'CE'
                test_item_emb = self.item_embedding.weight
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
                loss = self.loss_fct(logits, pos_items)



            return loss


        def predict(self, interaction):
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            test_item = interaction[self.ITEM_ID]
            seq_output, _, _ = self.forward(item_seq, item_seq_len)
            test_item_emb = self.item_embedding(test_item)
            scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
            return scores

        def full_sort_predict(self, interaction):
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            seq_output, _, _ = self.forward(item_seq, item_seq_len)
            test_items_emb = self.item_embedding.weight
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
            return scores
