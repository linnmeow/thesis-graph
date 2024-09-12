import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, AdamW, BartConfig
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartDecoderLayer

from graph_module.dataset_graph import CNN_DM_Graph, custom_collate_fn, load_data
from graph_module.encoders import GraphAttentionNetwork, EncoderDocument, EncoderGraph
from graph_module.get_graph_embeddings import BiLSTM
from graph_module.initialize_weight import xavier_initialization, kaiming_initialization, normal_initialization
from graph_module.decoder import DecoderLayerWithDualCrossAttention 

class Seq2SeqModel(nn.Module):
    def __init__(self, bart_model_name, gat_in_channels, gat_out_channels, gat_heads, dropout, initialization_scheme='xavier'):
        super(Seq2SeqModel, self).__init__()
        self.bart = BartModel.from_pretrained(bart_model_name)
        self.encoder_document = EncoderDocument(bart_model_name)
        self.encoder_graph = EncoderGraph(gat_in_channels, gat_out_channels, gat_heads, dropout)
        self.decoder = nn.ModuleList([
            DecoderLayerWithDualCrossAttention(self.bart.config) for _ in range(self.bart.config.decoder_layers)
        ])
        self.output_proj = nn.Linear(self.bart.config.d_model, self.bart.config.vocab_size)

        # apply initialization
        if initialization_scheme == 'xavier':
            self.apply(xavier_initialization)
        elif initialization_scheme == 'kaiming':
            self.apply(kaiming_initialization)
        elif initialization_scheme == 'normal':
            self.apply(normal_initialization)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, graph_node_features, edge_index, use_cache=False):
        # get encoder outputs
        encoder_hidden_states = self.encoder_document(input_ids, attention_mask)

        # get graph embeddings if not all subgraphs were pruned
        if graph_node_features is not None and edge_index is not None:
            graph_embeddings = self.encoder_graph(graph_node_features, edge_index)
        else:
            graph_embeddings = None  

        # initialize decoder hidden states
        hidden_states = self.bart.get_input_embeddings()(decoder_input_ids)

        # apply custom decoder layers
        for layer in self.decoder:
            hidden_states = layer(
                decoder_input_ids=decoder_input_ids,
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                graph_embeddings=graph_embeddings,
                use_cache=use_cache
            )[0]

        # project to vocabulary size
        logits = self.output_proj(hidden_states)

        return logits
    
    def _get_ngrams(self, sequence, n):
        return set(tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1))
    
    def _contains_repeated_ngrams(self, sequence, ngram_size, used_ngrams):
        for n in range(1, ngram_size + 1):
            ngrams = self._get_ngrams(sequence, n)
            if ngrams.intersection(used_ngrams):
                return True
        return False
        
    def generate_summary(
        self,
        input_ids,
        attention_mask,
        graph_node_features,
        edge_index,
        max_length,
        min_length,
        ngram_size
    ):
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize decoder input with BOS token
        generated_ids = torch.full((batch_size, 1), self.bart.config.bos_token_id, dtype=torch.long, device=device)
        
        used_ngrams = set()

        for step in range(max_length):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=generated_ids,
                decoder_attention_mask=None,  # No mask needed during inference
                graph_node_features=graph_node_features,
                edge_index=edge_index,
                use_cache=True
            )

            # Get the logits of the last generated token
            logits = outputs[:, -1, :]
            probs = F.log_softmax(logits, dim=-1)
            
            # Sample the next token
            next_token_probs = probs.squeeze(1)  # Remove the sequence dimension
            next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(1)

            # Check if the new token results in repeated n-grams
            new_sequence = torch.cat([generated_ids, next_token_id], dim=1)
            new_sequence_list = new_sequence[0].tolist()
            
            # Handling repeated n-grams
            if self._contains_repeated_ngrams(new_sequence_list, ngram_size, used_ngrams):
                # Ensure at least two tokens are available before calling topk
                if next_token_probs.size(0) > 1:
                    next_token_id = torch.topk(next_token_probs, 2).indices[1].unsqueeze(1)
                else:
                    next_token_id = torch.topk(next_token_probs, 1).indices[0].unsqueeze(1)

            # Append the predicted token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # Update used n-grams
            used_ngrams.update(self._get_ngrams(new_sequence_list, ngram_size))

            # Ensure min_length constraint is met before allowing <eos> token to break the loop
            if step >= min_length and next_token_id.item() == self.bart.config.eos_token_id:
                break

        return generated_ids


    def generate_summary_with_beam_search(
        self,
        input_ids,
        attention_mask,
        graph_node_features,
        edge_index,
        max_length,
        min_length,
        beam_size,
        ngram_size
    ):
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # initialize the beam search
        beams = [(torch.full((batch_size, 1), self.bart.config.bos_token_id, dtype=torch.long, device=device), 0, set())]  # (sequence, score, used_ngrams)

        # list to store the final hypotheses
        final_beams = []

        for step in range(max_length):
            new_beams = []

            # iterate over all beams
            for seq, score, used_ngrams in beams:
                # forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=seq,
                    decoder_attention_mask=None,  # no mask needed during inference
                    graph_node_features=graph_node_features,
                    edge_index=edge_index,
                    use_cache=True
                )

                # get logits and scores for the last token
                logits = outputs[:, -1, :]
                probs = F.log_softmax(logits, dim=-1)  # get log probabilities

                # expand the beams
                for token_id in range(logits.size(-1)):
                    new_seq = torch.cat([seq, torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)], dim=1)
                    new_score = score + probs[:, token_id].item()

                    # ensure that the <eos> token is not selected before reaching min_length
                    if step + 1 < min_length and token_id == self.bart.config.eos_token_id:
                        continue  # skip adding this sequence if it ends with <eos> before min_length
                    
                    # check for repeated n-grams
                    new_used_ngrams = used_ngrams.copy()
                    new_seq_list = new_seq[0].tolist()
                    if self._contains_repeated_ngrams(new_seq_list, ngram_size, new_used_ngrams):
                        continue

                    # Add new sequence and score
                    new_used_ngrams.update(self._get_ngrams(new_seq_list, ngram_size))
                    new_beams.append((new_seq, new_score, new_used_ngrams))

            # prune beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            # check if end-of-sequence token reached
            final_beams.extend([beam for beam in new_beams if beam[0][0, -1].item() == self.bart.config.eos_token_id and step + 1 >= min_length])
            new_beams = [beam for beam in new_beams if beam[0][0, -1].item() != self.bart.config.eos_token_id or step + 1 < min_length]

            if not new_beams and final_beams:
                # if no more beams but some final sequences, break early
                break

            beams = new_beams

        # get the best sequence
        if not final_beams:
            final_beams = beams

        best_seq = max(final_beams, key=lambda x: x[1])[0]

        return best_seq