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
    def __init__(self, bart_model_name, gat_in_channels, gat_out_channels, gat_heads, dropout, 
                 encoder_config=None, decoder_config=None, initialization_scheme='xavier'):
        super(Seq2SeqModel, self).__init__()
        # initialize BART model and its config
        self.bart = BartModel.from_pretrained(bart_model_name)
        self.bart_config = self.bart.config

        # if custom config are provided, overwrite the default config
        self.encoder_config = encoder_config if encoder_config is not None else self.bart_config
        self.decoder_config = decoder_config if decoder_config is not None else self.bart_config

        self.encoder_document = EncoderDocument(bart_model_name, encoder_config=self.encoder_config)
        self.encoder_graph = EncoderGraph(gat_in_channels, gat_out_channels, gat_heads, dropout)
        self.decoder = nn.ModuleList([
            DecoderLayerWithDualCrossAttention(decoder_config=self.decoder_config) for _ in range(self.decoder_config.decoder_layers)
        ])
        self.output_proj = nn.Linear(self.decoder_config.d_model, self.decoder_config.vocab_size)

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

        # apply custom decoder layers
        for layer in self.decoder:
            hidden_states = layer(
                decoder_input_ids=decoder_input_ids,
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
        # generate summary using greedy decoding
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # initialize decoder input with BOS token
        generated_ids = torch.full((batch_size, 1), self.bart.config.bos_token_id, dtype=torch.long, device=device)
        
        used_ngrams = set()

        for step in range(max_length):
            # forward pass
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
            
            # sample the next token
            next_token_probs = probs.squeeze(1)  # Remove the sequence dimension
            next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(1)  # Keep shape [batch_size, 1]

            # concatenate keeping the batch structure
            new_sequence = torch.cat([generated_ids, next_token_id], dim=1)  # [batch_size, sequence_length]

            # convert to list of tuples for n-gram checking
            new_sequence_list = [tuple(seq) for seq in new_sequence.tolist()]

            # check for repeated n-grams
            if self._contains_repeated_ngrams(new_sequence_list, ngram_size, used_ngrams):
                # ensure at least two tokens are available before calling topk
                if next_token_probs.size(1) > 1:
                    next_token_id = torch.topk(next_token_probs, 2, dim=-1).indices[:, 1].unsqueeze(1)
                else:
                    next_token_id = torch.topk(next_token_probs, 1, dim=-1).indices[:, 0].unsqueeze(1)

            # append the predicted token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)  # [batch_size, sequence_length]

            # Update used n-grams
            used_ngrams.update(self._get_ngrams(new_sequence_list, ngram_size))

            # # Print debugging information
            # print(f"Step: {step}, Min Length: {min_length}")

            # Check for EOS tokens across the batch
            eos_found = (next_token_id.squeeze() == self.bart.config.eos_token_id)

            if step >= min_length and eos_found.any():
                print("EOS token found in batch, stopping generation.")
                break  # Stop if any sequence reached the EOS token
            else:
                print("EOS token not found in batch, continuing generation.")

            # # Print if multiple tokens are generated
            # if next_token_id.shape[0] > 1:
            #     print(f"Multiple tokens generated: {next_token_id.tolist()}")

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

        final_beams = []

        with torch.no_grad():
            for step in range(max_length):
                new_beams = []

                # iterate over all beams
                for seq, score, used_ngrams in beams:
                    outputs = self.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=seq,
                        decoder_attention_mask=None,
                        graph_node_features=graph_node_features,
                        edge_index=edge_index,
                        use_cache=True
                    )

                    # Get log probabilities for the last token
                    logits = outputs[:, -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)

                    # expand the beams using vectorized operations
                    top_log_probs, top_indices = log_probs.topk(beam_size, dim=-1)

                    # iterate over the top tokens
                    for i in range(beam_size):
                        token_id = top_indices[0, i].item()
                        token_log_prob = top_log_probs[0, i].item()
                        new_score = score + token_log_prob

                        # create a new sequence by appending the token
                        new_seq = torch.cat([seq, torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)], dim=1)

                        # ensure the <eos> token is not selected before reaching min_length
                        if step + 1 < min_length and token_id == self.bart.config.eos_token_id:
                            continue  # Skip if <eos> is selected too early

                        # check for repeated n-grams
                        new_used_ngrams = used_ngrams.copy()
                        new_seq_list = new_seq[0].tolist()
                        if self._contains_repeated_ngrams(new_seq_list, ngram_size, new_used_ngrams):
                            continue

                        # add new sequence and score
                        new_used_ngrams.update(self._get_ngrams(new_seq_list, ngram_size))
                        new_beams.append((new_seq, new_score, new_used_ngrams))

                # prune beams and keep top beam_size sequences
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

                # move completed beams to final_beams
                final_beams.extend([beam for beam in new_beams if beam[0][0, -1].item() == self.bart.config.eos_token_id and step + 1 >= min_length])
                new_beams = [beam for beam in new_beams if beam[0][0, -1].item() != self.bart.config.eos_token_id or step + 1 < min_length]

                # if no more beams to expand and we have final beams, break early
                if not new_beams and final_beams:
                    break

                beams = new_beams

        # if no final beams found, use the remaining beams
        if not final_beams:
            final_beams = beams

        # select the best sequence from final beams
        best_seq = max(final_beams, key=lambda x: x[1])[0]

        return best_seq
