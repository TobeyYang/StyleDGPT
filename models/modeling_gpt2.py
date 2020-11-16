# coding=utf-8
# This file is inherited from the official DialoGPT repo.
# Note that the embedding matrix and lm_head are untied.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import GPT2PreTrainedModel, GPT2Model

logger = logging.getLogger(__name__)


class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
        self.apply(self._init_weights)

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past": past}

    def tie_weights(self):
        pass    # do not tie!

    def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            element_weights=None
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
            lm_logits = lm_logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

            loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss1 = loss1.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels != -1, dim=1).type(loss1.type())
            ppl = torch.exp(torch.mean(torch.sum(loss1, dim=1).float() / label_size.float()))

            if element_weights is not None:
                if element_weights.dim() == 1:
                    element_weights = element_weights.unsqueeze(1)
                loss1 = loss1 * element_weights
            loss = torch.sum(loss1) / torch.sum(label_size)
            return (loss, ppl,) + outputs
        return outputs
