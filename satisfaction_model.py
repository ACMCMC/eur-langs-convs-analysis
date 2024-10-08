# %%

# Define a custom classification head
import logging
from typing import Optional

import torch
import torch.nn as nn
import transformers


class SatisfactionModelConfig(transformers.PretrainedConfig):
    model_type = "satisfaction_model"

    def __init__(
        self,
        original_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        hidden_size=768,
        num_classes=3,
        num_layers=1,
        use_bias=False,
        use_weighted_loss=False,
        loss_label_smoothing=0.1,
        activation_function="Tanh",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.original_model_name = original_model_name
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.use_weighted_loss = use_weighted_loss
        self.loss_label_smoothing = loss_label_smoothing
        self.activation_function = activation_function


class SatisfactionModel(transformers.PreTrainedModel):
    config_class = SatisfactionModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = transformers.AutoModel.from_pretrained(config.original_model_name)

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        number_of_inputs = 6  # u1, a, u2, |a-u1|, |u2-a|, |u2-u1|
        # Get the input size as the size of the output of the model

        activation_function = getattr(nn, config.activation_function)

        self.classification_input_size = self.model.config.hidden_size

        assert config.num_layers >= 1, "num_layers must be at least 1"
        if config.num_layers == 1:
            layers = [
                nn.Linear(
                    self.classification_input_size * number_of_inputs,
                    config.num_classes,
                    bias=config.use_bias,
                ),
            ]
        elif config.num_layers == 2:
            layers = [
                nn.Linear(
                    self.classification_input_size * number_of_inputs,
                    config.hidden_size,
                    bias=config.use_bias,
                ),
                activation_function(),
                nn.Linear(
                    config.hidden_size,
                    config.num_classes,
                    bias=config.use_bias,
                ),
            ]
        elif config.num_layers > 2:
            layers = []
            layers.append(
                nn.Linear(
                    self.classification_input_size * number_of_inputs,
                    config.hidden_size,
                    bias=config.use_bias,
                )
            )
            layers.append(activation_function())
            for _ in range(config.num_layers - 2):
                layers.append(
                    nn.Linear(
                        config.hidden_size,
                        config.hidden_size,
                        bias=config.use_bias,
                    )
                )
                layers.append(activation_function())
            layers.append(
                nn.Linear(
                    config.hidden_size,
                    config.num_classes,
                    bias=config.use_bias,
                )
            )
        else:
            raise ValueError("num_layers must be at least 1")

        self.classification_head = nn.Sequential(*layers)

        # There is a class imbalance. We can use a weighted loss to address this.
        if config.use_weighted_loss:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=config.loss_label_smoothing,
                weight=(torch.tensor([1.6, 1.0, 1.6])),
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=config.loss_label_smoothing
            )

        # The classification head is not frozen
        for param in self.classification_head.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        assert (
            input_ids.shape[1] == 3
        ), f"Input_ids must have shape [batch_size, 3, seq_len], got {input_ids.shape}"
        # Unpack the input_ids
        u1_inputs, a_inputs, u2_inputs = input_ids.unbind(dim=1)
        assert u1_inputs.shape == a_inputs.shape == u2_inputs.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            print("Attention mask is None, using ones")
        u1_attention_mask, a_attention_mask, u2_attention_mask = attention_mask.unbind(
            dim=1
        )

        # Get the embeddings
        u1_outputs = self.model(
            input_ids=u1_inputs, attention_mask=u1_attention_mask, **kwargs
        )
        u1_embedding = u1_outputs.pooler_output
        a_outputs = self.model(
            input_ids=a_inputs, attention_mask=a_attention_mask, **kwargs
        )
        a_embedding = a_outputs.pooler_output
        u2_outputs = self.model(
            input_ids=u2_inputs, attention_mask=u2_attention_mask, **kwargs
        )
        u2_embedding = u2_outputs.pooler_output

        # Concatenate the embeddings
        concatenated = torch.cat(
            [
                u1_embedding,
                a_embedding,
                u2_embedding,
                torch.abs(a_embedding - u1_embedding),
                torch.abs(u2_embedding - a_embedding),
                torch.abs(u2_embedding - u1_embedding),
            ],
            dim=-1,
        )

        # Run through the classification head
        logits = self.classification_head(concatenated)

        # If we have labels, calculate the loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
