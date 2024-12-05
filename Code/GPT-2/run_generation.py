
#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Text generation with Fine tuned GPT-2 """


import argparse
import logging

import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="Path to pre-trained model",
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=100)

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--stop_token", type=str, default=None, help="Token to stop text generation")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_seed(args.seed)

    # Initialize the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Ensure the model and tokenizer are compatible
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)


    # Generate text
    # logger.info(f"Generating text with prompt: {prompt_text}")
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_return_sequences,
        do_sample=True,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_ids.shape) > 2:
        output_ids.squeeze_()

# Decode and display generated text
    for idx, sequence in enumerate(output_ids):
        print("=== GENERATED SEQUENCE {} ===".format(idx + 1))
        sequence = sequence.tolist()

        # Decode text
        text = tokenizer.decode(sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        if args.stop_token:
            text = text.split(args.stop_token)[0]
        # logger.info(f"Generated Sequence {idx + 1}:")
        print(text)

if __name__ == "__main__":
    main()