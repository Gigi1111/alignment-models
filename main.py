# Author : Debanjali Biswas

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



"""
Main Function

Usage: python main.py model_name embedding_name

(model_name: ['Sequence' : Sequential ordering of alignments, 
              'Cosine_similarity' : Cosine model, 
              'Naive' : Common Action Pair Heuristics model,
              'Alignment-no-feature' : Base Alignment model, 
              'Alignment-with-feature' : Extended Alignment model])

(embedding_name : ['bert' : BERT embeddings,
                   'elmo' : ELMO embeddings])
"""

# importing libraries
import time
import torch
import flair
import argparse
import torch.nn as nn
import torch.optim as optim

from model import AlignmentModel
from cosine_similarity_model import SimpleModel
from sequence_model import SequenceModel
from naive_model import NaiveModel
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings
from training_testing import Folds
from constants import OUTPUT_DIM, LR, EPOCHS, FOLDS, HIDDEN_DIM1, HIDDEN_DIM2, CUDA_DEVICE

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
flair.device = device


def main():
  
    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
    flair.device = device
    
    parser = argparse.ArgumentParser(description = """Automatic Alignment model""")
    parser.add_argument('model_name', type=str, help="""Model Name; one of {'Simple', 'Naive', 'Alignment-no-feature', 'Alignment-with-feature'}""") # TODO: add options for fat graphs (with parents and grandparents)
    parser.add_argument('--embedding_name', type=str, default='bert', help='Embedding Name (Default is bert, alternative: elmo)')
    parser.add_argument('--cuda-device', type=str, help="""Select cuda; default: cuda:0""")
    args = parser.parse_args()

    model_name = args.model_name
    
    embedding_name = args.embedding_name

    if args.cuda_device:
        device = torch.device("cuda:"+args.cuda_device if torch.cuda.is_available() else "cpu")
        flair.device = device 

    start = time.time()
    print("Start time:", start)
    print("-------Loading Model-------")

    # Loading Model definition
    
    if embedding_name[:4] == 'bert' :

        tokenizer = BertTokenizer.from_pretrained(
            embedding_name
        )  # Bert Tokenizer
    
        emb_model = BertModel.from_pretrained(embedding_name, output_hidden_states=True).to(
            device
        )  # Bert Model for Embeddings
        
        embedding_dim = emb_model.config.to_dict()[
            "hidden_size"
        ]  # BERT embedding dimension
    

    elif embedding_name[:7] == 'roberta' : # roberta-base / roberta-large

        tokenizer = RobertaTokenizer.from_pretrained(
            embedding_name
        )  # RoBerta Tokenizer
    
        emb_model = RobertaModel.from_pretrained(embedding_name, output_hidden_states=True).to(
            device
        )  # RoBerta Model for Embeddings
        
        embedding_dim = emb_model.config.to_dict()[
            "hidden_size"
        ]  # RoBerta embedding dimension
    
    elif embedding_name[:4] == 'elmo' :
        
        tokenizer = Sentence #Flair sentence for ELMo embeddings
        print(embedding_name[5:])
        emb_model = ELMoEmbeddings(embedding_name[5:])
        
        embedding_dim = emb_model.embedding_length

    TT = Folds()  # calling the Training and Testing class

    if model_name == "Alignment-with-feature":

        model = AlignmentModel(embedding_dim, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, device).to(
            device
        )  # Out Alignment Model with features

        print(model)
        """for name, param in model.named_parameters():
            if param.requires_grad:
                    print(name)"""

        optimizer = optim.Adam(model.parameters(), lr=LR, capturable=True)  # optimizer for training
        criterion = nn.CrossEntropyLoss()  # Loss function

        ################ Cross Validation Folds #################

        TT.run_folds(
            embedding_name, 
            emb_model, tokenizer, model, optimizer, criterion, EPOCHS, FOLDS, device
        )

    elif model_name == "Alignment-no-feature":

        model = AlignmentModel(
            embedding_dim, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, device, False
        ).to(
            device
        )  # Out Alignment Model w/o features

        print(model)

        optimizer = optim.Adam(model.parameters(), lr=LR, capturable=True)  # optimizer for training
        criterion = nn.CrossEntropyLoss()  # Loss function

        TT.run_folds(
            embedding_name,
            emb_model, 
            tokenizer,
            model,
            optimizer,
            criterion,
            EPOCHS,
            FOLDS,
            device,
            False,
        )

    elif model_name == "Cosine_similarity":

        cosine_similarity_model = SimpleModel(embedding_dim, device).to(device) # Simple Cosine Similarity Baseline

        print(cosine_similarity_model)

        print("-------Testing (Simple Baseline) -------")

        TT.test_simple_model(embedding_name, emb_model, tokenizer, cosine_similarity_model, device)
        
        
    elif model_name == 'Naive':
        
        naive_model = NaiveModel(device) # Naive Common Action Pair Heuristics Baseline
        
        print('Common Action Pair Heuristics Model')
        
        ################ Cross Validation Folds #################
        
        TT.run_naive_folds(
            naive_model,
            FOLDS
            )
        
    elif model_name == 'Sequence':
        
        sequence_model = SequenceModel()
        
        print('Sequential Alignments')
        
        sequence_model.test_sequence_model()

    else:

        print(
            "Incorrect Argument: Model_name should be ['Cosine_similarity', 'Naive', 'Alignment-no-feature', 'Alignment-with-feature']"
        )

    end = time.time()
    print("End time:", end)
    print("Time elapsed:", end - start)

if __name__ == "__main__":
    main()
