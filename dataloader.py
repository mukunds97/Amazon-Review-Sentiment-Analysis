from random import shuffle
import torch

class AmazonReviews(torch.utils.data.Dataset):
    def __init__(self, data, labels, sent_size, word2index, device):
        self.dataset = data
        self.labels = labels
        self.sent_word_limit = sent_size
        self.embedding_vector = word2index.wv
        self.word2index = word2index    
        self.device = device

    def __len__(self):
        return(len(self.dataset))
    
    def __getitem__(self, index):
        ''' Applying preprocessing to convert string tokens to indices based on word2vec model & adding special tokens '''
        review = self.dataset[index]
        cls_token = "<CLS>"
        sep_token = "<SEP>"
        pad_token = "<PAD>"
        unk_token = "<UNK>"
        cls_index = self.embedding_vector.key_to_index[cls_token]
        sep_index = self.embedding_vector.key_to_index[sep_token]
        pad_index = self.embedding_vector.key_to_index[pad_token]
        unk_index = self.embedding_vector.key_to_index[unk_token]
        processed_review = [pad_index] * self.sent_word_limit
        processed_review[0] = cls_index
        i = 1
        for token in review:
            if i == self.sent_word_limit-2:
                break
            if token in self.embedding_vector.key_to_index:
                token_index = self.embedding_vector.key_to_index[token]
            else:
                token_index = unk_index
            processed_review[i] = token_index
            i += 1
        processed_review[i] = sep_index
        sample = {'Review': torch.tensor(processed_review).to(self.device), 
                  'Class': torch.tensor(self.labels[index]).to(self.device)}
        return sample
    
def create_dataloader(dataset, labels, word2vec, sent_length, batch_size, device):
    dataset = AmazonReviews(dataset, labels, sent_length, word2vec, device)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)