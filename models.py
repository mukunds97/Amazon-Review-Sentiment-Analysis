import torch
from torch import nn

class ReviewClassifier(nn.Module):
    def __init__(self, hidden_units, dropout_rate, activation, w2v_vectors, vocab_size, embedding_dim, embedding_matrix, num_classes, sent_size):
        super(ReviewClassifier, self).__init__()
        # Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight = nn.Parameter(torch.tensor(embedding_matrix))
        # self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(w2v_vectors.vectors))
        embedding_dim = w2v_vectors.vector_size
        print('Embedding Dim', embedding_dim)
        # hidden_units = int(3 * embedding_dim)

        # Hidden layer
        self.hidden_layer = nn.Linear(embedding_dim * sent_size, hidden_units)
        self.activation = activation

        # Add Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # Output layer (softmax)
        self.output_layer = nn.Linear(hidden_units, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding_layer(x)
        embedded = torch.flatten(embedded, start_dim=1)
        hidden_ouput = self.hidden_layer(embedded)
        activation_output = self.activation(hidden_ouput)
        dropout_output = self.dropout(activation_output)
        logits = self.output_layer(dropout_output)
        output = self.softmax(logits)
        return output