import torch
import torch.optim as optim
from typing import List, Dict

try:
    from src.skipgram import SkipGramNeg, NegativeSamplingLoss  
    from src.data_processing import get_batches, cosine_similarity
except ImportError:
    from skipgram import SkipGramNeg, NegativeSamplingLoss
    from data_processing import get_batches, cosine_similarity

def train_skipgram(model: SkipGramNeg,
                   words: List[int], 
                   int_to_vocab: Dict[int, str], 
                   batch_size=512, 
                   epochs=5, 
                   learning_rate=0.003, 
                   window_size=5, 
                   print_every=1500,
                   device='cpu'):
    """Trains the SkipGram model using negative sampling."""
    
    # Definir la función de pérdida
    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    model.to(device)  # Mover el modelo al dispositivo adecuado

    # Bucle de entrenamiento
    for epoch in range(epochs):
        for input_words, target_words in get_batches(words, batch_size, window_size):
            steps += 1

            # Convertir inputs y context words en tensores
            inputs = torch.LongTensor(input_words).to(device)

            targets = torch.LongTensor(target_words).to(device)

            # Obtener input, output y noise vectors
            input_vectors = model.forward_input(inputs)  # (batch_size, embed_size)
            output_vectors = model.out_embed(targets)  # (batch_size, embed_size)
            noise_vectors = model.forward_noise(inputs.shape[0], n_samples=5)  # (batch_size, n_samples, embed_size)

            # Calcular la pérdida
            loss = criterion(input_vectors, output_vectors, noise_vectors)

            # Paso de optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % print_every == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Step: {steps}, Loss: {loss.item()}")

                # Calcular similitud coseno con palabras de validación
                valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
                _, closest_idxs = valid_similarities.topk(6)  # Top 6 más cercanos

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")