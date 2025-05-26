import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from episodes import episodes_to_batch, beliefs_to_batch, episodes_to_masks

class BeliefDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, belief_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, belief_dim)

    def forward(self, x):
        # x: [B, T, D]
        x = self.fc1(x)        # works on [B, T, D]
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=-1)  # [B, T, belief_dim]
        #print(f"Belief Decoder Output Shape: {x.shape}")
        return x

class linearBeliefDecoder(nn.Module):
    def __init__(self, input_dim, belief_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, belief_dim)

    def forward(self, x):
        # x: [B, T, D]
        x = self.fc(x)  # [B, T, belief_dim]
        x = nn.functional.softmax(x, dim=-1)  # Apply softmax to get probabilities
        return x

def sequence_cross_entropy(p_target, q_pred, mask=None, eps=1e-8):
    """
    Compute cross-entropy between two categorical distributions over time.

    Args:
        p_target: [B, T, C] true distribution (e.g., one-hot or soft label)
        q_pred:   [B, T, C] predicted distribution (must be softmaxed already)
        mask:     [B, T] optional mask for valid time steps (1 = valid, 0 = ignore)
        eps:      small constant to avoid log(0)

    Returns:
        Scalar cross-entropy loss
    """
    # Ensure numerical stability
    log_q = torch.log(q_pred + eps)
    
    # Element-wise cross entropy: -P * log(Q)
    cross_entropy = -torch.sum(p_target * log_q, dim=-1)  # [B, T]

    if mask is not None:
        cross_entropy = cross_entropy * mask  # mask out invalid steps
        return cross_entropy.sum() / mask.sum()
    else:
        return cross_entropy.mean()
    
    

def train_belief_decoder(belief_model, rnn_model, episodes, num_epochs=10, lr=1e-3, batch_size=1000):
    """
    Train the belief decoder using the RNN model's outputs as input.
    """

    optimizer = torch.optim.Adam(belief_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    belief_model.train()

    histories, rewards  = episodes_to_batch(episodes)
    beliefs             = beliefs_to_batch(episodes)
    mask, _, _          = episodes_to_masks(episodes)

    for epoch in range(num_epochs):
        # choose "batch_size" random episodes from the dataset
        indices = np.random.choice(len(episodes), size=min(batch_size, len(episodes)), replace=False)
        
        # Get RNN outputs
        _, hidden_states = rnn_model(histories[indices], mask[indices]) # shape: [B, T, H]
        #print(f"Hidden States Shape: {hidden_states.shape}")  # Debugging output

        optimizer.zero_grad()
        
        # Get beliefs from the belief decoder
        predicted_beliefs = belief_model(hidden_states)  # shape: [B, T, belief_dim]
        
        # Calculate loss
        loss = sequence_cross_entropy(beliefs[indices], predicted_beliefs, mask[indices])
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch+1}, Belief Decoder Loss: {loss.item():.4f}", end='\r')

    return loss.item()


def evaluate_belief(value_model, belief_model, episode):
    value_model.eval()  # switch to eval mode
    belief_model.eval()  # switch to eval mode

    with torch.no_grad():
        history = torch.tensor(episode.history, dtype=torch.float32).unsqueeze(0)  # shape: [1, T, D]
        length = [history.shape[1]]  # length = [T]
        mask = torch.ones((1, history.shape[1]), dtype=torch.float32)

        _, hidden = value_model(history, mask)  # shape: [1, T]
        predicted_beliefs = belief_model(hidden)
        
        return predicted_beliefs.numpy()
    


def plot_decoded_belief(value_model, belief_model, test_episode, cliff_dim = (3, 4)):
    true_beliefs = test_episode.belief_states
    true_beliefs = np.round(true_beliefs, 2)

    beliefs = evaluate_belief(value_model, belief_model, test_episode)
    beliefs = np.round(beliefs, 2)[0]

    for t in range(len(true_beliefs)):
        fig, axs = plt.subplots(1, 3, figsize=(6, 3))
        # true beliefs
        axs[0].imshow(true_beliefs[t].reshape(cliff_dim), cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        axs[0].set_title(f"True Beliefs at t={t}")
        axs[0].axis('off')
        axs[0].invert_yaxis()  # Invert y-axis to match the grid orientation
        # predicted beliefs
        axs[1].imshow(beliefs[t].reshape(cliff_dim), cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        axs[1].set_title(f"Decoded Beliefs at t={t}")
        axs[1].axis('off')
        axs[1].invert_yaxis()
        # Compute and display the difference
        diff_data = np.abs(true_beliefs[t] - beliefs[t]).reshape(cliff_dim)
        axs[2].imshow(diff_data, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        axs[2].set_title(f"Difference at t={t}")
        axs[2].axis('off')
        axs[2].invert_yaxis()

        # Annotate the cells in the third subplot
        for i in range(cliff_dim[0]):
            for j in range(cliff_dim[1]):
                axs[2].text(j, i, f"{diff_data[i, j]:.2f}", ha='center', va='center', color='white')

        plt.tight_layout()
        plt.show()