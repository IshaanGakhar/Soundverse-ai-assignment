import numpy as np

def multi_head_attention(Q, K, V, n_heads=2, scale=True):
    """
    Args:
        Q: Query matrix (batch_size x input_dim)
        K: Key matrix (batch_size x input_dim)
        V: Value matrix (batch_size x input_dim)
        n_heads: Number of attention heads
        scale: Whether to scale the dot product
    
    Returns:
        Output of multi-head attention
    """
    # input dimensions
    batch_size, input_dim = Q.shape
    
    # assert divisibility by heads
    assert input_dim % n_heads == 0, "Input dimension must be divisible by number of heads"
    
    # dimensions of each head
    head_dim = input_dim // n_heads
    
    # reshape and split into multiple heads
    Q_heads = Q.reshape(batch_size, n_heads, head_dim)
    K_heads = K.reshape(batch_size, n_heads, head_dim)
    V_heads = V.reshape(batch_size, n_heads, head_dim)
    
    # compute attention for each head
    head_outputs = []
    for i in range(n_heads):
        # extract head-specific Q, K, V
        Q_head = Q_heads[:, i, :]
        K_head = K_heads[:, i, :]
        V_head = V_heads[:, i, :]
        
        # compute attention scores
        attention_scores = np.dot(Q_head, K_head.T)
        
        # Optional scaling
        # if scale:
        #     attention_scores /= np.sqrt(head_dim)
        
        # compute attention weights using softmax
        attention_weights = np.exp(attention_scores)
        attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
        
        # compute weighted sum of values
        head_output = np.dot(attention_weights, V_head)
        head_outputs.append(head_output)
    
    # concatenate head outputs
    multi_head_output = np.concatenate(head_outputs, axis=1)
    
    return multi_head_output

def main():

    # sample input
    Q = np.array([[1, 0, 0, 1], [0, 1, 3, 2], [3, 4, 3, 1]])
    K = np.array([[1, 0, 7, 10], [0, 1, 12, 1], [3, 2, 2, 1]])
    V = np.array([[1, 0, 7, 7], [0, 1, 1, 1], [9, 1, 4, 1]])
    n_heads = 4

    print("Input Query (Q):")
    print(Q)
    print(Q.shape)
    print("\nInput Key (K):")
    print(K)
    print("\nInput Value (V):")
    print("\nNumber of Heads:", n_heads)
    
    # compute MHA
    output = multi_head_attention(Q, K, V, n_heads)
    
    print("\nMulti-Head Attention Output:")
    print(output)

# Run
if __name__ == "__main__":
    main()