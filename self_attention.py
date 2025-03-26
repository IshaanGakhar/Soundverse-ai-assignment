import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """
    Args:
        X: Input matrix
        W_q: Query weight matrix
        W_k: Key weight matrix
        W_v: Value weight matrix
    
    Returns:
        Q: Query matrix
        K: Key matrix
        V: Value matrix
    """
    # query matrix = input * query weights (all vectors)
    Q = np.dot(X, W_q)
    
    # key matrix = input * key weights
    K = np.dot(X, W_k)
    
    # value matrix = input * value weights
    V = np.dot(X, W_v)
    
    # return triplet
    return Q, K, V

def self_attention(Q, K, V, scale=True):
    """
    Args:
        Q (np.array): Query matrix
        K (np.array): Key matrix
        V (np.array): Value matrix
        scale (bool): Whether to scale the dot product by sqrt of key dimension
    
    Returns:
        Output of self-attention
    """
    # attention score = Q * K^T (transposed since we want to match query and key dimensions) (need to match dimensions? why arent they already matched)
    attention_scores = np.dot(Q, K.T)
    
    # scaling to prevent extremely small or large gradients
    if scale:
        d_k = K.shape[1]
        attention_scores /= np.sqrt(d_k)
    
    # apply softmax to each row
    attention_weights = np.exp(attention_scores)
    attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
    
    # self attention = attention weights * V (as per scaled dot product attention)
    output = np.dot(attention_weights, V)
    
    return output


def main():

    # sample input
    X = np.array([[1, 0], [0, 1]])
    
    # sample weights
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])
    
    # compute q, k, v triplet
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    
    # calc self attention
    output = self_attention(Q, K, V)
    
    print("Input Matrix X:")
    print(X)
    print("\nQuery Matrix Q:")
    print(Q)
    print("\nKey Matrix K:")
    print(K)
    print("\nValue Matrix V:")
    print(V)
    print("\nSelf-Attention Output:")
    print(output)

# Run
if __name__ == "__main__":
    main()