import torch
import torch.nn.functional as F
# ### Starting with a basic form of self-attention

# - Assume we have an input sentence that we encoded via a dictionary, which maps the words to integers as discussed in the RNN chapter:





# input sequence / sentence:
#  "Can you help me to translate this sentence"

sentence = torch.tensor(
    [0, # can
     7, # you
     1, # help
     2, # me
     5, # to
     6, # translate
     4, # this
     3] # sentence
)

print(sentence)

# - Next, assume we have an embedding of the words, i.e., the words are represented as real vectors.
# - Since we have 8 words, there will be 8 vectors. Each vector is 16-dimensional:

torch.manual_seed(123) # ensures the randomly initialized embedding vectors inside the [9,16] vectors
embed = torch.nn.Embedding(10, 16)
embedded_sentence = embed(sentence).detach()
embedded_sentence.shape

# - The goal is to compute the context vectors $\boldsymbol{z}^{(i)}=\sum_{j=1}^{T} \alpha_{i j} \boldsymbol{x}^{(j)}$,
# which involve attention weights $\alpha_{i j}$.
# - In turn, the attention weights $\alpha_{i j}$ involve the $\omega_{i j}$ values
# - Let's start with the $\omega_{i j}$'s first, which are computed as dot-products:
#
# $$\omega_{i j}=\boldsymbol{x}^{(i)^{\top}} \boldsymbol{x}^{(j)}$$
#
#



omega = torch.empty(8, 8)

for i, x_i in enumerate(embedded_sentence):
    for j, x_j in enumerate(embedded_sentence):
        omega[i, j] = torch.dot(x_i, x_j)


# - Actually, let's compute this more efficiently by replacing the nested for-loops with a matrix multiplication:
omega_mat = embedded_sentence.matmul(embedded_sentence.T)


# Use allclose function to check the matrix multiplication produces the expected results
# If two tensors contains the same values, returns True
print(torch.allclose(omega_mat, omega))


# - Next, let's compute the attention weights by normalizing the "omega" values so they sum to 1
#
# $$\alpha_{i j}=\frac{\exp \left(\omega_{i j}\right)}{\sum_{j=1}^{T} \exp \left(\omega_{i j}\right)}=\operatorname{softmax}\left(\left[\omega_{i j}\right]_{j=1 \ldots T}\right)$$
#
# $$\sum_{j=1}^{T} \alpha_{i j}=1$$

attention_weights = F.softmax(omega, dim=1)
attention_weights.shape

# Note that attention_weights is an 8×8 matrix, where each element represents an attention weight,
# 𝛼_ij. These attention weights indicate how relevant each word is to the ith word.

# - We can conform that the columns sum up to one:



attention_weights.sum(dim=1)


# - Now that we have the attention weights, we can compute the context vectors $\boldsymbol{z}^{(i)}=\sum_{j=1}^{T} \alpha_{i j} \boldsymbol{x}^{(j)}$, which involve attention weights $\alpha_{i j}$
# - For instance, to compute the context-vector of the 2nd input element (the element at index 1), we can perform the following computation:
x_2 = embedded_sentence[1, :]
context_vec_2 = torch.zeros(x_2.shape)
for j in range(8):
    x_j = embedded_sentence[j, :]
    context_vec_2 += attention_weights[1, j] * x_j

print(context_vec_2)



# - Or, more effiently, using linear algebra and matrix multiplication:



context_vectors = torch.matmul(
        attention_weights, embedded_sentence)


torch.allclose(context_vec_2, context_vectors[1]) # check context vector for second input (2nd row)


# ###  Parameterizing the self-attention mechanism: scaled dot-product attention
torch.manual_seed(123)

d = embedded_sentence.shape[1]
U_query = torch.rand(d, d)
U_key = torch.rand(d, d)
U_value = torch.rand(d, d)



# using the query projection matrix, we compute the query sequence
# for this example, 2nd input element, x^(i) as our query
x_2 = embedded_sentence[1]
query_2 = U_query.matmul(x_2)



# Similarly, compute key and value sequences k^(i) and v^(i)
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)



# We also need key and value sequences for all other input elements
# in the key matrix, the ith row corresponds to the key sequences of the ith input element
# the same applies for the value matrix. We can confirm this with allclose()
keys = U_key.matmul(embedded_sentence.T).T
torch.allclose(key_2, keys[1])

values = U_value.matmul(embedded_sentence.T).T
torch.allclose(value_2, values[1])



# compute unnormalized weight 𝜔_ij as the dot product between query and key
# this is different than previous that was not parameterized version
# where we did pairwise dot product between given input sequence element x^(i) and jth sequence element x^(j)

# i.e. following computes unnormalized attention weight 𝜔_23.
# that is, the dot product between our query and the third input sequence element
omega_23 = query_2.dot(keys[2])
omega_23

# scale up this computation to all keys, since we will be needing these later
omega_2 = query_2.matmul(keys.T)
omega_2

# Next step, go from unnormalized attention weight 𝜔_ij to normalized attention weight 𝛼_ij using softmax function
# also we can further us 𝜔_ij as (𝜔_ij / sqrt(m)) before using softmax.
# Here m = d_k ensures the Euclidean length of the weight vectors will be approx. the same range.
# The following normalize attention weight for entire input sequence with respect to the 2nd input element as query
attention_weights_2 = F.softmax(omega_2 / d ** 0.5, dim=0)
attention_weights_2

# context_vector_2nd = torch.zeros(values[1, :].shape)
# for j in range(8):
#    context_vector_2nd += attention_weights_2[j] * values[j, :]

# context_vector_2nd

# Finally the output is a weighted average of value sequencees z^(i)
context_vector_2 = attention_weights_2.matmul(values)
context_vector_2

# ## Attention is all we need: introducing the original transformer architecture


# ###  Encoding context embeddings via multi-head attention

torch.manual_seed(123)

# Create single projection matrix
d = embedded_sentence.shape[1]
one_U_query = torch.rand(d, d)

# Assume we have eight attention heads similar to the original transformer, h = 8
h = 8
multihead_U_query = torch.rand(h, d, d)
multihead_U_key = torch.rand(h, d, d)
multihead_U_value = torch.rand(h, d, d)

# As shown, multiple heads can be added by simply adding an additional dimension
multihead_query_2 = multihead_U_query.matmul(x_2)
print(multihead_query_2.shape)

# Similarly we compute key and value sequences for each head
multihead_key_2 = multihead_U_key.matmul(x_2)
multihead_value_2 = multihead_U_value.matmul(x_2)

# show key vector of the input element via third attention head
print(multihead_key_2[2])


# As mentioned, we need to repeat the key and value computation for all input sequence elements, not just x_2
# this is needed to compute self-attention later. A simple & illustrative way to do this is bye expanding
# the input seuqnce embeddings to size 8 as the first dimension, which is the number of attention heads
# we use .repeat() method
stacked_inputs = embedded_sentence.T.repeat(8, 1, 1)
print(stacked_inputs.shape)


# then we can have a batch matrix multiplication via torch.bmm() with attention heads to compute all keys
multihead_keys = torch.bmm(multihead_U_key, stacked_inputs)
print(multihead_keys.shape)
# so we now have a tensor referring to 8 attention head in its first dimension
# the second and third dimensions refer to embedding size and number of words.
# we will swap the second and third dimensions so that they keys have a more intuitive representation
# meaning same dimensionality as the original input sequence embedded_sentence
multihead_keys = multihead_keys.permute(0, 2, 1)
print(multihead_keys.shape)

# After rearranging, we can access second key value in the second attention head as follows:
multihead_keys[2, 1] # index: [2nd attention head, 2nd key]

# we see this is the same key value that we got via multihead_key_2[2] earlier, indicating that our complex
# matrix manipulations and computations are correct. So we repeat it for the value sequences
multihead_values = torch.matmul(multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0, 2, 1)

# We follow the steps of single head attention calculation to calculate the context vectors
# We will skip intermediate steps for brevity and assume we have already computed the context vectors
# for the second input element as the query and the eight different attention heads, represented as multihead_z_2 (random data)
multihead_z_2 = torch.rand(8, 16)
# Note first dimension indexes over 8 attention heads, and the context vectors, similar to the input sentences
# are 16-dimensional vectors. Think of multihead_z_2 as eight copies of the z^(2), meaning we have one z^(2)
# for each of the eight attention heads.

# We concatenate these vectors into one long vector of length d_v x h and use linear projection
# (via fully connected layer) to map it back to a vector of length d_v
linear = torch.nn.Linear(8*16, 16)
context_vector_2 = linear(multihead_z_2.flatten())
context_vector_2.shape


# ### Learning a language model: decoder and masked multi-head attention

# ### Implementation details: positional encodings and layer normalization