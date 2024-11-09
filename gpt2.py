from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer
from transformers import GPT2Model
# ## Building large-scale language models by leveraging unlabeled data
# ##  Pre-training and fine-tuning transformer models
#
#





# ## Leveraging unlabeled data with GPT









# ### Using GPT-2 to generate new text




# import pre-trained GPT model that can generate new text
generator = pipeline('text-generation', model='gpt2')
set_seed(123)

# prompt model with text snippet and ask it to generate new text based on input snippet
generator("Hey readers, today is",
          max_length=20,
          num_return_sequences=3)

# We can use a transformer model to generate features for training other models
# Following shows how we can use GPT-2 to generate features based on an input text
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Let us encode this sentence"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
# the code encoded the input sentence text into a tokenized format for the GPT-2 model
# as show, it mapped strings to an integer representation, and it set the attention mask to all 1s,
# which means that all words will be processed when we pass the encoded input to the model as show below:
model = GPT2Model.from_pretrained('gpt2')
output = model(**encoded_input)

#output variable  stores the last hidden state, that is our GPT-2 based feature encoding of the input sentence
output['last_hidden_state'].shape


# ### Bidirectional pre-training with BERT
#
