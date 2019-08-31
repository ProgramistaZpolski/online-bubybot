from textgenrnn import textgenrnn
from datetime import datetime
import os
textgen = textgenrnn(weights_path='colaboratory_weights.hdf5',
                       vocab_path='colaboratory_vocab.json',
                       config_path='colaboratory_config.json')
 
textgen.generate_samples(max_gen_length=1000)
textgen.generate_to_file('textgenrnn_texts.txt', max_gen_length=1000)
waiter = input("OK Super")
