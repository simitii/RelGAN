import argparse, pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--input_file', type=str, default="../dataset/preprocessed_data.txt")
parser.add_argument('--output_file', type=str, default="data/paper_generation.txt")
parser.add_argument('--char2idx', type=str, default="data/paper_gen_char2idx.pickle")
parser.add_argument('--idx2char', type=str, default="data/paper_gen_idx2char.pickle")

opt = parser.parse_args()

seq_len = opt.seq_len

text = open(opt.input_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

with open(opt.char2idx, "wb") as f:
    pickle.dump(char2idx, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(opt.idx2char, "wb") as f:
    pickle.dump(idx2char, f, protocol=pickle.HIGHEST_PROTOCOL)

vocab_text = []
for i in range(len(text)):
    if i % 100000 == 0:
        print("{}/{}".format(i, len(text)), end="\r")
    if i != 0 and i % seq_len != 0:
        vocab_text.append(" ") 
    vocab_text.append(str(char2idx[text[i]]))
    if i % seq_len == seq_len-1:
        vocab_text.append("\n")

vocab_text = "".join(vocab_text)

with open(opt.output_file, 'wt') as f:
    f.write(vocab_text)
