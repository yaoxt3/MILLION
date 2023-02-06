from learning_to_be_taught.environments.meta_world.language_meta_world import env_instructions, LanguageMetaWorld
from torchtext.vocab import GloVe
import pickle

embed_dim = LanguageMetaWorld.EMBED_DIM
word_embedding = GloVe(name='6B', dim=embed_dim)

used_vocab = dict()
for env_name, instructions in env_instructions.items():
        for instruction in instructions:
                for word in instruction.split():
                        used_vocab[word] = word_embedding.get_vecs_by_tokens(word)



with open('./meta_world_vocab.pkl', 'wb') as f:
        pickle.dump(used_vocab, f, pickle.HIGHEST_PROTOCOL)
