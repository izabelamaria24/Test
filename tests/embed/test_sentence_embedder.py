import pytest

from realestate.embed.sentence_embedder import MultilingualE5Embedder

pytestmark = pytest.mark.integration  # loads the real e5 model; excluded from CI


def test_embed_passage_and_query_return_same_dimensionality():
    embedder = MultilingualE5Embedder(device="cpu")  # cpu here so CI/non-Mac runs work too
    passage_vec = embedder.embed_passage("Garsoniera luminoasa in Centrul Istoric")
    query_vec = embedder.embed_query("apartament luminos in centru")

    assert len(passage_vec) == len(query_vec)
    assert len(passage_vec) == 768  # multilingual-e5-base hidden size


def test_similar_texts_have_higher_cosine_similarity_than_dissimilar():
    import numpy as np

    embedder = MultilingualE5Embedder(device="cpu")
    query_vec = np.array(embedder.embed_query("apartament luminos cu parchet"))
    similar_vec = np.array(embedder.embed_passage("Apartament luminos, parchet masiv, mult soare"))
    dissimilar_vec = np.array(embedder.embed_passage("Teren agricol de vanzare in Ilfov"))

    similar_score = query_vec @ similar_vec
    dissimilar_score = query_vec @ dissimilar_vec
    assert similar_score > dissimilar_score
