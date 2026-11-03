from app.dedupe import MinHasher, drop_near_duplicates, jaccard, shingles


def test_minhash_estimates_jaccard():
    a = "the refund window for annual plans is thirty days from the purchase date"
    b = "the refund window for annual plans is thirty days from the invoice date"
    true = jaccard(shingles(a), shingles(b))
    h = MinHasher(num_perm=256)
    estimate = MinHasher.similarity(h.signature(a), h.signature(b))
    assert abs(estimate - true) < 0.15


def test_drop_near_duplicates_keeps_first_copy():
    docs = [
        "Rotate IAM access keys every 90 days and delete unused keys in the account.",
        "Rotate IAM access keys every 90 days and delete unused keys in the account!",
        "Enable MFA on the root account.",
    ]
    assert drop_near_duplicates(docs) == [docs[0], docs[2]]
