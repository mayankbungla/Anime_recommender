# Hybrid scoring design

## The formula

```
final_score = alpha * cf_score + beta * content_score + gamma * popularity_score
```

Weights sum to 1 so the final score stays roughly on a 0 to 1 scale and
is easy to reason about.

## What each term actually means here

**cf_score**: the SVD model's predicted rating for this user and anime,
rescaled from its 1-10 scale to 0-1. This is the model's guess at "how
much would this specific person like this," learned from collaborative
patterns across all users.

**content_score**: cosine similarity between the anime's synopsis
embedding and a "taste vector" built by averaging the embeddings of
everything the user rated 7 or higher in train. Rescaled from cosine's
-1 to 1 range into 0 to 1. This captures "does this share tone or
subject matter with what the user already liked," independent of
whether other users have rated it at all.

**popularity_score**: log-scaled, min-max normalized count of how many
ratings an anime received in the training data. This is a plain
"how well known is this" signal, not personalized at all, it exists to
gently favor anime with an established audience over obscure ones when
the other two signals are weak or unavailable.

## Why these three, not something else

CF and content already disagree in genuinely interesting ways, per the
Week 4 evaluation (content beat CF on every metric, but CF is solving
the harder, more general problem, see reports/model_comparison.csv).
Popularity gets added on top for two reasons: it's the cheapest possible
signal to fall back on when an anime is too new or obscure for CF to
have learned anything (see the cold-start rule below), and pure content
similarity alone tends to recommend obscure lookalikes that technically
match but nobody's actually watched.

## Cold-start rule

If an anime has too few ratings for the CF model to have learned
anything about it (it's not in the trained catalogue), cf_score is
undefined, not zero. Rather than treating "unknown" the same as
"predicted low," alpha is dropped entirely and beta/gamma are
renormalized to still sum to 1:

```
final_score = (beta / (beta + gamma)) * content_score
            + (gamma / (beta + gamma)) * popularity_score
```

## Weights, tuned against real data

Tested against 200 real users using the Week 4 metrics
(reports/hybrid_weight_tuning.csv):

| alpha | beta | gamma | precision | recall | ndcg | coverage |
|---|---|---|---|---|---|---|
| 0.34 | 0.33 | 0.33 | 0.0545 | 0.0809 | 0.0769 | 0.0164 |
| 0.30 | 0.50 | 0.20 | 0.0485 | 0.0677 | 0.0653 | 0.0240 |
| 0.50 | 0.30 | 0.20 | 0.0460 | 0.0694 | 0.0635 | 0.0288 |
| 0.70 | 0.20 | 0.10 | 0.0345 | 0.0426 | 0.0459 | 0.0353 |

Roughly equal weights (0.34/0.33/0.33) won on every accuracy metric and
beat both standalone models (see reports/model_comparison.csv) on
precision, recall, and ndcg, so the hybrid genuinely adds value rather
than just picking whichever single model happened to be stronger.

The tradeoff: this same combo has the worst catalogue coverage of the
four (1.6%, versus 3.5% for the CF-heavy combo), leaning hardest into a
narrow, popular slice of anime. That's the exact problem Day 32's
diversity re-ranking is meant to address, so this isn't the final word
on weights, just the best accuracy tradeoff found so far.
