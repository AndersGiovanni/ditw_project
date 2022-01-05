# Rehydrating tweets

The *rehydration* guide is borrowed from the [Danish Gigaword](https://gigaword.dk/) project.

We recommend the open-source tool [twarc](https://github.com/DocNow/twarc) to rehydrate tweets. Once you install and set up `twarc`, rehydration is as simple as calling `twarc` on the raw tweet ID file. For example, for the `dkpol_tweet_ids.txt` file:

```bash
twarc hydrate \
    data/dkpol_tweet_ids.txt > data//hydrated_tweets.txt
```
