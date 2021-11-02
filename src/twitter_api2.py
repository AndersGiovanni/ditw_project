from typing import Dict
import tweepy
import argparse
import datetime
import rfc3339
import json
from src.json_utils import read_json, write_json
# For sending GET requests from the API
# https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all#Optional
# https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet
# https://developer.twitter.com/en/products/twitter-api/academic-research/application-info


class TwitterGenerator:
    def __init__(self, credential_path) -> None:

        self.credentials: Dict = read_json(path=credential_path)
        self.client: tweepy.Client = tweepy.Client(
            bearer_token=self.credentials['BEARER TOKEN'],
            consumer_key=self.credentials['API_KEY'],
            consumer_secret=self.credentials['API_SECRET_KEY'],
            access_token=self.credentials['Access token'],
            access_token_secret=self.credentials['Access token secret'],
            return_type=dict,
            wait_on_rate_limit=True)

    def search_all_tweets(self, query: str):

        for tweet in self.client.search_all_tweets(query, max_results=10):
            print(tweet)

    def search(self, query: str):

        data = list()

        tweets = self.client.search_recent_tweets(
            query,
            max_results=10,
            expansions=['geo.place_id', 'in_reply_to_user_id'],
            tweet_fields=['author_id', 'entities', 'public_metrics', 'context_annotations', 'created_at', 'referenced_tweets', 'geo'])

        for tweet in tweets.get('data'):
            print(tweet)
            print()

        # print(data)
        # print(len(data))

        #write_json(data, 'data/dkpol_tweets_sample.json')

    def search_all(self, query: str):

        iteration = 30

        kwargs = {
            'query': query,
            'max_results': 100,
            'start_time': rfc3339.rfc3339(datetime.date(2021, 1, 1)),
            'end_time': rfc3339.rfc3339(datetime.date(2021, 4, 1)),
            'expansions': ['geo.place_id', 'in_reply_to_user_id'],
            'tweet_fields': ['author_id', 'entities', 'public_metrics', 'context_annotations', 'created_at', 'referenced_tweets', 'geo'],
            'next_token': None
        }

        while iteration > 0:

            # Open jsonl file to store data
            jsonl_file = open('data/dkpol_tweets.jsonl', 'a')
            meta_file = open('data/dkpol_meta.jsonl', 'a')

            # Query the tweets
            tweets = self.client.search_all_tweets(**kwargs)

            next_token = tweets.get('meta').get('next_token')

            if next_token is not None:
                kwargs['next_token'] = next_token

            # Write all non-retweets to file
            count_tweets = 0
            for tweet in tweets.get('data'):
                if tweet['text'][:2] != 'RT':
                    jsonl_file.write(f'{json.dumps(tweet, sort_keys=True)}\n')
                    count_tweets += 1

            jsonl_file.close()

            # Write all meta-data for continous scraping
            meta = tweets.get('meta')
            meta['count_without_retweets'] = count_tweets
            meta_file.write(f'{json.dumps(meta, sort_keys=True)}\n')
            meta_file.close()

            iteration -= 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Collecting tweets')

    parser.add_argument(
        '--credentials',
        type=str,
        required=False,
        default="credentials.json",
        help='Specify the path to your credentials.json')

    args = parser.parse_args()

    t = TwitterGenerator(
        credential_path="academic_creds.json")

    t.search_all('#dkpol')
