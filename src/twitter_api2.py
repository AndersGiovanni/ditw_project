from typing import Dict
import tweepy
import argparse
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
            wait_on_rate_limit=True)

    def search_all_tweets(self, query: str):

        for tweet in self.client.search_all_tweets(query, max_results=10):
            print(tweet)

    def search(self, query: str):

        data = list()

        tweets = self.client.search_recent_tweets(
            query,
            max_results=100,
            expansions=['entities.mentions.username', 'in_reply_to_user_id'],
            tweet_fields=['entities', 'public_metrics'])[0]

        for tweet in tweets:

            if tweet.text[:2] != 'RT':
                data.append(dict(tweet))

        # print(data)
        print(len(data))

        write_json(data, 'data/dkpol_tweets_sample.json')


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
        credential_path=args.credentials)

    t.search('#dkpol')
