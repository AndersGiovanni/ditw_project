from typing import Dict
import tweepy
import argparse
from src.json_utils import read_json


class TwitterGenerator:
    def __init__(self, credential_path) -> None:

        self.credentials: Dict = read_json(path=credential_path)
        self.api: tweepy.API = tweepy.API(
            self._authorize(),
            wait_on_rate_limit=True)

    def _authorize(self) -> tweepy.OAuthHandler:
        auth = tweepy.OAuthHandler(
            self.credentials["API_KEY"],
            self.credentials["API_SECRET_KEY"])
        auth.set_access_token(
            self.credentials["Access token"],
            self.credentials["Access token secret"])
        return auth

    def search(self, query: str):

        for tweet in tweepy.Cursor(self.api.search_tweets, q=query, lang='da', tweet_mode='extended').items():
            print(tweet)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Collecting tweets')

    parser.add_argument(
        '--credentials',
        type=str,
        required=False,
        default="graph_db/meta/creds.json",
        help='Specify the path to your credentials.json')

    args = parser.parse_args()

    t = TwitterGenerator(
        credential_path=args.credentials)

    t.search('#dkpol')
