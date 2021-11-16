from typing import List, Optional
import re


def replace_url(texts: List[str], replacement: Optional[str] = 'URL') -> List[str]:
    '''Replace URLs in a tweet by a token. The URLs in tweets has the format: https://t.co/afQGDec3Es'''

    regex = re.compile(r'(https:\/\/t.co\/\w+)', re.IGNORECASE)
    texts_without_url = []
    for line in texts:
        line = regex.sub(replacement, line)
        texts_without_url.append(line)

    return texts_without_url
