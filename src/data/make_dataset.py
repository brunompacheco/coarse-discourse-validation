# -*- coding: utf-8 -*-
# import click
# import logging
import requests
import os
import time
import praw
import json
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


# @click.command()
def main(proj_root):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    print("========================================")
    print("Downloading annotations file from GitHub", end=' ')

    url = 'https://raw.githubusercontent.com/google-research-datasets/coarse-discourse/master/coarse_discourse_dataset.json'
    filename = os.path.join(proj_root, 'data', 'raw',
                            'coarse_discourse_dataset.json')

    response = requests.get(url)
    with open(filename, 'wb') as jsonfile:
        jsonfile.write(response.content)

    print("- Done!")

    jsonfile = open(filename)

    print("Getting Reddit access", end=' ')
    try:
        api_id = os.environ['REDDIT_CLIENT_ID']
        api_secret = os.environ['REDDIT_CLIENT_SECRET']
    except KeyError:
        print("Please ser your reddit API credentials as environment variables.")

    reddit = praw.Reddit(client_id=api_id,
                         client_secret=api_secret,
                         user_agent='linux:coarse-discourse-validation:0.1 (by /u/brunompac)')
    print("- Done!")

    lines = jsonfile.readlines()
    dump_with_reddit = open(os.path.join(proj_root, 'data', 'interim',
                            'coarse_discourse_dump_reddit.json'), 'w')

    threads_found = 0
    threads_total = len(lines)
    posts_found = 0
    posts_total = 0

    for line in lines:
        thread = json.loads(line)

        print("=Gettin thread: "+thread['title'])

        submission = reddit.submission(url=thread['url'])

        # Annotators only annotated the 40 "best" comments determined by Reddit
        submission.comment_sort = 'best'
        submission.comment_limit = 40

        post_id_dict = {}

        for post in thread['posts']:
            post_id_dict[post['id']] = post

        try:
            full_submission_id = 't3_' + submission.id

            if full_submission_id in post_id_dict:
                post_id_dict[full_submission_id]['body'] = submission.selftext

                # For a self-post, this URL will be the same URL as the thread.
                # For a link-post, this URL will be the link that the link-post
                # is linking to.
                post_id_dict[full_submission_id]['url'] = submission.url
                if submission.author:
                    post_id_dict[full_submission_id]['author'] = submission.author.name

            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                full_comment_id = 't1_' + comment.id
                if full_comment_id in post_id_dict:
                    post_id_dict[full_comment_id]['body'] = comment.body
                    if comment.author:
                        post_id_dict[full_comment_id]['author'] = comment.author.name
        except Exception as e:
            print("Error %s" % (e))
        threads_found += 1

        found_count = 0
        for post in thread['posts']:
            if 'body' not in post.keys():
                print("Can't find %s in URL: %s" % (post['id'], thread['url']))
                # TODO: Use BigQuery for not found posts
            else:
                found_count += 1

        print('Found %s posts out of %s' % (found_count, len(thread['posts'])))
        posts_found += found_count
        posts_total += len(thread['posts'])

        dump_with_reddit.write(json.dumps(thread)+'\n')

        # To keep within Reddit API limits
        time.sleep(2)

    print("\tFINISHED")
    print("Threads: Found {} out of {} [{}]"
          .format(threads_found, threads_total,
                  100*threads_found/threads_total))
    print("Posts: Found {} out of {} [{}]"
          .format(posts_found, posts_total,
                  100*posts_found/posts_total))

    # logger = logging.getLogger(__name__)
    # logger.info('making final data set from raw data')


if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
