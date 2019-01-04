# -*- coding: utf-8 -*-
import os
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def main(proj_root):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    print("===== Preprocessing data =====")
    print("Loading raw data", end=' ')
    filepath = os.path.join(proj_root, 'data', 'raw',
                            'coarse_discourse_dump_reddit.json')

    # 'lines' so pandas reads each line of file as a json object
    raw_df = pd.read_json(filepath, lines=True)

    print("- Done!")

    # Converts 'is_self_post' column to boolean type
    raw_df['is_self_post'] = raw_df['is_self_post'].fillna(0).astype('bool')

    # Creates DataFrame with each post (of the threads in raw_df) as a line
    print("Creating posts DataFrame - 0%")
    dpproc = pd.DataFrame()
    for i in range(1, len(raw_df)):
        print("\rCreating posts DataFrame - {0:.2f}%"
              .format(100*i/len(raw_df)))
        thread = raw_df.iloc[i]
        dposts = pd.DataFrame(thread['posts'])

        # Replicates threads features
        dposts['is_self_post'] = thread['is_self_post']
        dposts['subreddit'] = thread['subreddit']
        dposts['thread_title'] = thread['title']

        dpproc = pd.concat([dpproc, dposts], ignore_index=True, sort=True)

    print("\rCreating posts DataFrame - Done!")

    total_posts_raw = len(dpproc)
    print("\t{} posts gathered.".format(total_posts_raw))

    #   Cleaning data
    print("Cleaning data - 0/4 steps", end='\r')

    # 'is_first_post' column is useless, since it is only True
    # when 'in_reply_to' is NaN
    dpproc.drop(columns=['is_first_post'], inplace=True)
    print("Cleaning data - 1/4 steps", end='\r')

    # NaN values in 'post_depth' mean it is the first post,
    # so 0th depth
    dpproc['post_depth'] = dpproc['post_depth'].fillna(0)
    print("Cleaning data - 2/4 steps", end='\r')

    # If we don't have the post's text it is useles for us
    dpproc.dropna(subset=['body'], inplace=True)
    print("Cleaning data - 3/4 steps", end='\r')

    # We assume that NaN values at 'author' are deleted accounts
    dpproc['author'] = dpproc['author'].fillna('deleted')
    print("Cleaning data - Done!")

    # Results
    total_posts = len(dpproc)
    print("Total posts after cleaning: {} ({0:.2f}%)"
          .format(total_posts, 100*total_posts/total_posts_raw))

    #   Saving the final DataFrame
    print("Saving dataframe", end=' ')
    dpproc.to_pickle(os.path.join(proj_root, 'data', 'interim',
                                  'clean_data.pkl'))
    print("- Done!")

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
