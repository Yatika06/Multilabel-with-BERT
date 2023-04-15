import pandas as pd
import argparse
from pathlib import Path
import os


def main(cfg):
    if cfg.mode == 'make_dataset':
        make_data(input_path=cfg.input_dir, output_dir=cfg.output_dir)


def make_data(input_path: Path, output_dir: Path):
    subjects = pd.read_csv(os.path.join(input_path, 'train.csv'))
    subjects['text'] = [title + '. ' + abstract for (title, abstract) in zip(subjects['TITLE'], subjects['ABSTRACT'])]
    targets = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology',
               'Quantitative Finance']


    # no preprocessing required for BERT since it used all the punctuation and other info
    # preprocess = Preprocessing(is_lower=is_lower, is_remove_punct=is_remove_punct, stem_method=stem_method,
    #                            is_stopwords_removal=is_stopwords_removal)
    # lines = subjects['text']
    # data = preprocess.run(lines)

    new_set = pd.DataFrame()
    new_set['text'] = subjects['text']
    new_set[targets] = subjects[targets]
    new_set.to_csv(path_or_buf=os.path.join(output_dir, 'processed.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data preprocessing utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, help='type of operation')
    parser.add_argument("--input-dir", type=Path, help='Path to raw input data file')
    parser.add_argument("--output-dir", type=Path, help='Path to out file')
    # parser.add_argument("--stem-method", type=str, help='Method to perform stemming', default=None)
    # parser.add_argument("--is-stopwords-removal", type=bool, help='If stopwords needs to be removed', default=False)
    # parser.add_argument("--is-remove-punct", type=bool, help='If remove punctuations', default=True)
    # parser.add_argument("--is-lower", type=bool, help='If lower words', default=True)

    config = parser.parse_args()
    main(config)
