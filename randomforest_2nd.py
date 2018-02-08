import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
from sklearn.feature_extraction import stop_words
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from subprocess import check_output

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")

def wordCount(text):
    # convert to lower case and strip regex
    try:
         # convert to lower case and strip regex
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        # tokenize
        # words = nltk.word_tokenize(clean_txt)
        # remove words in stop words
        words = [w for w in txt.split(" ") \
                 if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]
        return len(words)
    except: 
        return 0
    
def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')

def to_number(dataset):
    dataset.general_cat = dataset.general_cat.cat.codes
    dataset.subcat_1 = dataset.subcat_1.cat.codes
    dataset.subcat_2 = dataset.subcat_2.cat.codes
    dataset.brand_name = dataset.brand_name.cat.codes


def main():
    train = pd.read_csv("../input/train.tsv", sep='\t')
    test = pd.read_csv("../input/test.tsv", sep='\t')

    submission: pd.DataFrame = test[['test_id']]

    train_y = np.log(train['price'] + 1)

    train_df = train.drop(['price', 'train_id'], axis=1)
    test_df = test.drop(['test_id'], axis=1)

    train_df['is_trained'] = 1
    test_df['is_trained'] = 0

    combined = pd.concat([train_df, test_df])

    combined['general_cat'], combined['subcat_1'], combined['subcat_2'] = \
    zip(*combined['category_name'].apply(lambda x: split_cat(x)))

    combined['desc_len'] = combined['item_description'].apply(lambda x: wordCount(x))

    handle_missing_inplace(combined)
    to_categorical(combined)
    to_number(combined)

    combined_df = combined.drop(['name', 'category_name', 'item_description'], axis=1)

    train_x = combined_df[combined_df.is_trained == 1].drop(['is_trained'], axis=1)
    test_x = combined_df[combined_df.is_trained == 0].drop(['is_trained'], axis=1)

    max_features_range = ['auto', 'sqrt', 0.2]
    min_samples_leaf_range = [50, 60, 70]
    param_grid = dict(max_features = max_features_range, min_samples_leaf = min_samples_leaf_range)
    cv = sklearn.model_selection.KFold(n_splits=10, shuffle=False, random_state=44)
    grid = GridSearchCV(RandomForestRegressor(n_estimators = 10, n_jobs= -1, random_state= 44), param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid.fit(train_x, train_y)
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    
    test_y = grid.predict(test_x)
    submission['price'] = np.expm1(test_y)
    submission.to_csv("1st-randomforest.csv", index=False)


if __name__ == '__main__':
    main()

























