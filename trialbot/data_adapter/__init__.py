from .geoquery import GeoQueryDatasetReader
from .tab_sep_seqs import TabSepDatasetReader, TabSepJiebaCutReader, TabSepSharedVocabReader, TabSepCharReader
from .spider import SpiderDatasetReader
from .keyword_seqs import CharKeywordReader, CharExtractedKeywordReader

DATA_READERS = dict()
DATA_READERS['geoquery'] = GeoQueryDatasetReader
DATA_READERS['tab_sep'] = TabSepDatasetReader
DATA_READERS['tab_sep_shared'] = TabSepSharedVocabReader
DATA_READERS['tab_sep_jieba'] = TabSepJiebaCutReader
DATA_READERS['tab_sep_char'] = TabSepCharReader
DATA_READERS['spider'] = SpiderDatasetReader
DATA_READERS['tab_sep_char_key'] = CharKeywordReader
DATA_READERS['tab_sep_char_key_processed'] = CharExtractedKeywordReader

import data_adapter.ns_vocabulary
import data_adapter.iterator
import data_adapter.translator
import data_adapter.dataset
