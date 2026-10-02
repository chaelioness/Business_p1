"""Meaningful invariants: point-in-time, outages, new symbols and integration parity."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from features import (TextFeatureBuilder, prepare_index, BASELINE_FEATURES, QUALITY_COLUMNS,
                      accepted_mentions, FORBIDDEN_COLUMNS)


def fixture():
    all_bins = pd.date_range('2024-01-01','2024-03-20',freq='30min')
    dates = pd.date_range('2024-01-01','2024-03-20',freq='D')
    times = dates+pd.Timedelta(hours=6)
    news = pd.DataFrame({'t':times,'symbol':'AAA','n':1.,'tone_n':1.,'tone_sum':.2,'negative_n':0.})
    reddit = news[['t','symbol','n']].copy()
    feeds = pd.concat([pd.DataFrame({'t':all_bins,'source':s,'n':1}) for s in ['news','reddit']],ignore_index=True)
    return news,reddit,feeds


def build(builder, symbols=('AAA','BBB')):
    return builder.build_at('2024-03-04','2024-03-05',symbols,
        pd.bdate_range('2024-01-01','2024-03-04'),include_optional=True)


class Invariants(unittest.TestCase):
    def test_cutoff_and_future_prefix_invariance(self):
        news,reddit,feeds=fixture()
        expected=build(TextFeatureBuilder(news,reddit,feeds,['AAA','BBB']))
        cutoff=pd.Timestamp('2024-03-05 09:30')
        n=news[news.t<cutoff].copy();r=reddit[reddit.t<cutoff].copy();f=feeds[feeds.t<cutoff].copy()
        assert_frame_equal(expected,build(TextFeatureBuilder(n,r,f,['AAA','BBB'])))
        # Exactly cutoff and later new-symbol observations must not change historical features.
        extra=pd.DataFrame({'t':[cutoff,cutoff+pd.Timedelta(days=1)],'symbol':['AAA','BBB'],
                            'n':[999,999],'tone_n':[999,999],'tone_sum':[-999,-999],'negative_n':[999,999]})
        assert_frame_equal(expected,build(TextFeatureBuilder(pd.concat([n,extra]),pd.concat([r,extra[['t','symbol','n']]]),feeds,['AAA','BBB'])))

    def test_unseen_news_is_missing_not_no_interest(self):
        n,r,f=fixture();out=build(TextFeatureBuilder(n,r,f,['AAA','BBB'])).set_index('symbol')
        self.assertTrue(pd.isna(out.loc['BBB','news_log_count_overnight']))
        self.assertEqual(out.loc['BBB','qc_news_symbol_unseen'],1)
        self.assertEqual(out.loc['BBB','reddit_log_count_overnight'],0)

    def test_current_observation_excluded_from_baseline(self):
        n,r,f=fixture();n.loc[n.t.eq(pd.Timestamp('2024-03-05 06:00')),'n']=10
        out=build(TextFeatureBuilder(n,r,f,['AAA','BBB'])).set_index('symbol')
        self.assertAlmostEqual(out.loc['AAA','news_spike20'],5.5)
        self.assertAlmostEqual(out.loc['AAA','news_log_count_overnight'],np.log1p(10))

    def test_internal_gap_not_only_last_seen(self):
        n,r,f=fixture()
        # Source recovered by cutoff but 7h internal absence still invalidates its 24h window.
        remove=f.source.eq('news')&f.t.ge('2024-03-04 15:00')&f.t.lt('2024-03-04 22:00')
        out=build(TextFeatureBuilder(n,r,f[~remove],['AAA','BBB'])).set_index('symbol')
        self.assertEqual(out.loc['AAA','qc_news_gap_24h'],1)
        self.assertTrue(pd.isna(out.loc['AAA','news_spike20']))

    def test_restart_has_insufficient_clean_history(self):
        n,r,f=fixture();remove=f.source.eq('news')&f.t.ge('2024-02-10')&f.t.lt('2024-03-03')
        out=build(TextFeatureBuilder(n,r,f[~remove],['AAA','BBB'])).set_index('symbol')
        self.assertEqual(out.loc['AAA','qc_news_gap_24h'],0)
        self.assertLess(out.loc['AAA','qc_news_history_fraction20'],.75)
        self.assertTrue(pd.isna(out.loc['AAA','news_spike20']))

    def test_subset_universe_invariance_and_unknown_guard(self):
        n,r,f=fixture();b=TextFeatureBuilder(n,r,f,['AAA','BBB'])
        assert_frame_equal(build(b).iloc[[0]].reset_index(drop=True),build(b,['AAA']))
        with self.assertRaises(ValueError):build(b,['NEW'])

    def test_no_day_y_read(self):
        n,r,f=fixture();b=TextFeatureBuilder(n,r,f,['AAA','BBB'])
        class Day:
            date=pd.Timestamp('2024-03-04');target=pd.Timestamp('2024-03-05');cutoff=target+pd.Timedelta(hours=9,minutes=30)
            symbols=['AAA','BBB'];ds=SimpleNamespace(dates=lambda until:pd.bdate_range('2024-01-01','2024-03-04'))
            @property
            def y(self):raise AssertionError('Forbidden label access')
        assert_frame_equal(b.build(Day(),include_optional=True),build(b))
        self.assertFalse(set(BASELINE_FEATURES+QUALITY_COLUMNS)&FORBIDDEN_COLUMNS)

    def test_matching_precision_and_dedup(self):
        self.assertEqual(accepted_mentions('NOW is now well. MS windows. $NOW $WELL AMD AMD amd $ms BRK.B',['NOW','WELL','MS','AMD','BRK-B']),{'NOW','WELL','MS','AMD','BRK-B'})
        self.assertEqual(accepted_mentions('cost of living well now v=123',['COST','WELL','NOW','V']),set())
        self.assertEqual(accepted_mentions('New company $ZZZ and ZZZ', ['ZZZ']),{'ZZZ'})

    def test_index_single_day_and_batch_parity(self):
        # Tiny raw fixture verifies 09:29:59 included, 09:30 excluded and cache round trip.
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data';cache=Path(tmp)/'cache';(data/'reddit').mkdir(parents=True)
            t=pd.date_range('2024-01-01','2024-03-06',freq='h')
            raw=pd.DataFrame({'known_at':t,'symbols':'AAA','url':['u'+str(i) for i in range(len(t))],'title':'event','tone':.2})
            raw=pd.concat([raw,pd.DataFrame({'known_at':pd.to_datetime(['2024-03-05 09:29:59','2024-03-05 09:30:00']), 'symbols':'AAA','url':['last_allowed','at_cutoff'],'title':'test','tone':[.3,.4]})],ignore_index=True)
            raw.to_parquet(data/'news.parquet',index=False)
            rr=pd.DataFrame({'created_et':t,'post_id':[str(i) for i in range(len(t))],'title':'AAA AAA','body':'$AAA','score':999,'n_comments':999})
            rr.to_parquet(data/'reddit/stocks.posts.parquet',index=False)
            b=prepare_index(data,['AAA','BBB'],cache,log=lambda _:None)
            out=build(b).set_index('symbol')
            # Overnight 16:00..09:30 has 18 hourly timestamps plus 09:29:59.
            self.assertAlmostEqual(out.loc['AAA','news_log_count_overnight'],np.log1p(19))
            assert_frame_equal(build(b),build(TextFeatureBuilder.from_index(cache,data_dir=data)))
            # Future rows and forbidden engagement do not affect earlier output.
            raw=raw[raw.known_at<pd.Timestamp('2024-03-05 09:30')];raw.to_parquet(data/'news.parquet',index=False)
            rr=rr[rr.created_et<pd.Timestamp('2024-03-05 09:30')].copy();rr['score']=-123;rr['n_comments']=0;rr.to_parquet(data/'reddit/stocks.posts.parquet',index=False)
            short=prepare_index(data,['AAA','BBB'],log=lambda _:None)
            assert_frame_equal(build(b),build(short))
            with self.assertRaises(ValueError):TextFeatureBuilder.from_index(cache,data_dir=data)
            with self.assertRaises(ValueError):prepare_index(data,['AAA','BBB'],cache,log=lambda _:None)


if __name__=='__main__':unittest.main(verbosity=2)
