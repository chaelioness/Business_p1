"""API-free text baseline v1; reusable inside the final src/model.py.

Feature calculation reads no labels and never fits on a whole-data statistic.
DuckDB is required only to project raw Parquet into a compact, local event index.
The index may contain future rows; all queries use aligned, strict < cutoff bounds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'chaemin-text-v1.0'
NEWS_FEATURES = ['news_log_count_overnight', 'news_spike20']
REDDIT_FEATURES = ['reddit_log_count_overnight', 'reddit_spike20', 'reddit_persistence3']
JOINT_FEATURES = ['text_both_spike']
CONTEXT_FEATURES = ['text_overnight_hours']
BASELINE_FEATURES = NEWS_FEATURES + REDDIT_FEATURES + JOINT_FEATURES + CONTEXT_FEATURES
OPTIONAL_FEATURES = ['news_z20', 'reddit_z20', 'news_tone_mean_overnight',
                     'news_tone_negative_ratio_overnight', 'text_spike_product']
QUALITY_COLUMNS = ['qc_news_gap_24h', 'qc_news_gap_overnight', 'qc_news_symbol_unseen',
                   'qc_news_history_fraction20', 'qc_reddit_gap_24h',
                   'qc_reddit_gap_overnight', 'qc_reddit_history_fraction20']
STEP = np.timedelta64(30, 'm')
MIN_HISTORY = 15              # fixed engineering default, not label-tuned
HISTORY = 20
GAP_BINS = 12                 # >= 6h with no record in the entire supplied source
# Conservative ambiguity rule; NEW tickers are supported without a fixed 50-stock list.
AMBIGUOUS_WORDS = {'COST', 'NOW', 'WELL', 'COP', 'LIN'}
FORBIDDEN_COLUMNS = {'score', 'n_comments', 'num_comments_reported',
                     'retrieved_on', 'retrieved_2nd_on', 'label', 'ret_pct'}


def market_time(value):
    t = pd.Timestamp(value)
    return t.tz_convert('America/New_York').tz_localize(None) if t.tzinfo else t


def _q(value):
    return "'" + str(value).replace("'", "''") + "'"


def _symbols(values):
    result = sorted(set(map(str, values)))
    if not result or any(not re.fullmatch(r'[A-Z][A-Z0-9.-]*', s) for s in result):
        raise ValueError('symbols must be nonempty uppercase ticker identifiers')
    return result


def ticker_pattern(symbols):
    spellings = list(_symbols(symbols))
    spellings += [s.replace('-', '.') for s in spellings if '-' in s]
    return r'(?i)\$?\b(?:' + '|'.join(re.escape(s) for s in sorted(set(spellings), key=len, reverse=True)) + r')\b'


def accepted_mentions(text, symbols):
    """Independent Python implementation, also useful for manual matching audits."""
    symbols = set(_symbols(symbols))
    ambiguous = {s for s in symbols if len(s) <= 2} | AMBIGUOUS_WORDS
    found = set()
    for token in re.findall(ticker_pattern(symbols), text or ''):
        s = token.lstrip('$').upper().replace('.', '-')
        if s in symbols and (token.startswith('$') or (token == token.upper() and s not in ambiguous)):
            found.add(s)
    return found


def source_fingerprint(folder):
    """Stale/wrong-folder cache guard; metadata is NOT a predictive feature."""
    folder = Path(folder).resolve()
    files = [folder / 'news.parquet'] + sorted((folder / 'reddit').glob('*.parquet'))
    if not files[0].is_file() or len(files) < 2:
        raise FileNotFoundError('news.parquet and reddit/*.parquet are required')
    return {'folder': str(folder), 'files': [
        {'path': str(p.relative_to(folder)), 'size': p.stat().st_size,
         'mtime_ns': p.stat().st_mtime_ns} for p in files]}


def prepare_index(folder, symbols, cache_dir=None, *, log=print):
    """Read permitted columns only; one bounded-memory scan, no API calls.

    News is deduplicated by (URL, symbol), earliest known_at. Missing URLs fall back
    to title. Reddit counts each ID-symbol once per file at earliest created_et.
    Store 30-minute buckets: every supported boundary is on this exact grid.
    """
    import duckdb
    import pyarrow.parquet as pq

    folder = Path(folder).resolve()
    symbols = _symbols(symbols)
    fingerprint = source_fingerprint(folder)
    manifest = {'version': VERSION, 'symbols': symbols, 'source': fingerprint,
                'bucket_minutes': 30, 'min_history': MIN_HISTORY, 'gap_hours': 6,
                'allowed_news_columns': ['known_at', 'symbols', 'url', 'title', 'tone'],
                'allowed_reddit_columns': ['created_et', 'id/post_id', 'title(posts)', 'body']}
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        meta = cache / 'manifest.json'
        if meta.exists():
            prior = json.loads(meta.read_text())
            if prior == manifest and all((cache / (s + '.parquet')).exists()
                                          for s in ['news', 'reddit', 'feeds']):
                log('Reusing matching text index')
                return TextFeatureBuilder.from_index(cache)
            raise ValueError('Cache differs from source/symbols/version. Use a NEW cache directory; do not silently reuse.')
    db = duckdb.connect()
    db.execute("SET memory_limit='1200MB'")
    db.execute('SET threads=4')
    if cache is not None:
        (cache / '.spill').mkdir(exist_ok=True)
        db.execute(f"SET temp_directory={_q(cache / '.spill')}")
    universe = pd.DataFrame({'symbol': symbols})
    db.register('universe', universe)
    bucket = "time_bucket(INTERVAL '30 minutes', t)"
    feeds = []
    try:
        news_path = folder / 'news.parquet'
        needed = {'known_at', 'symbols', 'url', 'title', 'tone'}
        if not needed <= set(pq.ParquetFile(news_path).schema_arrow.names):
            raise ValueError('Unexpected news schema; inspect before adapting')
        log('Index news: metadata/tone only, no body')
        db.execute(f'''CREATE TEMP VIEW raw_news AS
            SELECT known_at, symbols, url, title, tone FROM read_parquet({_q(news_path)})''')
        bad = db.execute("SELECT count(*) FROM raw_news WHERE known_at IS NULL").fetchone()[0]
        if bad:
            raise ValueError(f'News timestamps missing: {bad}; cannot assign availability safely')
        feeds.append(db.execute(f'''SELECT time_bucket(INTERVAL '30 minutes',known_at) t,
                         count(*) n, 'news' AS "source" FROM raw_news GROUP BY 1''').df())
        news = db.execute(f'''WITH tagged AS (
            SELECT known_at t, trim(tag) symbol, tone,
              coalesce(nullif(url,''),nullif(title,'')) doc,
              row_number() OVER(PARTITION BY coalesce(nullif(url,''),nullif(title,'')),trim(tag)
                                ORDER BY known_at,coalesce(tone,0)) rn
            FROM raw_news,unnest(string_split(symbols,',')) u(tag)
        ) SELECT {bucket} t, symbol, count(*) n,
                   count(tone) tone_n, sum(tone) tone_sum,
                   count(*) FILTER(WHERE tone<0) negative_n
          FROM tagged WHERE (rn=1 OR doc IS NULL) AND symbol IN(SELECT symbol FROM universe)
          GROUP BY 1,2 ORDER BY 2,1''').df()
        reddit_parts = []
        pattern = ticker_pattern(symbols)
        ambiguous = sorted({s for s in symbols if len(s) <= 2} | AMBIGUOUS_WORDS)
        ambiguous_sql = ','.join(map(_q, ambiguous))
        for path in sorted((folder / 'reddit').glob('*.parquet')):
            pieces = path.name.split('.')
            if len(pieces) != 3 or pieces[1] not in ('posts', 'comments'):
                raise ValueError(f'Unexpected Reddit filename: {path.name}')
            sub, kind, _ = pieces
            is_post = kind == 'posts'
            key = 'post_id' if is_post else 'id'
            cols = ['created_et', key, 'body'] + (['title'] if is_post else [])
            if not set(cols) <= set(pq.ParquetFile(path).schema_arrow.names):
                raise ValueError(f'Unexpected Reddit schema: {path.name}')
            assert not set(cols) & FORBIDDEN_COLUMNS
            log('Index Reddit: ' + path.name)
            db.execute(f'''CREATE OR REPLACE TEMP VIEW raw_reddit AS
                           SELECT {','.join(cols)} FROM read_parquet({_q(path)})''')
            if db.execute(f'SELECT count(*) FROM raw_reddit WHERE created_et IS NULL OR {key} IS NULL').fetchone()[0]:
                raise ValueError(f'Missing Reddit availability/ID: {path.name}')
            feeds.append(db.execute(f'''SELECT time_bucket(INTERVAL '30 minutes',created_et) t,
                              count(*) n, 'reddit' AS "source" FROM raw_reddit GROUP BY 1''').df())
            text = "coalesce(title,'') || ' ' || coalesce(body,'')" if is_post else "coalesce(body,'')"
            reddit_parts.append(db.execute(f'''WITH tokens AS (
                SELECT created_et t,{key} doc,
                  unnest(list_distinct(regexp_extract_all({text},{_q(pattern)}))) token
                FROM raw_reddit
            ), mapped AS (
                SELECT *,replace(upper(ltrim(token,'$')),'.','-') symbol FROM tokens
            ), documents AS (
                SELECT doc,symbol,min(t) t FROM mapped
                WHERE symbol IN(SELECT symbol FROM universe)
                  AND (starts_with(token,'$') OR (token=upper(token) AND symbol NOT IN({ambiguous_sql})))
                GROUP BY 1,2
            ) SELECT {bucket} t,symbol,count(*) n FROM documents GROUP BY 1,2''').df())
        reddit = pd.concat(reddit_parts, ignore_index=True).groupby(['symbol', 't'], as_index=False).n.sum()
        feed = pd.concat(feeds, ignore_index=True).groupby(['source', 't'], as_index=False).n.sum()
        tables = {'news': news, 'reddit': reddit, 'feeds': feed}
        if cache is not None:
            for name, frame in tables.items():
                frame.to_parquet(cache / (name + '.parquet'), index=False)
            # Manifest is written LAST; incomplete caches cannot pass reuse checks.
            (cache / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return TextFeatureBuilder(news, reddit, feed, symbols, manifest)
    finally:
        db.close()


class _SeriesIndex:
    def __init__(self, frame, columns=('n',)):
        frame = frame.sort_values('t')
        timestamps = pd.to_datetime(frame.t)
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_convert('America/New_York').dt.tz_localize(None)
        self.t = timestamps.to_numpy(dtype='datetime64[ns]')
        if np.isnat(self.t).any() or (self.t.astype('int64') % (30*60*1_000_000_000)).any():
            raise ValueError('Index requires finite, aligned 30-minute ET buckets')
        self.first = self.t[0] if len(self.t) else np.datetime64('NaT', 'ns')
        self.sums = {c: np.r_[0., np.nan_to_num(frame[c].to_numpy(dtype=float)).cumsum()]
                     for c in columns}

    def sums_between(self, lo, hi, column='n'):
        a = np.searchsorted(self.t, lo, side='left')
        b = np.searchsorted(self.t, hi, side='left')
        return self.sums[column][b] - self.sums[column][a]

    def gap(self, lo, hi):
        """Any >=6h run without a source record, using only the requested interval.

        This is a missing-coverage heuristic, NOT proof of a collection outage.
        It catches internal and leading/trailing gaps, including feed restart days.
        """
        lo, hi = np.datetime64(lo, 'ns'), np.datetime64(hi, 'ns')
        bins = int((hi - lo) / STEP)
        if bins <= 0:
            raise ValueError('Window must have positive length')
        a, b = np.searchsorted(self.t, [lo, hi], side='left')
        positions = ((self.t[a:b] - lo) / STEP).astype(int)
        empty_runs = np.diff(np.r_[-1, positions, bins]) - 1
        return int(empty_runs.max() >= GAP_BINS)


class TextFeatureBuilder:
    """Reusable index. build(day) returns symbol + model features + QC columns.

    Cache contains NO label, return, author, body, score or n_comments.
    Same calculations are used for batch training and single-day inference.
    """
    def __init__(self, news, reddit, feeds, symbols, manifest=None):
        self.symbols = _symbols(symbols)
        self.manifest = manifest
        self.news = {s: _SeriesIndex(news[news.symbol.eq(s)], ('n', 'tone_n', 'tone_sum', 'negative_n'))
                     for s in self.symbols}
        self.reddit = {s: _SeriesIndex(reddit[reddit.symbol.eq(s)]) for s in self.symbols}
        self.feeds = {s: _SeriesIndex(feeds[feeds.source.eq(s)]) for s in ('news', 'reddit')}

    @classmethod
    def from_index(cls, cache_dir, *, data_dir=None):
        path = Path(cache_dir)
        m = json.loads((path / 'manifest.json').read_text())
        if m['version'] != VERSION:
            raise ValueError('Feature version mismatch; rebuild index')
        check_dir = data_dir if data_dir is not None else m['source']['folder']
        if m['source'] != source_fingerprint(check_dir):
            raise ValueError('Index does not match supplied data directory')
        return cls(*(pd.read_parquet(path / (s + '.parquet')) for s in ['news', 'reddit', 'feeds']),
                   m['symbols'], m)

    def build(self, day, *, include_optional=False):
        if self.manifest is not None and Path(day.ds.dir).resolve() != Path(self.manifest['source']['folder']):
            raise ValueError('Builder belongs to a different dataset. Build a new index for this dataset.')
        if self.manifest is not None and self.manifest['source'] != source_fingerprint(day.ds.dir):
            raise ValueError('Source files changed after indexing; rebuild rather than silently returning stale features.')
        # Dataset.dates is used for the calendar only, not its underlying return values.
        calendar = day.ds.dates(until=day.target)
        return self.build_at(day.date, day.target, day.symbols, calendar,
                             cutoff=day.cutoff, include_optional=include_optional)

    def build_at(self, date, target, symbols, prior_calendar, *, cutoff=None, include_optional=False):
        date, target = market_time(date).normalize(), market_time(target).normalize()
        cutoff = market_time(cutoff) if cutoff is not None else target + pd.Timedelta(hours=9, minutes=30)
        if cutoff != target + pd.Timedelta(hours=9, minutes=30) or date >= target:
            raise ValueError('Expected date < target and target 09:30 cutoff')
        symbols = _symbols(symbols)
        if not set(symbols) <= set(self.symbols):
            raise ValueError('New symbols require a newly prepared index. Never fill unknown cached symbols with 0.')
        calendar = sorted({market_time(d).normalize() for d in prior_calendar if market_time(d) < target})
        if not calendar or calendar[-1] != date:
            raise ValueError('Calendar must end at the supplied reference date')
        # 22 prior cutoffs allow past20 for each of the last3 spike observations.
        ends = np.array([d + pd.Timedelta(hours=9, minutes=30) for d in calendar[-22:]] + [cutoff], dtype='datetime64[ns]')
        starts = ends - np.timedelta64(24, 'h')
        overnight = np.datetime64(date + pd.Timedelta(hours=16), 'ns')
        cutoff64 = np.datetime64(cutoff, 'ns')
        gap = {s: np.array([self.feeds[s].gap(a, b) for a, b in zip(starts, ends)], dtype=bool)
               for s in ('news', 'reddit')}
        overnight_gap = {s: self.feeds[s].gap(overnight, cutoff64) for s in ('news', 'reddit')}
        rows = []
        for symbol in symbols:
            indexes = {'news': self.news[symbol], 'reddit': self.reddit[symbol]}
            values, spikes, zscores, fractions = {}, {}, {}, {}
            for source, ix in indexes.items():
                count = ix.sums_between(starts, ends)
                ok = ~gap[source]
                if source == 'news':
                    # No hardcoded AMZN/JNJ/V. A symbol is unobserved UNTIL a tag is seen.
                    ok = ok & (ix.first < ends)
                values[source] = np.where(ok, count, np.nan)
                ratios = np.full(len(ends), np.nan)
                z = np.full(len(ends), np.nan)
                fraction = np.zeros(len(ends))
                for i in range(len(ends)):
                    past = values[source][max(0, i-HISTORY):i]
                    past = past[np.isfinite(past)]
                    fraction[i] = len(past) / HISTORY
                    if len(past) >= MIN_HISTORY and np.isfinite(values[source][i]):
                        ratios[i] = (values[source][i]+1) / (past.mean()+1)
                        sd = past.std(ddof=1)
                        if sd > 0:
                            z[i] = (values[source][i]-past.mean()) / sd
                spikes[source], zscores[source], fractions[source] = ratios, z, fraction
            news_seen = bool(indexes['news'].first < cutoff64)
            no = indexes['news'].sums_between(overnight, cutoff64)
            ro = indexes['reddit'].sums_between(overnight, cutoff64)
            reddit_last3 = spikes['reddit'][-3:]
            persistence = (float(((reddit_last3 >= 2) & (values['reddit'][-3:] >= 3)).sum())
                           if len(reddit_last3) == 3 and np.isfinite(reddit_last3).all() else np.nan)
            joint_ok = np.isfinite(spikes['news'][-1]) and np.isfinite(spikes['reddit'][-1])
            both = float(spikes['news'][-1] >= 2 and values['news'][-1] >= 5
                         and spikes['reddit'][-1] >= 2 and values['reddit'][-1] >= 3) if joint_ok else np.nan
            row = dict(symbol=symbol,
                news_log_count_overnight=float(np.log1p(no)) if news_seen and not overnight_gap['news'] else np.nan,
                news_spike20=spikes['news'][-1],
                reddit_log_count_overnight=float(np.log1p(ro)) if not overnight_gap['reddit'] else np.nan,
                reddit_spike20=spikes['reddit'][-1], reddit_persistence3=persistence,
                text_both_spike=both, text_overnight_hours=(cutoff-(date+pd.Timedelta(hours=16))).total_seconds()/3600,
                qc_news_gap_24h=int(gap['news'][-1]), qc_news_gap_overnight=overnight_gap['news'],
                qc_news_symbol_unseen=int(not news_seen), qc_news_history_fraction20=fractions['news'][-1],
                qc_reddit_gap_24h=int(gap['reddit'][-1]), qc_reddit_gap_overnight=overnight_gap['reddit'],
                qc_reddit_history_fraction20=fractions['reddit'][-1])
            if include_optional:
                nt = indexes['news'].sums_between(overnight, cutoff64, 'tone_n')
                tone_ok = news_seen and not overnight_gap['news'] and nt > 0
                row.update(news_z20=zscores['news'][-1], reddit_z20=zscores['reddit'][-1],
                    news_tone_mean_overnight=indexes['news'].sums_between(overnight, cutoff64, 'tone_sum')/nt if tone_ok else np.nan,
                    news_tone_negative_ratio_overnight=indexes['news'].sums_between(overnight, cutoff64, 'negative_n')/nt if tone_ok else np.nan,
                    text_spike_product=spikes['news'][-1]*spikes['reddit'][-1] if joint_ok else np.nan)
            rows.append(row)
        out = pd.DataFrame(rows)
        expected = ['symbol'] + BASELINE_FEATURES + QUALITY_COLUMNS + (OPTIONAL_FEATURES if include_optional else [])
        out = out[expected]
        if np.isinf(out.select_dtypes(include='number').to_numpy()).any():
            raise AssertionError('Infinite feature value')
        return out


def build_text_features(day, builder, *, include_optional=False):
    """Thin handoff interface. Build/reuse `builder` once, never once per stock/day."""
    return builder.build(day, include_optional=include_optional)
