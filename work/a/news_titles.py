"""장 마감 후(기준일 16:00 ~ cutoff) 기사 제목을 종목별로 모은 표. LLM 없이 텍스트 피처를 만들 재료.

    t = titles(days)        # date, target, symbol, title, source, tone  (cache/news_titles.parquet)

day.news 로만 읽고 guard 아래에서 만듦. (url, symbol) 중복은 첫 기사만. 정답은 넣지 않음.
"""

import pandas as pd

from lab.cv import CACHE
from lab.leak import guard

PATH = CACHE / "news_titles.parquet"
CLOSE = pd.Timedelta(hours=16)


def one(day):
    n = day.news(since=day.date + CLOSE)
    if not len(n):
        return pd.DataFrame(columns=["symbol", "title", "source", "tone", "known_at"])
    n = n.assign(symbol=n.symbols.astype(str).str.split(",")).explode("symbol")
    n["symbol"] = n.symbol.str.strip()
    n = n[n.symbol.isin(set(day.symbols))].sort_values("known_at")
    n = n.drop_duplicates(["url", "symbol"])
    return n[["symbol", "title", "source", "tone", "known_at"]]


def titles(days, verbose=True):
    old = pd.read_parquet(PATH) if PATH.exists() else None
    have = set(old.date) if old is not None else set()
    rows = []
    for i, d in enumerate(days, 1):
        if d.date in have:
            continue
        x = one(guard(d))
        rows.append(x.assign(date=d.date, target=d.target))
        if verbose and i % 100 == 0:
            print(f"  뉴스 제목 {i}/{len(days)}일", flush=True)
    if rows:
        new = pd.concat([old, *rows] if old is not None else rows, ignore_index=True)
        new.to_parquet(PATH, index=False)
        old = new
    need = {d.date for d in days}
    return old[old.date.isin(need)].reset_index(drop=True)


if __name__ == "__main__":
    from lab.folds import HOLDOUT
    from src import Dataset

    ds = Dataset()
    t = titles(ds.split(HOLDOUT)[0])
    print(t.shape, t.date.max())
