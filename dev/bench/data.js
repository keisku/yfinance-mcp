window.BENCHMARK_DATA = {
  "lastUpdate": 1767202230475,
  "repoUrl": "https://github.com/keisku/yfinance-mcp",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "keisuke.umegaki.630@gmail.com",
            "name": "Keisuke Umegaki",
            "username": "keisku"
          },
          "committer": {
            "email": "keisuke.umegaki.630@gmail.com",
            "name": "Keisuke Umegaki",
            "username": "keisku"
          },
          "distinct": true,
          "id": "e2b794979c2bf143b5b5457fc1beecbd3696494a",
          "message": "fix(ci): Fix coverage path and add benchmark push permission\n\n- Fix coverage source path: yahoo_finance_mcp -> src/yfinance_mcp\n- Add contents: write permission for benchmark auto-push to gh-pages",
          "timestamp": "2026-01-01T02:29:33+09:00",
          "tree_id": "d94a799915c1a336fcb5abf19181ac853237a918",
          "url": "https://github.com/keisku/yfinance-mcp/commit/e2b794979c2bf143b5b5457fc1beecbd3696494a"
        },
        "date": 1767202230086,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 295.6733542452416,
            "unit": "iter/sec",
            "range": "stddev: 0.00007100282274917797",
            "extra": "mean: 3.382110648937833 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 293.2285708444173,
            "unit": "iter/sec",
            "range": "stddev: 0.00008015118505306753",
            "extra": "mean: 3.4103088833406523 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 293.744289575547,
            "unit": "iter/sec",
            "range": "stddev: 0.00007620140678897835",
            "extra": "mean: 3.4043214982833354 msec\nrounds: 293"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 77.7170381196452,
            "unit": "iter/sec",
            "range": "stddev: 0.00011523975662594431",
            "extra": "mean: 12.867191341755746 msec\nrounds: 79"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 41.88750114646152,
            "unit": "iter/sec",
            "range": "stddev: 0.0002493611165678242",
            "extra": "mean: 23.87346995237208 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 308.52710683818157,
            "unit": "iter/sec",
            "range": "stddev: 0.00008467608498454211",
            "extra": "mean: 3.241206292205913 msec\nrounds: 308"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 318.72002488416854,
            "unit": "iter/sec",
            "range": "stddev: 0.00007183541317471117",
            "extra": "mean: 3.137549955837971 msec\nrounds: 317"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 44.58732276380548,
            "unit": "iter/sec",
            "range": "stddev: 0.00017991828391455576",
            "extra": "mean: 22.42789963634612 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 23519.614348941774,
            "unit": "iter/sec",
            "range": "stddev: 0.000006331910355765241",
            "extra": "mean: 42.51770395397633 usec\nrounds: 10218"
          }
        ]
      }
    ]
  }
}