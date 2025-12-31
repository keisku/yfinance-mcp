window.BENCHMARK_DATA = {
  "lastUpdate": 1767202877367,
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
      },
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
          "id": "21fad2101be107be1a68b9ce46d5ea5318ab98cc",
          "message": "ci: Run Docker build only after tests pass\n\n- Add workflow_run trigger to wait for Test workflow on main\n- Skip Docker build if Test workflow fails\n- Remove coverage threshold (cache modules intentionally untested)\n- Fix coverage module path for accurate reporting",
          "timestamp": "2026-01-01T02:40:21+09:00",
          "tree_id": "cfa9b37757a2319d5f7ec9d134f57f811fdd3c10",
          "url": "https://github.com/keisku/yfinance-mcp/commit/21fad2101be107be1a68b9ce46d5ea5318ab98cc"
        },
        "date": 1767202877005,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 287.7221160213747,
            "unit": "iter/sec",
            "range": "stddev: 0.00009343601367948223",
            "extra": "mean: 3.4755757180852602 msec\nrounds: 188"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 290.0051231518045,
            "unit": "iter/sec",
            "range": "stddev: 0.00008923131252107331",
            "extra": "mean: 3.4482149457633726 msec\nrounds: 295"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 289.1043335265785,
            "unit": "iter/sec",
            "range": "stddev: 0.00006993614872864605",
            "extra": "mean: 3.458958874125856 msec\nrounds: 286"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 76.27806034814685,
            "unit": "iter/sec",
            "range": "stddev: 0.00017172110309441802",
            "extra": "mean: 13.109929584415484 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 41.043503856695736,
            "unit": "iter/sec",
            "range": "stddev: 0.0002929755000557673",
            "extra": "mean: 24.364391585365645 msec\nrounds: 41"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 307.06898120421346,
            "unit": "iter/sec",
            "range": "stddev: 0.00007016736774720425",
            "extra": "mean: 3.2565972508143344 msec\nrounds: 307"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 313.0024670786255,
            "unit": "iter/sec",
            "range": "stddev: 0.0001394701814838496",
            "extra": "mean: 3.1948629968747255 msec\nrounds: 320"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 43.83408858837152,
            "unit": "iter/sec",
            "range": "stddev: 0.00027899930633307695",
            "extra": "mean: 22.813295136362978 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 26893.278311393835,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022498245282502865",
            "extra": "mean: 37.18401261538767 usec\nrounds: 8878"
          }
        ]
      }
    ]
  }
}