window.BENCHMARK_DATA = {
  "lastUpdate": 1767450409678,
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
          "id": "c424f34410f49ea006b8bac37399dfbead7cdc11",
          "message": "docs: Simplify README and use ghcr.io Docker image\n\n- Update Cursor/Claude config to use ghcr.io/keisku/yfinance-mcp\n- Remove redundant workflow example section\n- Remove duplicate Docker config block",
          "timestamp": "2026-01-01T02:44:10+09:00",
          "tree_id": "fe505fa4916e658b2d7035d84bf688b26551851c",
          "url": "https://github.com/keisku/yfinance-mcp/commit/c424f34410f49ea006b8bac37399dfbead7cdc11"
        },
        "date": 1767203110639,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 290.18686460726946,
            "unit": "iter/sec",
            "range": "stddev: 0.00008793321065516738",
            "extra": "mean: 3.446055359374626 msec\nrounds: 192"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 290.77499019540113,
            "unit": "iter/sec",
            "range": "stddev: 0.0000858098669538027",
            "extra": "mean: 3.439085319297918 msec\nrounds: 285"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 283.36448798044836,
            "unit": "iter/sec",
            "range": "stddev: 0.00016376250063129358",
            "extra": "mean: 3.529023721804541 msec\nrounds: 266"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 74.45687517017555,
            "unit": "iter/sec",
            "range": "stddev: 0.00044868050087397757",
            "extra": "mean: 13.430593181817548 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 41.30914436346952,
            "unit": "iter/sec",
            "range": "stddev: 0.00023738050045330343",
            "extra": "mean: 24.20771515384665 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 305.81092616032174,
            "unit": "iter/sec",
            "range": "stddev: 0.00006663604008898465",
            "extra": "mean: 3.269994347670066 msec\nrounds: 279"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 316.7846304145749,
            "unit": "iter/sec",
            "range": "stddev: 0.00006499448329397463",
            "extra": "mean: 3.1567188051115473 msec\nrounds: 313"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 44.324177694086046,
            "unit": "iter/sec",
            "range": "stddev: 0.0004470790023663005",
            "extra": "mean: 22.56105024444537 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 23368.496154107506,
            "unit": "iter/sec",
            "range": "stddev: 0.000008219914629762605",
            "extra": "mean: 42.79265526567608 usec\nrounds: 8128"
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
          "id": "ca84235391e29ffe9a80ebd5b55e180ea07d391d",
          "message": "uv run pip-audit on the ci",
          "timestamp": "2026-01-01T02:45:17+09:00",
          "tree_id": "d1389eaf43a3f42899c8711c3ee857c26bc01636",
          "url": "https://github.com/keisku/yfinance-mcp/commit/ca84235391e29ffe9a80ebd5b55e180ea07d391d"
        },
        "date": 1767203177736,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 287.09763393372305,
            "unit": "iter/sec",
            "range": "stddev: 0.0002640953931463398",
            "extra": "mean: 3.4831356368156334 msec\nrounds: 201"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 286.7608533347649,
            "unit": "iter/sec",
            "range": "stddev: 0.00025424242235110847",
            "extra": "mean: 3.487226336408614 msec\nrounds: 217"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 291.9430402137644,
            "unit": "iter/sec",
            "range": "stddev: 0.00012632257269042782",
            "extra": "mean: 3.425325704862795 msec\nrounds: 288"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 75.69722784815573,
            "unit": "iter/sec",
            "range": "stddev: 0.0007800038624326143",
            "extra": "mean: 13.210523402599923 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 41.7947771120886,
            "unit": "iter/sec",
            "range": "stddev: 0.00020709973681287328",
            "extra": "mean: 23.92643457143268 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 311.2935745201058,
            "unit": "iter/sec",
            "range": "stddev: 0.000054537238616803274",
            "extra": "mean: 3.2124016743410557 msec\nrounds: 304"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 319.54479555727755,
            "unit": "iter/sec",
            "range": "stddev: 0.00004837368786834006",
            "extra": "mean: 3.1294516884746213 msec\nrounds: 321"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 44.55085891204346,
            "unit": "iter/sec",
            "range": "stddev: 0.0002125349651768853",
            "extra": "mean: 22.446256355557477 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 23956.754189569572,
            "unit": "iter/sec",
            "range": "stddev: 0.000004914624683517161",
            "extra": "mean: 41.741881729344854 usec\nrounds: 9901"
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
          "id": "fdd313e3acd5bd42787fb099bec610c0bf8f1576",
          "message": "increase alert-threshold to 130%",
          "timestamp": "2026-01-01T02:47:27+09:00",
          "tree_id": "d6a0e7a65b180bde7e026a9787a815b97e90badc",
          "url": "https://github.com/keisku/yfinance-mcp/commit/fdd313e3acd5bd42787fb099bec610c0bf8f1576"
        },
        "date": 1767203306655,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 298.1711152460731,
            "unit": "iter/sec",
            "range": "stddev: 0.00008962958490887084",
            "extra": "mean: 3.353778917098409 msec\nrounds: 193"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 293.30316917507787,
            "unit": "iter/sec",
            "range": "stddev: 0.00008591796967535943",
            "extra": "mean: 3.4094415099997852 msec\nrounds: 300"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 293.2438453023156,
            "unit": "iter/sec",
            "range": "stddev: 0.00006022857329231429",
            "extra": "mean: 3.410131247491534 msec\nrounds: 299"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 76.74801136889774,
            "unit": "iter/sec",
            "range": "stddev: 0.0004490369319236461",
            "extra": "mean: 13.029653565789351 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 41.88900431758762,
            "unit": "iter/sec",
            "range": "stddev: 0.0003932135545628504",
            "extra": "mean: 23.872613261904092 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 315.3883872837163,
            "unit": "iter/sec",
            "range": "stddev: 0.00007725453823525337",
            "extra": "mean: 3.170693786833763 msec\nrounds: 319"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 315.6977980418345,
            "unit": "iter/sec",
            "range": "stddev: 0.00010775524734228743",
            "extra": "mean: 3.1675862365929004 msec\nrounds: 317"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 44.59157280749541,
            "unit": "iter/sec",
            "range": "stddev: 0.0003004117678380286",
            "extra": "mean: 22.425762022727078 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 26712.269911463038,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024182974373331352",
            "extra": "mean: 37.435979919133345 usec\nrounds: 9163"
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
          "id": "79531860d01f4dceef01111444805aec8098e3d6",
          "message": "docs: Simplify README to essential information only",
          "timestamp": "2026-01-01T02:52:08+09:00",
          "tree_id": "eecaf6794364d300ebeba0eac1b7665beecac3c5",
          "url": "https://github.com/keisku/yfinance-mcp/commit/79531860d01f4dceef01111444805aec8098e3d6"
        },
        "date": 1767203590493,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 282.60632890663237,
            "unit": "iter/sec",
            "range": "stddev: 0.0001674927349225954",
            "extra": "mean: 3.5384911720444188 msec\nrounds: 186"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 285.8492737234546,
            "unit": "iter/sec",
            "range": "stddev: 0.0001336745688400108",
            "extra": "mean: 3.4983471777768163 msec\nrounds: 270"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 286.3510961356831,
            "unit": "iter/sec",
            "range": "stddev: 0.0001283393425244867",
            "extra": "mean: 3.4922164206634125 msec\nrounds: 271"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 75.75592091451746,
            "unit": "iter/sec",
            "range": "stddev: 0.0004924374640975079",
            "extra": "mean: 13.20028834615309 msec\nrounds: 78"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 40.52225448205888,
            "unit": "iter/sec",
            "range": "stddev: 0.0009137195150014588",
            "extra": "mean: 24.677797738098388 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 300.35439256478776,
            "unit": "iter/sec",
            "range": "stddev: 0.00017125881346863819",
            "extra": "mean: 3.3294002843134565 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 311.2107162318138,
            "unit": "iter/sec",
            "range": "stddev: 0.00011954890074153378",
            "extra": "mean: 3.2132569601334766 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 43.08649587775108,
            "unit": "iter/sec",
            "range": "stddev: 0.0008998326967221606",
            "extra": "mean: 23.209128048781015 msec\nrounds: 41"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 23405.374109907683,
            "unit": "iter/sec",
            "range": "stddev: 0.00000840856596271547",
            "extra": "mean: 42.72523033830473 usec\nrounds: 8570"
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
          "id": "52ef96c36fadda867fb8d0dfe1c29d373a10e387",
          "message": "docs: Add cache and API constraints to README\n\n- Improve project description\n- Document DuckDB cache location and env vars\n- List Yahoo Finance API limitations",
          "timestamp": "2026-01-01T03:04:21+09:00",
          "tree_id": "5b8659bc7f75ca0bbdbfa1dd6795cbacc5ba9ec9",
          "url": "https://github.com/keisku/yfinance-mcp/commit/52ef96c36fadda867fb8d0dfe1c29d373a10e387"
        },
        "date": 1767204322457,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 288.9868306046782,
            "unit": "iter/sec",
            "range": "stddev: 0.00008482978655827322",
            "extra": "mean: 3.460365297296049 msec\nrounds: 185"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 291.4266673760813,
            "unit": "iter/sec",
            "range": "stddev: 0.00008214908948007448",
            "extra": "mean: 3.431394968084772 msec\nrounds: 282"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 290.3907469153358,
            "unit": "iter/sec",
            "range": "stddev: 0.00007487411933713813",
            "extra": "mean: 3.4436358961931828 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 75.83497822181752,
            "unit": "iter/sec",
            "range": "stddev: 0.00023740924256871647",
            "extra": "mean: 13.186527159999931 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 40.05815815078191,
            "unit": "iter/sec",
            "range": "stddev: 0.0007634280405887693",
            "extra": "mean: 24.963703928571185 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 299.15764828616994,
            "unit": "iter/sec",
            "range": "stddev: 0.0001432075049650828",
            "extra": "mean: 3.342719150684773 msec\nrounds: 292"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 315.47326445920163,
            "unit": "iter/sec",
            "range": "stddev: 0.00006026653105643447",
            "extra": "mean: 3.169840720779444 msec\nrounds: 308"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 42.946497167935206,
            "unit": "iter/sec",
            "range": "stddev: 0.0018581205667823333",
            "extra": "mean: 23.284786093023246 msec\nrounds: 43"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 24382.969761095592,
            "unit": "iter/sec",
            "range": "stddev: 0.000004592893474412542",
            "extra": "mean: 41.01223147951225 usec\nrounds: 9530"
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
          "id": "17146b03a5479596e7c966fdf46d1ee311c60608",
          "message": "update readme",
          "timestamp": "2026-01-01T03:06:27+09:00",
          "tree_id": "86d62f6a731b75a34537106c2303a96415e8c5da",
          "url": "https://github.com/keisku/yfinance-mcp/commit/17146b03a5479596e7c966fdf46d1ee311c60608"
        },
        "date": 1767204802929,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 279.0748614596941,
            "unit": "iter/sec",
            "range": "stddev: 0.00027568283617753414",
            "extra": "mean: 3.5832679259232627 msec\nrounds: 189"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 272.77583126136926,
            "unit": "iter/sec",
            "range": "stddev: 0.00033963519764149976",
            "extra": "mean: 3.6660139403692873 msec\nrounds: 218"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 288.3310802076665,
            "unit": "iter/sec",
            "range": "stddev: 0.00007475286365091633",
            "extra": "mean: 3.4682351943459016 msec\nrounds: 283"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 75.6566559902663,
            "unit": "iter/sec",
            "range": "stddev: 0.0003829862689722327",
            "extra": "mean: 13.217607716215426 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 41.2776934121676,
            "unit": "iter/sec",
            "range": "stddev: 0.00033803254556546785",
            "extra": "mean: 24.226159878042647 msec\nrounds: 41"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 310.90646314017204,
            "unit": "iter/sec",
            "range": "stddev: 0.000052303908347782446",
            "extra": "mean: 3.2164014536717764 msec\nrounds: 313"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 304.8294711710658,
            "unit": "iter/sec",
            "range": "stddev: 0.0002154177671044451",
            "extra": "mean: 3.2805227006374813 msec\nrounds: 314"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 42.99797293325675,
            "unit": "iter/sec",
            "range": "stddev: 0.0004845487800397482",
            "extra": "mean: 23.256910309521846 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 22901.157921009344,
            "unit": "iter/sec",
            "range": "stddev: 0.000005035168442503438",
            "extra": "mean: 43.66591433713523 usec\nrounds: 7798"
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
          "id": "225fd1b0e218bba663094ddd1dc39c4d5e40a720",
          "message": "refactor(tools): Remove peers tool and improve quality_details readability\n\nConsolidate peer comparison into summary tool by updating its description\nto indicate it can be called multiple times for comparison. The peers tool\nwas redundant since summary provides richer data per symbol.\n\nQuality score details now use self-documenting names (e.g., \"ROA>0\",\n\"GrossMargin>20%\") instead of cryptic abbreviations (e.g., \"roa+\",\n\"margin+\"), making the output immediately understandable without a legend.",
          "timestamp": "2026-01-01T10:46:28+09:00",
          "tree_id": "c6d97881a1ab673857b6ceef5dbde8e6f7e0283f",
          "url": "https://github.com/keisku/yfinance-mcp/commit/225fd1b0e218bba663094ddd1dc39c4d5e40a720"
        },
        "date": 1767232060366,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 280.9045912597243,
            "unit": "iter/sec",
            "range": "stddev: 0.00012772398908691526",
            "extra": "mean: 3.559927573684263 msec\nrounds: 190"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 17.653155226460868,
            "unit": "iter/sec",
            "range": "stddev: 0.010422867326622842",
            "extra": "mean: 56.64709719999905 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 19.15994585946708,
            "unit": "iter/sec",
            "range": "stddev: 0.007643274890189808",
            "extra": "mean: 52.192214285714805 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 71.97775114628674,
            "unit": "iter/sec",
            "range": "stddev: 0.0006978834942310357",
            "extra": "mean: 13.89318204687462 msec\nrounds: 64"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 8.669465082144447,
            "unit": "iter/sec",
            "range": "stddev: 0.017315553337518358",
            "extra": "mean: 115.34737040000209 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 289.4429158454244,
            "unit": "iter/sec",
            "range": "stddev: 0.00008755722394255871",
            "extra": "mean: 3.4549126796872276 msec\nrounds: 256"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 265.3183307020803,
            "unit": "iter/sec",
            "range": "stddev: 0.00024437613095379076",
            "extra": "mean: 3.769057333331697 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 41.10823925672464,
            "unit": "iter/sec",
            "range": "stddev: 0.00043575908843804576",
            "extra": "mean: 24.32602364102511 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 26391.198549665245,
            "unit": "iter/sec",
            "range": "stddev: 0.000004678723524736028",
            "extra": "mean: 37.891420433903875 usec\nrounds: 8251"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "82f34599a3bd7e637bb8889fdc0dd056b6652420",
          "message": "refactor(benchmarks): Replace flaky benchmarks with deterministic fixture-based approach",
          "timestamp": "2026-01-01T01:46:47Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/1/commits/82f34599a3bd7e637bb8889fdc0dd056b6652420"
        },
        "date": 1767237468183,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_stock",
            "value": 277.9468930208445,
            "unit": "iter/sec",
            "range": "stddev: 0.0003348151504575182",
            "extra": "mean: 3.5978095999979587 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_msft_stock",
            "value": 280.74140886399124,
            "unit": "iter/sec",
            "range": "stddev: 0.00014616783926065988",
            "extra": "mean: 3.5619967999963364 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_googl_stock",
            "value": 260.4514477872976,
            "unit": "iter/sec",
            "range": "stddev: 0.00022537889102096253",
            "extra": "mean: 3.839487199996938 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 76.80724797763224,
            "unit": "iter/sec",
            "range": "stddev: 0.00037746759734211786",
            "extra": "mean: 13.01960461194 msec\nrounds: 67"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_short_period",
            "value": 327.4991869497971,
            "unit": "iter/sec",
            "range": "stddev: 0.00007845597620491594",
            "extra": "mean: 3.053442694968558 msec\nrounds: 318"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_long_period",
            "value": 32.661538400137644,
            "unit": "iter/sec",
            "range": "stddev: 0.0561514539199976",
            "extra": "mean: 30.617051399997308 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 301.90873950916375,
            "unit": "iter/sec",
            "range": "stddev: 0.00017612622469406143",
            "extra": "mean: 3.312259200001222 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 50.90562081175882,
            "unit": "iter/sec",
            "range": "stddev: 0.04316876270774682",
            "extra": "mean: 19.644196142855158 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly_long_period",
            "value": 282.17979830829205,
            "unit": "iter/sec",
            "range": "stddev: 0.00008944997686654644",
            "extra": "mean: 3.5438397999968174 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly_long_period",
            "value": 306.99649619801664,
            "unit": "iter/sec",
            "range": "stddev: 0.00006069727638097274",
            "extra": "mean: 3.2573661666646103 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 93.58537357940124,
            "unit": "iter/sec",
            "range": "stddev: 0.0005378561436107866",
            "extra": "mean: 10.685430444444009 msec\nrounds: 99"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 99.64911950640503,
            "unit": "iter/sec",
            "range": "stddev: 0.00018127356503042249",
            "extra": "mean: 10.035211599995364 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_short_period",
            "value": 101.65650761249313,
            "unit": "iter/sec",
            "range": "stddev: 0.0005084843076992053",
            "extra": "mean: 9.837048542056195 msec\nrounds: 107"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_long_period",
            "value": 22.510519143084572,
            "unit": "iter/sec",
            "range": "stddev: 0.05718318088067574",
            "extra": "mean: 44.423675600000934 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1month",
            "value": 296.17438018116235,
            "unit": "iter/sec",
            "range": "stddev: 0.00017692238892436047",
            "extra": "mean: 3.376389272388535 msec\nrounds: 268"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1year",
            "value": 280.8842677293515,
            "unit": "iter/sec",
            "range": "stddev: 0.0001745760050448804",
            "extra": "mean: 3.56018515413458 msec\nrounds: 266"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_multi_year",
            "value": 234.66230803699509,
            "unit": "iter/sec",
            "range": "stddev: 0.00027142298366082283",
            "extra": "mean: 4.261442787149044 msec\nrounds: 249"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_cache_stats_retrieval",
            "value": 11251341.978935556,
            "unit": "iter/sec",
            "range": "stddev: 9.351444851928943e-9",
            "extra": "mean: 88.87828686321788 nsec\nrounds: 109927"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_repeated_same_query",
            "value": 29.800172507856853,
            "unit": "iter/sec",
            "range": "stddev: 0.001518502019553872",
            "extra": "mean: 33.556852724136036 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_interleaved_symbols",
            "value": 33.909508162240506,
            "unit": "iter/sec",
            "range": "stddev: 0.0008051380410166088",
            "extra": "mean: 29.490253742858386 msec\nrounds: 35"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_mixed_intervals",
            "value": 97.63819772662794,
            "unit": "iter/sec",
            "range": "stddev: 0.0004664315078474222",
            "extra": "mean: 10.24189326804093 msec\nrounds: 97"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_typical_analysis_workflow",
            "value": 72.20023844753558,
            "unit": "iter/sec",
            "range": "stddev: 0.00042294079973854054",
            "extra": "mean: 13.850369770269547 msec\nrounds: 74"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_multi_stock_comparison",
            "value": 92.28136243022455,
            "unit": "iter/sec",
            "range": "stddev: 0.0003660301145793303",
            "extra": "mean: 10.836424318682077 msec\nrounds: 91"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_dashboard_load",
            "value": 63.02780300862195,
            "unit": "iter/sec",
            "range": "stddev: 0.0004727894458155499",
            "extra": "mean: 15.866013921875144 msec\nrounds: 64"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "54be90c69ad995b1461bff4dfbab17509028b062",
          "message": "refactor(benchmarks): Replace flaky benchmarks with deterministic fixture-based approach",
          "timestamp": "2026-01-01T01:46:47Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/1/commits/54be90c69ad995b1461bff4dfbab17509028b062"
        },
        "date": 1767237708401,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_stock",
            "value": 276.75997758477274,
            "unit": "iter/sec",
            "range": "stddev: 0.0003377279040444935",
            "extra": "mean: 3.6132391999984748 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_japan_stock",
            "value": 19.373981027890014,
            "unit": "iter/sec",
            "range": "stddev: 0.03810020412845931",
            "extra": "mean: 51.61561779999886 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_stock",
            "value": 19.799419760580196,
            "unit": "iter/sec",
            "range": "stddev: 0.037010246102939334",
            "extra": "mean: 50.50653060000059 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 75.1574368676088,
            "unit": "iter/sec",
            "range": "stddev: 0.0033177044393559152",
            "extra": "mean: 13.305403186666922 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_short_period",
            "value": 322.5765053210832,
            "unit": "iter/sec",
            "range": "stddev: 0.00006163440647372283",
            "extra": "mean: 3.100039784374964 msec\nrounds: 320"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_long_period",
            "value": 37.033454393229185,
            "unit": "iter/sec",
            "range": "stddev: 0.047993190305644334",
            "extra": "mean: 27.002611999998294 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 292.1909741508349,
            "unit": "iter/sec",
            "range": "stddev: 0.00017355965382175243",
            "extra": "mean: 3.4224191999982168 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 45.36244819681362,
            "unit": "iter/sec",
            "range": "stddev: 0.04975165543601346",
            "extra": "mean: 22.044665571428368 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly_long_period",
            "value": 276.42056814458704,
            "unit": "iter/sec",
            "range": "stddev: 0.000052101758487454215",
            "extra": "mean: 3.6176758000038944 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly_long_period",
            "value": 298.280250302898,
            "unit": "iter/sec",
            "range": "stddev: 0.00005614843208634909",
            "extra": "mean: 3.3525518333329765 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 9.917177977355117,
            "unit": "iter/sec",
            "range": "stddev: 0.03909820305737116",
            "extra": "mean: 100.8351370000014 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 14.518053962299872,
            "unit": "iter/sec",
            "range": "stddev: 0.10185436381424293",
            "extra": "mean: 68.87975500000039 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_short_period",
            "value": 12.261810048180676,
            "unit": "iter/sec",
            "range": "stddev: 0.0011085742193724401",
            "extra": "mean: 81.55402799999933 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_long_period",
            "value": 3.8552614208356903,
            "unit": "iter/sec",
            "range": "stddev: 0.35652368912282295",
            "extra": "mean: 259.3857823999997 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1month",
            "value": 304.30934157067674,
            "unit": "iter/sec",
            "range": "stddev: 0.00006459749995379969",
            "extra": "mean: 3.286129813953631 msec\nrounds: 301"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1year",
            "value": 278.1873665508056,
            "unit": "iter/sec",
            "range": "stddev: 0.00006979420589322261",
            "extra": "mean: 3.594699545126069 msec\nrounds: 277"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_multi_year",
            "value": 242.42449700902014,
            "unit": "iter/sec",
            "range": "stddev: 0.000056382201382419987",
            "extra": "mean: 4.124995668085441 msec\nrounds: 235"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_cache_stats_retrieval",
            "value": 10976879.731045686,
            "unit": "iter/sec",
            "range": "stddev: 7.857699019356337e-9",
            "extra": "mean: 91.10056997087435 nsec\nrounds: 107216"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_repeated_same_query",
            "value": 30.586647436398348,
            "unit": "iter/sec",
            "range": "stddev: 0.0002817978993933981",
            "extra": "mean: 32.694004862069065 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_interleaved_symbols",
            "value": 3.7332482208489135,
            "unit": "iter/sec",
            "range": "stddev: 0.04017775864492263",
            "extra": "mean: 267.86324959999774 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_mixed_intervals",
            "value": 96.02160591756616,
            "unit": "iter/sec",
            "range": "stddev: 0.00015078316372059642",
            "extra": "mean: 10.414322802083653 msec\nrounds: 96"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_typical_analysis_workflow",
            "value": 71.82647986610523,
            "unit": "iter/sec",
            "range": "stddev: 0.00022517152287651542",
            "extra": "mean: 13.92244199999975 msec\nrounds: 72"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_multi_stock_comparison",
            "value": 13.93202906859915,
            "unit": "iter/sec",
            "range": "stddev: 0.0014927527295076617",
            "extra": "mean: 71.77705379999963 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_dashboard_load",
            "value": 12.726101278081202,
            "unit": "iter/sec",
            "range": "stddev: 0.003192173019778743",
            "extra": "mean: 78.57866114285527 msec\nrounds: 14"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "d9f9363cfe662909cada919b57924898188e0a39",
          "message": "refactor(benchmarks): Replace flaky benchmarks with deterministic fixture-based approach",
          "timestamp": "2026-01-01T01:46:47Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/1/commits/d9f9363cfe662909cada919b57924898188e0a39"
        },
        "date": 1767237972904,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 30.487484015867718,
            "unit": "iter/sec",
            "range": "stddev: 0.06913474598951239",
            "extra": "mean: 32.80034520000186 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 510.6404186607918,
            "unit": "iter/sec",
            "range": "stddev: 0.00028752270921380375",
            "extra": "mean: 1.958325199996125 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 568.8753864563389,
            "unit": "iter/sec",
            "range": "stddev: 0.00020983828180665508",
            "extra": "mean: 1.7578542222212137 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 143.15137835576587,
            "unit": "iter/sec",
            "range": "stddev: 0.0006329554315874542",
            "extra": "mean: 6.985612094595119 msec\nrounds: 148"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_short_period",
            "value": 602.699034072323,
            "unit": "iter/sec",
            "range": "stddev: 0.00004325950251983556",
            "extra": "mean: 1.6592029246225761 msec\nrounds: 597"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_long_period",
            "value": 186.4106956905135,
            "unit": "iter/sec",
            "range": "stddev: 0.0030270522832802787",
            "extra": "mean: 5.364499050313293 msec\nrounds: 159"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 260.544603207064,
            "unit": "iter/sec",
            "range": "stddev: 0.000050744751784190904",
            "extra": "mean: 3.8381144252881136 msec\nrounds: 261"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 604.3287209349709,
            "unit": "iter/sec",
            "range": "stddev: 0.000030484508743514612",
            "extra": "mean: 1.6547285696646639 msec\nrounds: 567"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly_long_period",
            "value": 213.08539023295228,
            "unit": "iter/sec",
            "range": "stddev: 0.00006004798901312608",
            "extra": "mean: 4.692954307692168 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly_long_period",
            "value": 218.13679895544416,
            "unit": "iter/sec",
            "range": "stddev: 0.00007376671587521617",
            "extra": "mean: 4.584279244898319 msec\nrounds: 196"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 72.18666860362262,
            "unit": "iter/sec",
            "range": "stddev: 0.0004275438635452258",
            "extra": "mean: 13.852973399991697 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 37.769422510075266,
            "unit": "iter/sec",
            "range": "stddev: 0.00026291012231342066",
            "extra": "mean: 26.476443999990806 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_short_period",
            "value": 75.47576867805736,
            "unit": "iter/sec",
            "range": "stddev: 0.00015990001235160757",
            "extra": "mean: 13.249285400000492 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_long_period",
            "value": 21.778182882950862,
            "unit": "iter/sec",
            "range": "stddev: 0.020579485540606016",
            "extra": "mean: 45.91751320000412 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1month",
            "value": 321.010306065391,
            "unit": "iter/sec",
            "range": "stddev: 0.00008031612604038365",
            "extra": "mean: 3.1151647816450367 msec\nrounds: 316"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1year",
            "value": 292.3064347598239,
            "unit": "iter/sec",
            "range": "stddev: 0.00006052095958124885",
            "extra": "mean: 3.421067349480892 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_multi_year",
            "value": 226.92639852592393,
            "unit": "iter/sec",
            "range": "stddev: 0.00006262167950905277",
            "extra": "mean: 4.40671515740713 msec\nrounds: 216"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_cache_stats_retrieval",
            "value": 11069840.505582558,
            "unit": "iter/sec",
            "range": "stddev: 8.152162864428423e-9",
            "extra": "mean: 90.33553821265055 nsec\nrounds: 108015"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_repeated_same_query",
            "value": 59.408849413889854,
            "unit": "iter/sec",
            "range": "stddev: 0.00012683329668314815",
            "extra": "mean: 16.832509127271514 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_interleaved_symbols",
            "value": 24.99070820478334,
            "unit": "iter/sec",
            "range": "stddev: 0.0007398903196923331",
            "extra": "mean: 40.01487239999847 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_mixed_intervals",
            "value": 138.6565321017882,
            "unit": "iter/sec",
            "range": "stddev: 0.00008704786406839425",
            "extra": "mean: 7.212065561151471 msec\nrounds: 139"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_typical_analysis_workflow",
            "value": 102.00615253581563,
            "unit": "iter/sec",
            "range": "stddev: 0.0001248301561433379",
            "extra": "mean: 9.80333024176054 msec\nrounds: 91"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_multi_stock_comparison",
            "value": 199.3525522450398,
            "unit": "iter/sec",
            "range": "stddev: 0.00005572717254659025",
            "extra": "mean: 5.016238762626033 msec\nrounds: 198"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_dashboard_load",
            "value": 118.73184235304846,
            "unit": "iter/sec",
            "range": "stddev: 0.00015248048502137772",
            "extra": "mean: 8.422340462186257 msec\nrounds: 119"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "2450864e3cb2c8d561ccce09db8028ebccfef729",
          "message": "refactor(benchmarks): Replace flaky benchmarks with deterministic fixture-based approach",
          "timestamp": "2026-01-01T01:46:47Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/1/commits/2450864e3cb2c8d561ccce09db8028ebccfef729"
        },
        "date": 1767238081613,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 25.64022632340668,
            "unit": "iter/sec",
            "range": "stddev: 0.08299904584331252",
            "extra": "mean: 39.001215799999045 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 520.6085497461161,
            "unit": "iter/sec",
            "range": "stddev: 0.00032072655112367303",
            "extra": "mean: 1.9208289999994577 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 555.0962794663227,
            "unit": "iter/sec",
            "range": "stddev: 0.00028383404921859974",
            "extra": "mean: 1.8014892857170903 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 150.02901269100198,
            "unit": "iter/sec",
            "range": "stddev: 0.0002874437550358187",
            "extra": "mean: 6.665377463088346 msec\nrounds: 149"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_short_period",
            "value": 602.3134060378226,
            "unit": "iter/sec",
            "range": "stddev: 0.00011881185542616098",
            "extra": "mean: 1.6602652206901143 msec\nrounds: 580"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_long_period",
            "value": 185.57019020138264,
            "unit": "iter/sec",
            "range": "stddev: 0.003132342846415783",
            "extra": "mean: 5.388796546011997 msec\nrounds: 163"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 261.95161314755,
            "unit": "iter/sec",
            "range": "stddev: 0.000047946448717783726",
            "extra": "mean: 3.8174989189195334 msec\nrounds: 259"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 601.4698583701581,
            "unit": "iter/sec",
            "range": "stddev: 0.00007370653697289862",
            "extra": "mean: 1.6625937045453363 msec\nrounds: 572"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly_long_period",
            "value": 210.29204544522494,
            "unit": "iter/sec",
            "range": "stddev: 0.000324589089480753",
            "extra": "mean: 4.755291613065181 msec\nrounds: 199"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly_long_period",
            "value": 219.37245630280916,
            "unit": "iter/sec",
            "range": "stddev: 0.0000805422987462742",
            "extra": "mean: 4.5584574146339385 msec\nrounds: 205"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 70.72020952190418,
            "unit": "iter/sec",
            "range": "stddev: 0.0007447008506715763",
            "extra": "mean: 14.140229600002385 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 34.8968984048483,
            "unit": "iter/sec",
            "range": "stddev: 0.0005729575458760487",
            "extra": "mean: 28.65584179999985 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_short_period",
            "value": 72.73375120750572,
            "unit": "iter/sec",
            "range": "stddev: 0.00026331991968167075",
            "extra": "mean: 13.748775271428672 msec\nrounds: 70"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_long_period",
            "value": 21.816113714067363,
            "unit": "iter/sec",
            "range": "stddev: 0.01986132970677499",
            "extra": "mean: 45.83767819999878 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1month",
            "value": 318.8408360936426,
            "unit": "iter/sec",
            "range": "stddev: 0.00006747948353617988",
            "extra": "mean: 3.136361114378407 msec\nrounds: 306"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1year",
            "value": 291.4928698813471,
            "unit": "iter/sec",
            "range": "stddev: 0.00008722669851248952",
            "extra": "mean: 3.4306156456144286 msec\nrounds: 285"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_multi_year",
            "value": 223.83561149209777,
            "unit": "iter/sec",
            "range": "stddev: 0.00011159242123673034",
            "extra": "mean: 4.467564358208943 msec\nrounds: 201"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_cache_stats_retrieval",
            "value": 11030724.138318751,
            "unit": "iter/sec",
            "range": "stddev: 9.569880192988914e-9",
            "extra": "mean: 90.65587965582242 nsec\nrounds: 106644"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_repeated_same_query",
            "value": 59.073522493653925,
            "unit": "iter/sec",
            "range": "stddev: 0.0001574934754970025",
            "extra": "mean: 16.928057745454858 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_interleaved_symbols",
            "value": 24.5129301612931,
            "unit": "iter/sec",
            "range": "stddev: 0.0003265317314595551",
            "extra": "mean: 40.794796600001746 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_mixed_intervals",
            "value": 135.1957940381678,
            "unit": "iter/sec",
            "range": "stddev: 0.0002163028485388269",
            "extra": "mean: 7.396679808823675 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_typical_analysis_workflow",
            "value": 100.72976316041203,
            "unit": "iter/sec",
            "range": "stddev: 0.0002880193972091838",
            "extra": "mean: 9.92755238000015 msec\nrounds: 100"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_multi_stock_comparison",
            "value": 196.71973025278152,
            "unit": "iter/sec",
            "range": "stddev: 0.00006935782571734227",
            "extra": "mean: 5.083374192893703 msec\nrounds: 197"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_dashboard_load",
            "value": 118.30812736568781,
            "unit": "iter/sec",
            "range": "stddev: 0.00007291679128059749",
            "extra": "mean: 8.452504677967069 msec\nrounds: 118"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "b246d9fffc88f29718ab69600aa79842d96f0e65",
          "message": "refactor(benchmarks): Replace flaky benchmarks with deterministic fixture-based approach",
          "timestamp": "2026-01-01T01:46:47Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/1/commits/b246d9fffc88f29718ab69600aa79842d96f0e65"
        },
        "date": 1767238341647,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 32.05648177701824,
            "unit": "iter/sec",
            "range": "stddev: 0.06545478141135999",
            "extra": "mean: 31.19493920000025 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 505.5912323189053,
            "unit": "iter/sec",
            "range": "stddev: 0.0003532480752615512",
            "extra": "mean: 1.977882400004205 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 586.3442381410554,
            "unit": "iter/sec",
            "range": "stddev: 0.00004949950752451735",
            "extra": "mean: 1.705482777779821 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 149.50975988946936,
            "unit": "iter/sec",
            "range": "stddev: 0.00010635630869025593",
            "extra": "mean: 6.688526560000412 msec\nrounds: 150"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_short_period",
            "value": 597.3706043001744,
            "unit": "iter/sec",
            "range": "stddev: 0.00005437010142646775",
            "extra": "mean: 1.6740026924684555 msec\nrounds: 478"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_long_period",
            "value": 194.88858142580278,
            "unit": "iter/sec",
            "range": "stddev: 0.00006850412234790802",
            "extra": "mean: 5.131136943396122 msec\nrounds: 159"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 259.83597349670987,
            "unit": "iter/sec",
            "range": "stddev: 0.000141424763555479",
            "extra": "mean: 3.8485818054468215 msec\nrounds: 257"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 602.1626943525838,
            "unit": "iter/sec",
            "range": "stddev: 0.0000629209872346915",
            "extra": "mean: 1.6606807585035663 msec\nrounds: 588"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly_long_period",
            "value": 213.41553448896042,
            "unit": "iter/sec",
            "range": "stddev: 0.00004890449477893475",
            "extra": "mean: 4.685694517948637 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly_long_period",
            "value": 218.9460284267615,
            "unit": "iter/sec",
            "range": "stddev: 0.0001621008461241064",
            "extra": "mean: 4.5673356451610845 msec\nrounds: 217"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 72.97413760076783,
            "unit": "iter/sec",
            "range": "stddev: 0.0003226354635463016",
            "extra": "mean: 13.703484999999205 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 37.941092038267286,
            "unit": "iter/sec",
            "range": "stddev: 0.00009421012520019145",
            "extra": "mean: 26.356647800000133 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_short_period",
            "value": 75.73123778886448,
            "unit": "iter/sec",
            "range": "stddev: 0.00019425267859695934",
            "extra": "mean: 13.204590723684698 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_long_period",
            "value": 22.675851111994078,
            "unit": "iter/sec",
            "range": "stddev: 0.017865267420981844",
            "extra": "mean: 44.099778000000356 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1month",
            "value": 323.57687183882757,
            "unit": "iter/sec",
            "range": "stddev: 0.00004961444174712664",
            "extra": "mean: 3.0904557372014407 msec\nrounds: 293"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1year",
            "value": 241.08032075903748,
            "unit": "iter/sec",
            "range": "stddev: 0.000196554172204601",
            "extra": "mean: 4.147995144736477 msec\nrounds: 228"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_multi_year",
            "value": 192.16819143676523,
            "unit": "iter/sec",
            "range": "stddev: 0.00039011437814996674",
            "extra": "mean: 5.203774841837232 msec\nrounds: 196"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_cache_stats_retrieval",
            "value": 11090984.58784878,
            "unit": "iter/sec",
            "range": "stddev: 7.89740577536502e-9",
            "extra": "mean: 90.1633206753884 nsec\nrounds: 108496"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_repeated_same_query",
            "value": 59.40107128881345,
            "unit": "iter/sec",
            "range": "stddev: 0.00011039197106663912",
            "extra": "mean: 16.834713218182692 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_interleaved_symbols",
            "value": 24.598369797356078,
            "unit": "iter/sec",
            "range": "stddev: 0.001468079970580711",
            "extra": "mean: 40.65310051999802 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_mixed_intervals",
            "value": 137.56485539965496,
            "unit": "iter/sec",
            "range": "stddev: 0.00007357592415837812",
            "extra": "mean: 7.269298521739356 msec\nrounds: 138"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_typical_analysis_workflow",
            "value": 101.33857723280293,
            "unit": "iter/sec",
            "range": "stddev: 0.00012700482284236193",
            "extra": "mean: 9.8679103980582 msec\nrounds: 103"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_multi_stock_comparison",
            "value": 197.28476871641445,
            "unit": "iter/sec",
            "range": "stddev: 0.00005990308599242159",
            "extra": "mean: 5.068815025641654 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_dashboard_load",
            "value": 118.57563529471652,
            "unit": "iter/sec",
            "range": "stddev: 0.0000726335362260382",
            "extra": "mean: 8.433435735043942 msec\nrounds: 117"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "28a165993ba3fc1593b79a6c3ef433325eb0b985",
          "message": "refactor(benchmarks): Replace flaky benchmarks with deterministic fixture-based approach",
          "timestamp": "2026-01-01T01:46:47Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/1/commits/28a165993ba3fc1593b79a6c3ef433325eb0b985"
        },
        "date": 1767238462703,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 17.0934023743494,
            "unit": "iter/sec",
            "range": "stddev: 0.12650730073537317",
            "extra": "mean: 58.50210380003773 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 536.4433889097286,
            "unit": "iter/sec",
            "range": "stddev: 0.00034672665551104394",
            "extra": "mean: 1.864129600016895 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 562.7774915142563,
            "unit": "iter/sec",
            "range": "stddev: 0.00022899016998005292",
            "extra": "mean: 1.7769011999916984 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 149.8677867901071,
            "unit": "iter/sec",
            "range": "stddev: 0.00010801422029196063",
            "extra": "mean: 6.6725479932556855 msec\nrounds: 148"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_short_period",
            "value": 590.8306417295537,
            "unit": "iter/sec",
            "range": "stddev: 0.000050205896514053266",
            "extra": "mean: 1.6925323931620648 msec\nrounds: 585"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_long_period",
            "value": 192.67694761001255,
            "unit": "iter/sec",
            "range": "stddev: 0.00006313596274032661",
            "extra": "mean: 5.1900344717108995 msec\nrounds: 159"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 259.58820629868006,
            "unit": "iter/sec",
            "range": "stddev: 0.00004916232725357496",
            "extra": "mean: 3.8522551323052334 msec\nrounds: 257"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 607.2616837952455,
            "unit": "iter/sec",
            "range": "stddev: 0.00003537254807242421",
            "extra": "mean: 1.6467365333347406 msec\nrounds: 570"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly_long_period",
            "value": 212.36499870387706,
            "unit": "iter/sec",
            "range": "stddev: 0.00006134113158814242",
            "extra": "mean: 4.70887390155289 msec\nrounds: 193"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly_long_period",
            "value": 218.21843276827676,
            "unit": "iter/sec",
            "range": "stddev: 0.00008877790403931774",
            "extra": "mean: 4.582564301806194 msec\nrounds: 222"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 68.80125779742907,
            "unit": "iter/sec",
            "range": "stddev: 0.0005693277505500463",
            "extra": "mean: 14.534617999925104 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 37.977632435452186,
            "unit": "iter/sec",
            "range": "stddev: 0.00018742546000322314",
            "extra": "mean: 26.331288599931213 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_short_period",
            "value": 75.36652224487035,
            "unit": "iter/sec",
            "range": "stddev: 0.0001669199301874588",
            "extra": "mean: 13.268490706668672 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_long_period",
            "value": 19.3480993796798,
            "unit": "iter/sec",
            "range": "stddev: 0.03246872766004813",
            "extra": "mean: 51.68466320005791 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1month",
            "value": 317.67664781384,
            "unit": "iter/sec",
            "range": "stddev: 0.00006147895324463217",
            "extra": "mean: 3.147854923809208 msec\nrounds: 315"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1year",
            "value": 240.74551638292914,
            "unit": "iter/sec",
            "range": "stddev: 0.00007886312411070075",
            "extra": "mean: 4.153763754459305 msec\nrounds: 224"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_multi_year",
            "value": 195.9995199358983,
            "unit": "iter/sec",
            "range": "stddev: 0.00007037768209846923",
            "extra": "mean: 5.102053312819594 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_cache_stats_retrieval",
            "value": 11149842.054625317,
            "unit": "iter/sec",
            "range": "stddev: 8.123851943378133e-9",
            "extra": "mean: 89.68736912153545 nsec\nrounds: 108249"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_repeated_same_query",
            "value": 59.40079869844525,
            "unit": "iter/sec",
            "range": "stddev: 0.00012003475665471616",
            "extra": "mean: 16.834790472710832 msec\nrounds: 55"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_interleaved_symbols",
            "value": 24.79583247913126,
            "unit": "iter/sec",
            "range": "stddev: 0.000217734258051023",
            "extra": "mean: 40.32935780001026 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_mixed_intervals",
            "value": 137.571580476509,
            "unit": "iter/sec",
            "range": "stddev: 0.00006407699264166998",
            "extra": "mean: 7.268943167886007 msec\nrounds: 137"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_typical_analysis_workflow",
            "value": 102.37002863746366,
            "unit": "iter/sec",
            "range": "stddev: 0.0002548876664540501",
            "extra": "mean: 9.76848412870363 msec\nrounds: 101"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_multi_stock_comparison",
            "value": 196.7489366240567,
            "unit": "iter/sec",
            "range": "stddev: 0.0002551072769971104",
            "extra": "mean: 5.0826195920477915 msec\nrounds: 201"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_dashboard_load",
            "value": 118.4691711732495,
            "unit": "iter/sec",
            "range": "stddev: 0.0001833273053094035",
            "extra": "mean: 8.441014570259787 msec\nrounds: 121"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "bbcaef5d4ac8d6f10105b64659575ed9c2303bca",
          "message": "refactor(benchmarks): Replace flaky benchmarks with deterministic fixture-based approach",
          "timestamp": "2026-01-01T01:46:47Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/1/commits/bbcaef5d4ac8d6f10105b64659575ed9c2303bca"
        },
        "date": 1767242601572,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 34.13870428555135,
            "unit": "iter/sec",
            "range": "stddev: 0.061415368764438524",
            "extra": "mean: 29.292265800000905 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 499.45399689130124,
            "unit": "iter/sec",
            "range": "stddev: 0.0003537098599782424",
            "extra": "mean: 2.0021863999971856 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 586.6328547136011,
            "unit": "iter/sec",
            "range": "stddev: 0.00008233343083906883",
            "extra": "mean: 1.7046436999990533 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 149.23590430482585,
            "unit": "iter/sec",
            "range": "stddev: 0.0000725830284413419",
            "extra": "mean: 6.700800351351259 msec\nrounds: 148"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_short_period",
            "value": 595.063393623406,
            "unit": "iter/sec",
            "range": "stddev: 0.00004106925654888745",
            "extra": "mean: 1.6804932225974964 msec\nrounds: 593"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_long_period",
            "value": 192.88351042784237,
            "unit": "iter/sec",
            "range": "stddev: 0.00012002014577214761",
            "extra": "mean: 5.184476359756525 msec\nrounds: 164"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 258.2741391964283,
            "unit": "iter/sec",
            "range": "stddev: 0.00006511370099353378",
            "extra": "mean: 3.8718549333329038 msec\nrounds: 255"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 598.6096413187566,
            "unit": "iter/sec",
            "range": "stddev: 0.00002967482254165633",
            "extra": "mean: 1.6705377444255112 msec\nrounds: 583"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly_long_period",
            "value": 211.8448517300431,
            "unit": "iter/sec",
            "range": "stddev: 0.00005991225692593799",
            "extra": "mean: 4.720435695432024 msec\nrounds: 197"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly_long_period",
            "value": 217.1979994773416,
            "unit": "iter/sec",
            "range": "stddev: 0.00006527541847421802",
            "extra": "mean: 4.604093971428689 msec\nrounds: 210"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 69.63872197065572,
            "unit": "iter/sec",
            "range": "stddev: 0.0009368586687956983",
            "extra": "mean: 14.359827000004088 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 37.43578078983932,
            "unit": "iter/sec",
            "range": "stddev: 0.00036732466882755436",
            "extra": "mean: 26.712412000003383 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_short_period",
            "value": 74.45041588951469,
            "unit": "iter/sec",
            "range": "stddev: 0.00013387907183905188",
            "extra": "mean: 13.431758413331256 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_long_period",
            "value": 21.75590518358707,
            "unit": "iter/sec",
            "range": "stddev: 0.01978082336118334",
            "extra": "mean: 45.96453200000212 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1month",
            "value": 317.4965559156427,
            "unit": "iter/sec",
            "range": "stddev: 0.00008057593029958946",
            "extra": "mean: 3.149640464968367 msec\nrounds: 314"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1year",
            "value": 239.6502614036639,
            "unit": "iter/sec",
            "range": "stddev: 0.00006465385591780562",
            "extra": "mean: 4.172747378379081 msec\nrounds: 222"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_multi_year",
            "value": 194.31253986087947,
            "unit": "iter/sec",
            "range": "stddev: 0.00007642023638640899",
            "extra": "mean: 5.146348252747675 msec\nrounds: 182"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_cache_stats_retrieval",
            "value": 10483204.608082807,
            "unit": "iter/sec",
            "range": "stddev: 8.197361863891909e-9",
            "extra": "mean: 95.39067845999834 nsec\nrounds: 102998"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_repeated_same_query",
            "value": 58.85204293263416,
            "unit": "iter/sec",
            "range": "stddev: 0.00010275590851600684",
            "extra": "mean: 16.99176358490502 msec\nrounds: 53"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_interleaved_symbols",
            "value": 24.55838811785812,
            "unit": "iter/sec",
            "range": "stddev: 0.00021698532401688146",
            "extra": "mean: 40.71928479999997 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_mixed_intervals",
            "value": 136.0024707600081,
            "unit": "iter/sec",
            "range": "stddev: 0.00017384369842636223",
            "extra": "mean: 7.352807595419456 msec\nrounds: 131"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_typical_analysis_workflow",
            "value": 100.86265458191721,
            "unit": "iter/sec",
            "range": "stddev: 0.00008757071581798036",
            "extra": "mean: 9.914472349999812 msec\nrounds: 100"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_multi_stock_comparison",
            "value": 196.03745870283535,
            "unit": "iter/sec",
            "range": "stddev: 0.00005775172889082981",
            "extra": "mean: 5.101065921874944 msec\nrounds: 192"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_dashboard_load",
            "value": 117.6537044536866,
            "unit": "iter/sec",
            "range": "stddev: 0.00007956610748889068",
            "extra": "mean: 8.499519880342074 msec\nrounds: 117"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "3bccfe8a7528064b1192e41f78ffa0129833d96a",
          "message": "refactor(benchmarks): Replace flaky benchmarks with deterministic fixture-based approach",
          "timestamp": "2026-01-01T01:46:47Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/1/commits/3bccfe8a7528064b1192e41f78ffa0129833d96a"
        },
        "date": 1767242696450,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 32.18471513505092,
            "unit": "iter/sec",
            "range": "stddev: 0.06563105849556347",
            "extra": "mean: 31.07064939999873 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 552.9137948796737,
            "unit": "iter/sec",
            "range": "stddev: 0.0002669886653187435",
            "extra": "mean: 1.8086001999961354 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 585.4011604882062,
            "unit": "iter/sec",
            "range": "stddev: 0.00007821471759061492",
            "extra": "mean: 1.7082302999980925 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 148.03857128308482,
            "unit": "iter/sec",
            "range": "stddev: 0.00016948256646172726",
            "extra": "mean: 6.754996291390594 msec\nrounds: 151"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_short_period",
            "value": 591.7976416469394,
            "unit": "iter/sec",
            "range": "stddev: 0.000045806660188766476",
            "extra": "mean: 1.6897667878788039 msec\nrounds: 594"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_long_period",
            "value": 192.3901203515871,
            "unit": "iter/sec",
            "range": "stddev: 0.00007901437428134933",
            "extra": "mean: 5.197772100628298 msec\nrounds: 159"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 258.7471398391082,
            "unit": "iter/sec",
            "range": "stddev: 0.00007432113776920889",
            "extra": "mean: 3.8647770198418847 msec\nrounds: 252"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 598.5814537917398,
            "unit": "iter/sec",
            "range": "stddev: 0.000035843674198644275",
            "extra": "mean: 1.6706164109587045 msec\nrounds: 584"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly_long_period",
            "value": 211.89528745586256,
            "unit": "iter/sec",
            "range": "stddev: 0.00008865859338397813",
            "extra": "mean: 4.719312128205297 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly_long_period",
            "value": 214.8838356962446,
            "unit": "iter/sec",
            "range": "stddev: 0.00010965820683071499",
            "extra": "mean: 4.653677168223949 msec\nrounds: 214"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 70.48203197332242,
            "unit": "iter/sec",
            "range": "stddev: 0.0005075559412117612",
            "extra": "mean: 14.188013199995453 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 37.67095011236899,
            "unit": "iter/sec",
            "range": "stddev: 0.000051997698890646274",
            "extra": "mean: 26.545653799999513 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_short_period",
            "value": 74.18262086906418,
            "unit": "iter/sec",
            "range": "stddev: 0.00017099121280399058",
            "extra": "mean: 13.480246293333948 msec\nrounds: 75"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_long_period",
            "value": 21.377355042198747,
            "unit": "iter/sec",
            "range": "stddev: 0.02147816234525098",
            "extra": "mean: 46.77847179999617 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1month",
            "value": 314.64727391058267,
            "unit": "iter/sec",
            "range": "stddev: 0.00008151263269178487",
            "extra": "mean: 3.178161970296246 msec\nrounds: 303"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_1year",
            "value": 236.4585829289154,
            "unit": "iter/sec",
            "range": "stddev: 0.00010696702545359173",
            "extra": "mean: 4.229070425836992 msec\nrounds: 209"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestDateRangeQueries::test_date_range_multi_year",
            "value": 191.89087381653982,
            "unit": "iter/sec",
            "range": "stddev: 0.00006552432556669827",
            "extra": "mean: 5.211295253968488 msec\nrounds: 189"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_cache_stats_retrieval",
            "value": 10540564.99362518,
            "unit": "iter/sec",
            "range": "stddev: 8.290139524764006e-9",
            "extra": "mean: 94.87157477846672 nsec\nrounds: 106180"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheOperations::test_repeated_same_query",
            "value": 58.64001172626697,
            "unit": "iter/sec",
            "range": "stddev: 0.00020851811426173282",
            "extra": "mean: 17.05320259259198 msec\nrounds: 54"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_interleaved_symbols",
            "value": 24.43964844910827,
            "unit": "iter/sec",
            "range": "stddev: 0.00031804024355984814",
            "extra": "mean: 40.917118840000626 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestConcurrentAccess::test_mixed_intervals",
            "value": 135.95712127510242,
            "unit": "iter/sec",
            "range": "stddev: 0.00008201636672655374",
            "extra": "mean: 7.355260177777302 msec\nrounds: 135"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_typical_analysis_workflow",
            "value": 100.35869437093741,
            "unit": "iter/sec",
            "range": "stddev: 0.00008550683539720077",
            "extra": "mean: 9.964258764705363 msec\nrounds: 102"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_multi_stock_comparison",
            "value": 195.73495775494166,
            "unit": "iter/sec",
            "range": "stddev: 0.00006577744290006556",
            "extra": "mean: 5.108949425641129 msec\nrounds: 195"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestRealWorldPatterns::test_dashboard_load",
            "value": 117.33573284774141,
            "unit": "iter/sec",
            "range": "stddev: 0.00011620076473490526",
            "extra": "mean: 8.522552983051053 msec\nrounds: 118"
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
          "id": "e89f2049f366fefede0a517fcef33a77c03ad3a7",
          "message": "fix ruff check",
          "timestamp": "2026-01-01T19:33:42+09:00",
          "tree_id": "d5757c4711a03e087456295e56c42b24515276b3",
          "url": "https://github.com/keisku/yfinance-mcp/commit/e89f2049f366fefede0a517fcef33a77c03ad3a7"
        },
        "date": 1767263682876,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_us",
            "value": 299.11151681968784,
            "unit": "iter/sec",
            "range": "stddev: 0.00006881698677230232",
            "extra": "mean: 3.343234692640825 msec\nrounds: 231"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_japan",
            "value": 54.37830335278068,
            "unit": "iter/sec",
            "range": "stddev: 0.000903091960555815",
            "extra": "mean: 18.389687399999843 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_europe",
            "value": 56.45219518029402,
            "unit": "iter/sec",
            "range": "stddev: 0.00048372963139797255",
            "extra": "mean: 17.71410300000298 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHit::test_cache_hit_varied_periods",
            "value": 78.13250550867862,
            "unit": "iter/sec",
            "range": "stddev: 0.00014885990001258425",
            "extra": "mean: 12.798770415584897 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolio::test_portfolio_scan",
            "value": 19.209186990892366,
            "unit": "iter/sec",
            "range": "stddev: 0.0005317337214304156",
            "extra": "mean: 52.05842394444539 msec\nrounds: 18"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_cache_hit",
            "value": 315.0099555532332,
            "unit": "iter/sec",
            "range": "stddev: 0.00009743024156685604",
            "extra": "mean: 3.17450284466013 msec\nrounds: 309"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_monthly_cache_hit",
            "value": 319.8225794901671,
            "unit": "iter/sec",
            "range": "stddev: 0.000050154056592874546",
            "extra": "mean: 3.126733583332708 msec\nrounds: 12"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestWeeklyMonthly::test_weekly_portfolio_scan",
            "value": 44.94745274205196,
            "unit": "iter/sec",
            "range": "stddev: 0.0003240824220891517",
            "extra": "mean: 22.248201822222946 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestIntradayCache::test_intraday_cache_hit",
            "value": 24057.70017874502,
            "unit": "iter/sec",
            "range": "stddev: 0.000006871298847722135",
            "extra": "mean: 41.566733003161296 usec\nrounds: 11105"
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
          "id": "9392422fe8da76056a9aa38dbbb16371d75e774f",
          "message": "deterministic and network-independent benchmark",
          "timestamp": "2026-01-01T19:52:07+09:00",
          "tree_id": "bc1c19de7c74cf40c134321e30a195b28d30be17",
          "url": "https://github.com/keisku/yfinance-mcp/commit/9392422fe8da76056a9aa38dbbb16371d75e774f"
        },
        "date": 1767264786922,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 30.149614083129766,
            "unit": "iter/sec",
            "range": "stddev: 0.06443664244032779",
            "extra": "mean: 33.167920400001094 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 218.81607288080264,
            "unit": "iter/sec",
            "range": "stddev: 0.0004818723074933055",
            "extra": "mean: 4.5700482000000875 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 220.92994026848453,
            "unit": "iter/sec",
            "range": "stddev: 0.00035047236961135644",
            "extra": "mean: 4.526321777776034 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 86.95572717284067,
            "unit": "iter/sec",
            "range": "stddev: 0.00015758482535311777",
            "extra": "mean: 11.500105082352013 msec\nrounds: 85"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 253.45184878817483,
            "unit": "iter/sec",
            "range": "stddev: 0.0000855626790986784",
            "extra": "mean: 3.9455226102365537 msec\nrounds: 254"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 256.3423455632336,
            "unit": "iter/sec",
            "range": "stddev: 0.00011796217987019249",
            "extra": "mean: 3.9010331976279886 msec\nrounds: 253"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 33.998743365645794,
            "unit": "iter/sec",
            "range": "stddev: 0.0010520324338283544",
            "extra": "mean: 29.412851800000794 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 36.555407092670265,
            "unit": "iter/sec",
            "range": "stddev: 0.000273987671265605",
            "extra": "mean: 27.355734200003212 msec\nrounds: 10"
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
          "id": "825380329cf1aff7cc20948aaf69e80879fc5aed",
          "message": "Add Investopedia references to technical indicators\n\n- README: Add hyperlinks to Investopedia definitions for all 19 indicators\n- server.py: Expand technicals tool description with key signal thresholds\n\nReferences: https://www.investopedia.com/terms",
          "timestamp": "2026-01-02T11:16:32+09:00",
          "tree_id": "549a8e9240567ab162ff46c376f62abeb4b307fe",
          "url": "https://github.com/keisku/yfinance-mcp/commit/825380329cf1aff7cc20948aaf69e80879fc5aed"
        },
        "date": 1767320276785,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 27.439157235909374,
            "unit": "iter/sec",
            "range": "stddev: 0.07200138351403243",
            "extra": "mean: 36.444268000013835 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 227.1885126220006,
            "unit": "iter/sec",
            "range": "stddev: 0.0003765241931611165",
            "extra": "mean: 4.401630999996087 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 225.69802191499426,
            "unit": "iter/sec",
            "range": "stddev: 0.0003474441092547301",
            "extra": "mean: 4.430698999996707 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 87.71230579162453,
            "unit": "iter/sec",
            "range": "stddev: 0.0001051234921055398",
            "extra": "mean: 11.400908811765476 msec\nrounds: 85"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 257.72364042284073,
            "unit": "iter/sec",
            "range": "stddev: 0.0000528803421422727",
            "extra": "mean: 3.8801252316602586 msec\nrounds: 259"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 259.1016432161357,
            "unit": "iter/sec",
            "range": "stddev: 0.000055852587190477726",
            "extra": "mean: 3.8594892243343533 msec\nrounds: 263"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 33.890386682059614,
            "unit": "iter/sec",
            "range": "stddev: 0.0012570377990479878",
            "extra": "mean: 29.5068925999999 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 37.11071616421863,
            "unit": "iter/sec",
            "range": "stddev: 0.00018463835484605972",
            "extra": "mean: 26.94639455554832 msec\nrounds: 9"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "cd6ec48cf5934e897d01c0538b7dc1809e720e00",
          "message": "Improve technical indicator descriptions with full abbreviation expansions",
          "timestamp": "2026-01-02T02:17:05Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/2/commits/cd6ec48cf5934e897d01c0538b7dc1809e720e00"
        },
        "date": 1767321803865,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 17.59802414453141,
            "unit": "iter/sec",
            "range": "stddev: 0.11740800090647524",
            "extra": "mean: 56.824561199999835 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 231.44904646914824,
            "unit": "iter/sec",
            "range": "stddev: 0.00035211708833606973",
            "extra": "mean: 4.320605400002364 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 226.35458506097098,
            "unit": "iter/sec",
            "range": "stddev: 0.00032460562213051556",
            "extra": "mean: 4.417847333336056 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 88.65017035154177,
            "unit": "iter/sec",
            "range": "stddev: 0.00015830033185783757",
            "extra": "mean: 11.280294172413942 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 260.14108284579146,
            "unit": "iter/sec",
            "range": "stddev: 0.00007220414085911401",
            "extra": "mean: 3.844067953668003 msec\nrounds: 259"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 261.39132771637907,
            "unit": "iter/sec",
            "range": "stddev: 0.00007667377805887733",
            "extra": "mean: 3.825681627376113 msec\nrounds: 263"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 34.95364081142636,
            "unit": "iter/sec",
            "range": "stddev: 0.0005664224344232291",
            "extra": "mean: 28.609322999997744 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 37.04882519564288,
            "unit": "iter/sec",
            "range": "stddev: 0.00031607998371897377",
            "extra": "mean: 26.9914091666692 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "keisku",
            "username": "keisku"
          },
          "committer": {
            "name": "keisku",
            "username": "keisku"
          },
          "id": "919562fcea03ec024e794f863d75636ac769ba81",
          "message": "Improve technical indicator descriptions with full abbreviation expansions",
          "timestamp": "2026-01-02T02:17:05Z",
          "url": "https://github.com/keisku/yfinance-mcp/pull/2/commits/919562fcea03ec024e794f863d75636ac769ba81"
        },
        "date": 1767322152199,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 17.917130048441237,
            "unit": "iter/sec",
            "range": "stddev: 0.11522660779104131",
            "extra": "mean: 55.81251000000407 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 231.4115116396627,
            "unit": "iter/sec",
            "range": "stddev: 0.00035337370378576635",
            "extra": "mean: 4.321306200000663 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 232.66524129730053,
            "unit": "iter/sec",
            "range": "stddev: 0.00016008126591772974",
            "extra": "mean: 4.298020600000996 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 88.18980995171951,
            "unit": "iter/sec",
            "range": "stddev: 0.00009281526178845915",
            "extra": "mean: 11.339178534883578 msec\nrounds: 86"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 256.4305103939107,
            "unit": "iter/sec",
            "range": "stddev: 0.0001582532885479838",
            "extra": "mean: 3.8996919612407646 msec\nrounds: 258"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 259.73570606370816,
            "unit": "iter/sec",
            "range": "stddev: 0.00004512357084653107",
            "extra": "mean: 3.850067498054038 msec\nrounds: 257"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 34.28328029036069,
            "unit": "iter/sec",
            "range": "stddev: 0.0006280628259877007",
            "extra": "mean: 29.168737399996303 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 36.34424813900302,
            "unit": "iter/sec",
            "range": "stddev: 0.0008243362719411326",
            "extra": "mean: 27.514670166662352 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "41987730+keisku@users.noreply.github.com",
            "name": "Keisuke Umegaki",
            "username": "keisku"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dc8e0f6318305ee63753a2758f747d53e825a7aa",
          "message": "Merge pull request #2 from keisku/claude/improve-indicator-descriptions-0mqCC",
          "timestamp": "2026-01-02T11:49:45+09:00",
          "tree_id": "bf164717e6cc7b22b7115afc4ca7741a78c14178",
          "url": "https://github.com/keisku/yfinance-mcp/commit/dc8e0f6318305ee63753a2758f747d53e825a7aa"
        },
        "date": 1767322243014,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 17.717525026598906,
            "unit": "iter/sec",
            "range": "stddev: 0.11642696814116694",
            "extra": "mean: 56.44129179999595 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 223.79299040965427,
            "unit": "iter/sec",
            "range": "stddev: 0.00037596075495828393",
            "extra": "mean: 4.468415200000209 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 205.1464684240242,
            "unit": "iter/sec",
            "range": "stddev: 0.0005611566208464906",
            "extra": "mean: 4.874566000000868 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 86.48435236672329,
            "unit": "iter/sec",
            "range": "stddev: 0.00022544478170411812",
            "extra": "mean: 11.562785320512749 msec\nrounds: 78"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 252.3877605581778,
            "unit": "iter/sec",
            "range": "stddev: 0.00009078135148155129",
            "extra": "mean: 3.9621572685950057 msec\nrounds: 242"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 245.19786737347033,
            "unit": "iter/sec",
            "range": "stddev: 0.00022747076094934397",
            "extra": "mean: 4.078338897119612 msec\nrounds: 243"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 33.53587728852601,
            "unit": "iter/sec",
            "range": "stddev: 0.0011731139687578063",
            "extra": "mean: 29.818811399997003 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 35.08379340124538,
            "unit": "iter/sec",
            "range": "stddev: 0.0008447851903755111",
            "extra": "mean: 28.503189166668125 msec\nrounds: 6"
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
          "id": "f7530447d8dacfe60dca391dfc4904b84a20611d",
          "message": "chore: Increase default period for history and technicals\n\n- history: 1mo -> 3mo (more useful context for price analysis)\n- technicals: 3mo -> 1y (ensures all indicators including SMA 200\n  have sufficient data points)",
          "timestamp": "2026-01-03T17:22:59+09:00",
          "tree_id": "0075f4f60ab10c0917cecddb14e1925b2a5f55cf",
          "url": "https://github.com/keisku/yfinance-mcp/commit/f7530447d8dacfe60dca391dfc4904b84a20611d"
        },
        "date": 1767428757588,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 17.15478232409986,
            "unit": "iter/sec",
            "range": "stddev: 0.12002967329984135",
            "extra": "mean: 58.29278279999812 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 218.43663762565356,
            "unit": "iter/sec",
            "range": "stddev: 0.00043908557482458336",
            "extra": "mean: 4.577986600003214 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 214.83686598898663,
            "unit": "iter/sec",
            "range": "stddev: 0.00046271295289949043",
            "extra": "mean: 4.654694600000653 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 85.30427272945528,
            "unit": "iter/sec",
            "range": "stddev: 0.00029543341284900986",
            "extra": "mean: 11.722742226189842 msec\nrounds: 84"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 251.4875187224654,
            "unit": "iter/sec",
            "range": "stddev: 0.00010972376972412724",
            "extra": "mean: 3.9763404763779633 msec\nrounds: 254"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 253.124706661203,
            "unit": "iter/sec",
            "range": "stddev: 0.000055236671916822926",
            "extra": "mean: 3.9506218622050944 msec\nrounds: 254"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 33.71512589136407,
            "unit": "iter/sec",
            "range": "stddev: 0.0007970472061414793",
            "extra": "mean: 29.66027780000502 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 36.252201271426316,
            "unit": "iter/sec",
            "range": "stddev: 0.0001537509170007961",
            "extra": "mean: 27.5845318333315 msec\nrounds: 6"
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
          "id": "1def161b2b94febdd1f84cc27f92598bf5914536",
          "message": "fix(server): Standardize tool input schema consistency\n\nAdd missing enum and description to technicals.period parameter to\nmatch history.period. Add descriptions to financials.statement and\nfinancials.freq parameters. Standardize limit description format\nacross history and financials tools.",
          "timestamp": "2026-01-03T17:27:48+09:00",
          "tree_id": "b8eccc2b072e700553aba171d4ee28a48f937554",
          "url": "https://github.com/keisku/yfinance-mcp/commit/1def161b2b94febdd1f84cc27f92598bf5914536"
        },
        "date": 1767428935474,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 14.395818648099105,
            "unit": "iter/sec",
            "range": "stddev: 0.1453225901041361",
            "extra": "mean: 69.46461499999828 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 228.2681840716933,
            "unit": "iter/sec",
            "range": "stddev: 0.00036348241623424644",
            "extra": "mean: 4.3808120000022654 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 219.78368801507906,
            "unit": "iter/sec",
            "range": "stddev: 0.0004027244414674875",
            "extra": "mean: 4.549928200000863 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 87.67104372481606,
            "unit": "iter/sec",
            "range": "stddev: 0.00012625355999846863",
            "extra": "mean: 11.40627460919507 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 254.21148633196324,
            "unit": "iter/sec",
            "range": "stddev: 0.00018974249258832903",
            "extra": "mean: 3.933732556420151 msec\nrounds: 257"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 257.22483136382084,
            "unit": "iter/sec",
            "range": "stddev: 0.00020713731405821957",
            "extra": "mean: 3.887649550387272 msec\nrounds: 258"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 34.864220803976025,
            "unit": "iter/sec",
            "range": "stddev: 0.0005868859824187148",
            "extra": "mean: 28.682700400003114 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 37.00177239018064,
            "unit": "iter/sec",
            "range": "stddev: 0.00015314250229788573",
            "extra": "mean: 27.025732428573487 msec\nrounds: 7"
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
          "id": "eae6a975074cf2acb45c8c9020d543afa58704fc",
          "message": "update readme",
          "timestamp": "2026-01-03T17:29:43+09:00",
          "tree_id": "433541a205cf4972bc0fb9860b73ac25021bf46e",
          "url": "https://github.com/keisku/yfinance-mcp/commit/eae6a975074cf2acb45c8c9020d543afa58704fc"
        },
        "date": 1767429042827,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 25.573557481679217,
            "unit": "iter/sec",
            "range": "stddev: 0.07709322750065964",
            "extra": "mean: 39.10288979999734 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 217.97650492284018,
            "unit": "iter/sec",
            "range": "stddev: 0.0005102138841646093",
            "extra": "mean: 4.587650400000598 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 214.7945687562604,
            "unit": "iter/sec",
            "range": "stddev: 0.00033669065595464515",
            "extra": "mean: 4.655611199996201 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 85.29338254537006,
            "unit": "iter/sec",
            "range": "stddev: 0.0002426126241372501",
            "extra": "mean: 11.724238975609515 msec\nrounds: 82"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 252.3047279988385,
            "unit": "iter/sec",
            "range": "stddev: 0.00011737259131133465",
            "extra": "mean: 3.9634612000001983 msec\nrounds: 225"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 257.92336010658966,
            "unit": "iter/sec",
            "range": "stddev: 0.00006043317433029219",
            "extra": "mean: 3.8771207058823176 msec\nrounds: 255"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 34.17838501648156,
            "unit": "iter/sec",
            "range": "stddev: 0.0008865580803174907",
            "extra": "mean: 29.258257800003662 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 36.59323482718022,
            "unit": "iter/sec",
            "range": "stddev: 0.00042097309476913833",
            "extra": "mean: 27.327455599996142 msec\nrounds: 10"
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
          "id": "666e217954bb746d2efbe00c1f450a0d98ec50b0",
          "message": "perf(mcp): Auto-downsample time-series for consistent LLM token usage\n\nStabilize data point output from history and technicals tools to prevent\nLLM processing slowdowns from large time-series responses.\n\nChanges:\n- Add auto-interval selection based on time span (calculate_span_days,\n  auto_interval, auto_downsample in helpers.py)\n- Remove explicit interval/format params from history tool schema\n- Apply auto-downsampling to technicals output after indicator calculation\n- Add MAX_SPAN_DAYS (1120 days) validation with DateRangeExceededError\n- Generate dynamic period enum options via get_valid_periods()\n- Make TARGET_DATA_POINTS/MAX_SPAN_DAYS configurable via environment\n\nDefault 80 data points (~4-8K tokens) at biweekly resolution balances\ndetail with LLM context efficiency.",
          "timestamp": "2026-01-03T23:25:46+09:00",
          "tree_id": "ed2df1e26e8b6a4d3c4d9a0e27b097b59f6b705a",
          "url": "https://github.com/keisku/yfinance-mcp/commit/666e217954bb746d2efbe00c1f450a0d98ec50b0"
        },
        "date": 1767450409440,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_us_symbol",
            "value": 30.16245461863855,
            "unit": "iter/sec",
            "range": "stddev: 0.06468631605824839",
            "extra": "mean: 33.15380039998672 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_tokyo_symbol",
            "value": 226.1522150052216,
            "unit": "iter/sec",
            "range": "stddev: 0.0004243250279405507",
            "extra": "mean: 4.42180059999373 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_europe_symbol",
            "value": 230.6182172270785,
            "unit": "iter/sec",
            "range": "stddev: 0.0002925028075204863",
            "extra": "mean: 4.33617088894304 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitDaily::test_cache_hit_varied_periods",
            "value": 88.13006029752286,
            "unit": "iter/sec",
            "range": "stddev: 0.0001784079558826445",
            "extra": "mean: 11.346866172836464 msec\nrounds: 81"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_weekly",
            "value": 257.522640569674,
            "unit": "iter/sec",
            "range": "stddev: 0.00008107576068295385",
            "extra": "mean: 3.883153721116979 msec\nrounds: 251"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestCacheHitWeeklyMonthly::test_cache_hit_monthly",
            "value": 259.6045392324842,
            "unit": "iter/sec",
            "range": "stddev: 0.00010688463707021841",
            "extra": "mean: 3.852012768946493 msec\nrounds: 264"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_daily",
            "value": 34.93372189131934,
            "unit": "iter/sec",
            "range": "stddev: 0.00010568269943578202",
            "extra": "mean: 28.625635800017335 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_price.py::TestPortfolioScans::test_portfolio_scan_weekly",
            "value": 36.931953167635186,
            "unit": "iter/sec",
            "range": "stddev: 0.0003630346284136022",
            "extra": "mean: 27.076824111115155 msec\nrounds: 9"
          }
        ]
      }
    ]
  }
}