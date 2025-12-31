window.BENCHMARK_DATA = {
  "lastUpdate": 1767203590794,
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
      }
    ]
  }
}