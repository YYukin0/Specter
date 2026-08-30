window.SPECTER_CLOUD_BENCH = {
  "status": "ready",
  "target_model": "meta-llama/Llama-3.1-8B-Instruct",
  "draft_model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
  "locked_config": {
    "num_speculative_tokens": 3,
    "temperature": 0.0,
    "top_p": 1.0,
    "output_len": 1024,
    "dataset": "openai/gsm8k",
    "concurrencies": [
      1,
      4,
      16,
      32,
      64
    ]
  },
  "arms": [
    {
      "name": "eagle3",
      "points": [
        {
          "concurrency": 1,
          "speedup": 2.3589753012384413,
          "ttft_p99_ms": 1469.139575958252,
          "tpot_p99_ms": 28.265960456789003
        },
        {
          "concurrency": 4,
          "speedup": 2.463663280738868,
          "ttft_p99_ms": 1877.3980140686035,
          "tpot_p99_ms": 26.717491222150397
        },
        {
          "concurrency": 16,
          "speedup": 2.4162983712434456,
          "ttft_p99_ms": 150.04229545593262,
          "tpot_p99_ms": 17.286755317865417
        },
        {
          "concurrency": 32,
          "speedup": 2.2398001446429454,
          "ttft_p99_ms": 218.18089485168457,
          "tpot_p99_ms": 20.13142199455937
        },
        {
          "concurrency": 64,
          "speedup": 1.6269664475696286,
          "ttft_p99_ms": 406.6288471221924,
          "tpot_p99_ms": 31.07585388681163
        }
      ]
    },
    {
      "name": "ngram",
      "points": [
        {
          "concurrency": 1,
          "speedup": 1.5155678954424243,
          "ttft_p99_ms": 1302.5994300842285,
          "tpot_p99_ms": 34.72516148589378
        },
        {
          "concurrency": 4,
          "speedup": 1.2595072843854072,
          "ttft_p99_ms": 82.2441577911377,
          "tpot_p99_ms": 30.47420623454642
        },
        {
          "concurrency": 16,
          "speedup": 1.2745276424381022,
          "ttft_p99_ms": 110.57519912719727,
          "tpot_p99_ms": 31.05462751080913
        },
        {
          "concurrency": 32,
          "speedup": 1.2560810936612712,
          "ttft_p99_ms": 186.9516372680664,
          "tpot_p99_ms": 35.4704392158379
        },
        {
          "concurrency": 64,
          "speedup": 1.0893462359603276,
          "ttft_p99_ms": 264.15395736694336,
          "tpot_p99_ms": 47.405735651652016
        }
      ]
    },
    {
      "name": "baseline",
      "points": [
        {
          "concurrency": 1,
          "speedup": 1.0,
          "ttft_p99_ms": 1390.775203704834,
          "tpot_p99_ms": 39.558477179948675
        },
        {
          "concurrency": 4,
          "speedup": 1.0,
          "ttft_p99_ms": 1155.2605628967285,
          "tpot_p99_ms": 40.327320664615954
        },
        {
          "concurrency": 16,
          "speedup": 1.0,
          "ttft_p99_ms": 86.64941787719727,
          "tpot_p99_ms": 33.12251948508896
        },
        {
          "concurrency": 32,
          "speedup": 1.0,
          "ttft_p99_ms": 180.05728721618652,
          "tpot_p99_ms": 35.119709552534474
        },
        {
          "concurrency": 64,
          "speedup": 1.0,
          "ttft_p99_ms": 258.7850093841553,
          "tpot_p99_ms": 42.56787032724541
        }
      ]
    }
  ]
};
