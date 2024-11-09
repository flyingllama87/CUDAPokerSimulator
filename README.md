# CUDA Poker Simulator

Back in 2015 or so, a work colleague issued a challenge to write a faster Texas Hold ‘Em poker simulator than he had written. He provided his source in C++. His only stipulation was that the simulation had to be executed on the work PCs, so emphasis was placed on the code.

In the simulation, virtual players are dealt their hand along with the community cards. The player’s hands are evaluated, and statistics on the winning player and hand are recorded and accumulated. This happens hundreds of thousands of times on modern CPUs.

After reviewing his clean code, I didn’t think I could rewrite the logic significantly faster, so I took a different approach and decided to port his code over to CUDA, NVIDIA’s GPGPU platform. It was a challenge, but I eventually ported his simulation code after rewriting significant sections to run better on GPGPUs.

In 2015, this resulted in a simulator that executed 524,288 poker games per second on the 960GTX GPU (a $260 AUD device at the time) compared to 453,266 games per second on the i7 CPU (a $400 AUD device at the time).

By 2017, on a 1060GTX GPU ($400 AUD), it executed over 1,000,000 poker games per second compared to 429,093 on an i7 8600K CPU ($400 AUD device).

In 2024, I revisited the project with newer GPUs. I updated the code and ran it on updated hardware. The results are in the table below.

An interesting development in coding is, of course, Large Language Models (LLMs). I provided ChatGPT with the source code for both the CPU and GPU variants for optimization. After some tweaking, it produced a much more performant simulation! Hilariously, despite improving the simulation after a few tries, it forgot about all the timing and sanity-checking features I'd implemented and for some bizarre reason, it couldn't produce a working program with them— it would break other features when trying to add these basic functionalities. Still, I am impressed by it's optimization. Turns out, there was a lot of work that could have been done on the simulation implementation.

### Benchmark Results

| Device         | Program & Test               | Games per second       | Total time (ms) |
|----------------|------------------------------|------------------------|-----------------|
| Geforce 4080   | 100,000,000 Games CUDA GPT   | 317,633,376            | 314.5           |
| Geforce 4080   | 10,000,000 Games CUDA GPT    | 297,380,160            | 33.5            |
| Geforce 3060   | 100,000,000 Games CUDA GPT   | 133,293,456            | 750.2           |
| Geforce 3060   | 10,000,000 Games CUDA GPT    | 120,097,656            | 83.1            |
| RYZEN 9950X    | 10,000,000 Games CPU GPT     | 12,487,072             | 10,017          |
| Geforce 4080   | 100,000,000 Games CUDA       | 19,705,616             | 5,077           |
| Geforce 4080   | 10,000,000 Games CUDA        | 19,012,534             | 527.8           |
| 14700K CPU     | 10 Seconds CPU GPT           | 9,128,558.6            | 10,011          |
| 14700K CPU     | 10,000,000 Games CPU GPT     | 9,020,373.5            | 10,013          |
| Geforce 3060   | 100,000,000 Games CUDA       | 6,919,743              | 14,453.1        |
| Geforce 3060   | 10,000,000 Games CUDA        | 6,819,891.5            | 1,468.8         |
| RYZEN 9950X    | 10,000,000 Games CPU         | 2,161,760              | 4,262.4         |
| 14700K CPU     | 10,000,000 Games CPU         | 1,825,330              | 5,283.15        |
| 8600K CPU      | 10,000,000 Games CPU GPT     | 1,351,683.5            | 10,009          |
| 8600K CPU      | 10 Seconds CPU GPT           | 1,348,607.4            | 10,017          |
| Geforce 1060   | 10 second test CUDA 2017     | ~1,000,000             | N/A             |
| Geforce 960    | 10 second test CUDA 2016     | 524,288                | N/A             |
| 4790K CPU      | 10 second test CPU  2016     | 463,811                | N/A             |
| 8600K CPU      | 10,000,000 Games CPU         | 429,093                | 23,216.7        |


- '2015 Era CPU to GPU Uplift': '13.04%'
- '14700K CPU to GPT CPU Uplift': '394.18%'
- '8600K CPU to GPT CPU Uplift': '215.01%'
- '3060 CUDA to GPT CUDA Uplift': '1826.28%'
- '4080 CUDA to GPT CUDA Uplift': '1511.89%'
- '9950X CPU to 3060 CUDA Uplift': '220.10%'
- '9950X CPU to 4080 CUDA Uplift': '811.55%'