

# CUDA Poker Simulator

### Table of Contents
- [Introduction](#introduction)
- [LLMs](#llms)
- [Further Optimizations](#further-optimizations)
- [Benchmark Results](#benchmark-results)
- [Performance Analysis](#performance-analysis)
  - [CPU vs GPU](#cpu-vs-gpu)
  - [Original vs GPT Optimized Versions](#original-vs-gpt-optimized-versions)

### Introduction

Back in 2015 or so, a work colleague issued a challenge to write a faster Texas Hold ‘Em poker simulator than he had written. He provided his source in C++. His only stipulation was that the simulation had to be executed on the work PCs, so emphasis was placed on the code.

In the simulation, virtual players are dealt their hand along with the community cards. The player’s hands are evaluated, and statistics on the winning player and hand are recorded and accumulated. This happens hundreds of thousands of times on modern CPUs.

I didn’t think I could rewrite the logic significantly faster, so I took a different approach and decided to port his code over to CUDA, NVIDIA’s GPGPU platform. It was a challenge, but I eventually ported his simulation code after rewriting significant sections to run better on GPGPUs.

I managed to get it running better on a GPU as they are best for parallel processing. Since then, the pace of GPU development has far outstripped CPUs and thus what was a difference of 13% faster in 2015 (i7 4790K vs Geforce 960GTX) is 279% faster in 2024 (i7 14700K vs Geforce 3060).

In 2024, I revisited the project. I updated the code and ran it on updated hardware. The results are published in this document below. Whilst revisting the project, I realized that not reviewing the logic of the poker simulation or researching public algorithms for poker hand evaluation and card shuffling was a mistake. It turns out, the implemented data structures and algorthims were very inefficient. I believe the data structures are more forgivable as they model poker quite well and make it easy to reason about. The algorithm for hand evaluation was by implemented inefficiently. 

### LLMs

An interesting development in coding is, of course, Large Language Models (LLMs). When I revisited the project in 2024, I provided ChatGPT with the source code for both the CPU and GPU variants for optimization. After some tweaking, it produced a much more performant simulation!

Hilariously, despite improving the simulation after a few tries, it forgot about all the timing and sanity-checking features I'd implemented and for some bizarre reason, it couldn't produce a working program with them— it would break other features when trying to add these basic functionalities. Still, I am impressed by it's optimization.

The performance speed up led me to look into the poker simulation implementation and public methods and realize that one should not assume that a seasoned professional programmer will neccessarily have the best approach. 

LLMs are no substitite for a human developer though. Initially, I was experiencing a memory bug with this project running with newer versions of CUDA. No LLM could provide any insight into the issue cause. Tried and true step-through debugging by a human was needed and it was determined that the root cause was nVidia changing the behaviour of their CUDA random number library.

I *do* think that they are a good resource and productivity tool, but they need an actual thinking person to challenge and curate the LLM output. 

### Further Optimizations

Here are some ideas for even faster simulation:

- Hand Eval: Hashing and Lookup Tables. Could we store all combinations into a usable space?
- Hand Eval: Represent hands using bitmasks and perform evaluation with bitwise operations. std::bitset may be of use here?
- Data Structures: Use structures of arrays (SoA) instead of arrays of structures (AoS) to maximize cache hits and minimize cache misses.
- CPU Version: Leverage CPU SIMD or at least, confirm the machine code produced is leveraging it.
- RNG: Use RNGs like the Xorshift or Permuted Congruential Generator that are known for speed. If the number distribution  
- RNG: With a simpler RNG implementation, RNG instances could be stored in thread-local storage.
- Look into "Poker-Eval" Library or "Cactus Kev's" implementations. 

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

### Performance Analysis

##### CPU vs GPU
- 2015 CPU to GPU (non-GPT) Uplift (4790K vs 960): **13.04%**
- 2024 CPU to GPU (non-GPT) Uplift (14700K vs 3060): **279.10%**
- 2024 CPU (GPT) to GPU (GPT) Uplift (14700K vs 3060): **1377.69%**

##### Original vs GPT optimized versions 
- 14700K CPU to CPU (GPT) Uplift: **394.18%**
- 8600K CPU to CPU (GPT) Uplift: **215.01%**
- 3060 CUDA to CUDA (GPT) Uplift: **1826.28%**
- 4080 CUDA to CUDA (GPT) Uplift: **1511.89%**
