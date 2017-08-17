# CUDAPokerSimulator
CUDA poker simulator

A work colleague (Simon Eschbach) issued a challenge to write a faster Texas Hold ‘Em poker statistics simulator than he had written and provided source code for his simulator. His only stipulation was that the simulation had to be executed on the work PCs so emphasis was placed on the code.

In the simulation, virtual players are dealt their hand along with the community cards. The player’s hands are evaluated and statistics on the winning player and hand are recorded and accumulated. This happens hundreds of thousands of times on modern CPUs.

After looking at his clean code I didn’t think I could re-write the logic to significantly faster so I took a different approach and decided to try to port his code over to CUDA, nVidia’s GPGPU platform. It was a challenge but I eventually ported his simulation code over to CUDA after re-writing significnat sections to the code to run better on GPGPUS.

In 2015, the result of this work was a simulator that executed 524,288 poker games per second on the 960GTX GPU ($260 device) compared to 453,266 games per second on the i7 CPU ($400 AUD device).

In 2017, the result of this work was a simulator that executed over a million poker games per second on the 1060GTX GPU ($400 device) compared to just over 500,000 games per second on the i7 CPU ($400 AUD device).

The simulation source and diff is provided below. If you want to run the binary, note that you’ll need a maxwell class nVidia card (700 series and up).
