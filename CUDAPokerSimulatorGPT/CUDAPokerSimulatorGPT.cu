#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iomanip>

using namespace std;

#define NUM_SUITS 4
#define NUM_CARDS 13
#define NUM_HANDS 9
#define NUM_PLAYERS 9

int gGridSize = 28;
int gBlockSize = 640;
int gGamesPerKernel = 1000;

enum HandType { HighCard = 0, OnePair, TwoPair, ThreeOfAKind, Straight, Flush, FullHouse, FourOfAKind, StraightFlush };
enum CardType { c2 = 0, c3, c4, c5, c6, c7, c8, c9, cT, cJ, cQ, cK, cA };
enum SuitType { Club = 0, Diamond, Heart, Spade };

const char* pHands[NUM_HANDS] = { "HighCard", "OnePair", "TwoPair", "ThreeOfAKind", "Straight", "Flush", "FullHouse", "FourOfAKind", "StraightFlush" };

struct Card {
    unsigned char value; // 0-51 representing 52 unique cards

    __device__ Card() : value(0) {}
    __device__ Card(unsigned char v) : value(v) {}

    __device__ CardType getCard() const {
        return (CardType)(value % NUM_CARDS);
    }

    __device__ SuitType getSuit() const {
        return (SuitType)(value / NUM_CARDS);
    }

    __device__ bool operator>(const Card& c) const {
        return getCard() > c.getCard();
    }

    __device__ bool operator==(const Card& c) const {
        return getCard() == c.getCard();
    }
};

struct Player {
    Card mCards[2];
};

struct Hand {
    HandType mType;
    int mValue;

    __device__ Hand() : mType(HighCard), mValue(0) {}
};

struct TableStats {
    unsigned int mHands[NUM_HANDS];
    unsigned int mWinnerCount[NUM_PLAYERS];
    unsigned int mWinners;
    unsigned int mHandsPlayed;

    __device__ __host__ void reset() {
        mWinners = 0;
        mHandsPlayed = 0;
        for (int i = 0; i < NUM_HANDS; ++i) mHands[i] = 0;
        for (int i = 0; i < NUM_PLAYERS; ++i) mWinnerCount[i] = 0;
    }
};

__device__ void shuffleDeck(unsigned char* deck, curandState_t* state) {
    // Initialize deck
    for (int i = 0; i < 52; ++i) {
        deck[i] = i;
    }

    // Fisher-Yates shuffle
    for (int i = 51; i > 0; --i) {
        int j = curand(state) % (i + 1);
        unsigned char temp = deck[i];
        deck[i] = deck[j];
        deck[j] = temp;
    }
}

__device__ void evaluateHand(const Card* playerCards, const Card* communityCards, Hand* hand) {
    int counts[13] = { 0 };
    int suits[4] = { 0 };
    unsigned char allCards[7];

    // Combine player and community cards
    allCards[0] = playerCards[0].value;
    allCards[1] = playerCards[1].value;
    for (int i = 0; i < 5; ++i) {
        allCards[2 + i] = communityCards[i].value;
    }

    // Count card ranks and suits
    for (int i = 0; i < 7; ++i) {
        int cardRank = allCards[i] % NUM_CARDS;
        int cardSuit = allCards[i] / NUM_CARDS;
        counts[cardRank]++;
        suits[cardSuit]++;
    }

    // Check for flush
    bool isFlush = false;
    for (int i = 0; i < NUM_SUITS; ++i) {
        if (suits[i] >= 5) {
            isFlush = true;
            break;
        }
    }

    // Check for straight
    int straightHigh = -1;
    int consec = 0;
    for (int i = 12; i >= 0; --i) {
        if (counts[i] > 0) {
            consec++;
            if (consec >= 5) {
                straightHigh = i + 4;
                break;
            }
        }
        else {
            consec = 0;
        }
    }
    // Special case for wheel straight (A-2-3-4-5)
    if (consec == 4 && counts[12] && counts[0]) {
        straightHigh = 3; // 5 high straight
    }

    // Determine hand type
    int four = -1, three = -1;
    int pair1 = -1, pair2 = -1;
    for (int i = 12; i >= 0; --i) {
        if (counts[i] == 4) four = i;
        else if (counts[i] == 3) three = i;
        else if (counts[i] == 2) {
            if (pair1 == -1) pair1 = i;
            else if (pair2 == -1) pair2 = i;
        }
    }

    if (isFlush && straightHigh != -1) {
        hand->mType = StraightFlush;
        hand->mValue = straightHigh;
    }
    else if (four != -1) {
        hand->mType = FourOfAKind;
        hand->mValue = four;
    }
    else if (three != -1 && pair1 != -1) {
        hand->mType = FullHouse;
        hand->mValue = three * 13 + pair1;
    }
    else if (isFlush) {
        hand->mType = Flush;
    }
    else if (straightHigh != -1) {
        hand->mType = Straight;
        hand->mValue = straightHigh;
    }
    else if (three != -1) {
        hand->mType = ThreeOfAKind;
        hand->mValue = three;
    }
    else if (pair1 != -1 && pair2 != -1) {
        hand->mType = TwoPair;
        hand->mValue = pair1 * 13 + pair2;
    }
    else if (pair1 != -1) {
        hand->mType = OnePair;
        hand->mValue = pair1;
    }
    else {
        hand->mType = HighCard;
        for (int i = 12; i >= 0; --i) {
            if (counts[i] > 0) {
                hand->mValue = i;
                break;
            }
        }
    }
}

__device__ void playGame(TableStats* ts, curandState_t* state) {
    unsigned char deck[52];
    shuffleDeck(deck, state);

    // Deal player cards
    Player players[NUM_PLAYERS];
    int deckIndex = 0;
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        players[i].mCards[0] = Card(deck[deckIndex++]);
        players[i].mCards[1] = Card(deck[deckIndex++]);
    }

    // Deal community cards
    Card communityCards[5];
    for (int i = 0; i < 5; ++i) {
        communityCards[i] = Card(deck[deckIndex++]);
    }

    // Evaluate hands
    Hand hands[NUM_PLAYERS];
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        evaluateHand(players[i].mCards, communityCards, &hands[i]);
    }

    // Find the best hand(s)
    Hand bestHand = hands[0];
    int winners[NUM_PLAYERS];
    int numWinners = 1;
    winners[0] = 0;
    for (int i = 1; i < NUM_PLAYERS; ++i) {
        if (hands[i].mType > bestHand.mType || (hands[i].mType == bestHand.mType && hands[i].mValue > bestHand.mValue)) {
            bestHand = hands[i];
            winners[0] = i;
            numWinners = 1;
        }
        else if (hands[i].mType == bestHand.mType && hands[i].mValue == bestHand.mValue) {
            winners[numWinners++] = i;
        }
    }

    // Update statistics
    for (int i = 0; i < NUM_PLAYERS; ++i) {
        atomicAdd(&ts->mHands[hands[i].mType], 1);
        atomicAdd(&ts->mHandsPlayed, 1);
    }
    for (int i = 0; i < numWinners; ++i) {
        atomicAdd(&ts->mWinnerCount[winners[i]], 1);
        atomicAdd(&ts->mWinners, 1);
    }
}

__global__ void RunGames(TableStats* ts, int seed, int gamesPerKernel) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;

    curandState_t state;
    curand_init(seed, tId, 0, &state);

    // Initialize per-thread statistics
    __shared__ TableStats sharedStats;
    if (threadIdx.x == 0) {
        sharedStats.reset();
    }
    __syncthreads();

    for (int i = 0; i < gamesPerKernel; ++i) {
        playGame(&sharedStats, &state);
    }
    __syncthreads();

    // Aggregate results into global memory
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_HANDS; ++i) {
            atomicAdd(&ts->mHands[i], sharedStats.mHands[i]);
        }
        for (int i = 0; i < NUM_PLAYERS; ++i) {
            atomicAdd(&ts->mWinnerCount[i], sharedStats.mWinnerCount[i]);
        }
        atomicAdd(&ts->mHandsPlayed, sharedStats.mHandsPlayed);
        atomicAdd(&ts->mWinners, sharedStats.mWinners);
    }
}

int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(10); // Adjust precision as needed

    int gridSize = 28;
    int blockSize = 640;
    int gamesPerKernel = 1000;

    cudaEvent_t start, memAllocStop, kernelLaunchStop, resultsCalculated, kernelCompleteStop, memDeallocStop;
    cudaEventCreate(&start);
    cudaEventCreate(&memAllocStop);
    cudaEventCreate(&kernelLaunchStop);
    cudaEventCreate(&kernelCompleteStop);
    cudaEventCreate(&resultsCalculated);
    cudaEventCreate(&memDeallocStop);

    // Start overall timer
    cudaEventRecord(start, 0);

    int recommendedBlockSize;      // The launch configurator returned block size 
    int recommendedGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 

    cudaOccupancyMaxPotentialBlockSize(&recommendedGridSize, &recommendedBlockSize, RunGames, 0, 0);
    printf("Recommended CUDA kernel \"gridsize\" is %d and \"blocksize\" is %d.\n\n", recommendedGridSize, recommendedBlockSize);

    gridSize = recommendedGridSize;
    blockSize = recommendedBlockSize;


    // Parse target games from command-line arguments
    int targetGames = -1;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-target" && i + 1 < argc) {
            targetGames = std::stoi(argv[i + 1]);
            break;
        }
    }

    if (targetGames > 0) {
        int totalGamesPerLaunch = gridSize * blockSize * gamesPerKernel;
        gamesPerKernel = targetGames / (gridSize * blockSize);
        if (gamesPerKernel < 1) gamesPerKernel = 1;
        printf("Adjusting games per kernel to %d to meet target of %d games.\n", gamesPerKernel, targetGames);
    }

    // Allocate statistics on device and host
    TableStats* d_ts;
    cudaMalloc((void**)&d_ts, sizeof(TableStats));
    cudaMemset(d_ts, 0, sizeof(TableStats));

    TableStats h_ts;
    memset(&h_ts, 0, sizeof(TableStats));

    int Seed = (int)time(NULL);
    cudaEventRecord(memAllocStop, 0);
    // Launch kernel
    printf("Launching kernel with grid size %d, block size %d, games per kernel %d\n", gridSize, blockSize, gamesPerKernel);
    RunGames <<<gridSize, blockSize >>> (d_ts, Seed, gamesPerKernel);
    cudaEventRecord(kernelLaunchStop, 0);

    cudaDeviceSynchronize();
    cudaEventRecord(kernelCompleteStop, 0);

    // Copy results back to host
    cudaMemcpy(&h_ts, d_ts, sizeof(TableStats), cudaMemcpyDeviceToHost);

    // Display results
    unsigned int totalGames = h_ts.mHandsPlayed / NUM_PLAYERS;
    printf("\nTotal Games Played: %u\n", totalGames);
    printf("Total Hands Played: %u\n", h_ts.mHandsPlayed);
    printf("Total Winners: %u\n", h_ts.mWinners);

    for (int i = 0; i < NUM_PLAYERS; ++i) {
        printf("Player %d won %u times (%.2f%%)\n", i + 1, h_ts.mWinnerCount[i], (h_ts.mWinnerCount[i] * 100.0f) / h_ts.mWinners);
    }

    printf("\nHand Type Statistics:\n");
    for (int i = 0; i < NUM_HANDS; ++i) {
        printf("%15s: %8.4f%% (%u times)\n", pHands[i], (h_ts.mHands[i] * 100.0f) / h_ts.mHandsPlayed, h_ts.mHands[i]);
    }

    cudaEventRecord(resultsCalculated, 0);

    // Clean up

    cudaFree(d_ts);

    cudaEventRecord(memDeallocStop, 0);

    float memAllocTime, kernelLaunchTime, kernelExecTime, resultsExecTime, memDeallocTime, totalTime;
    cudaEventElapsedTime(&memAllocTime, start, memAllocStop);
    cudaEventElapsedTime(&kernelLaunchTime, memAllocStop, kernelLaunchStop);
    cudaEventElapsedTime(&kernelExecTime, kernelLaunchStop, kernelCompleteStop);
    cudaEventElapsedTime(&resultsExecTime, kernelCompleteStop, resultsCalculated);
    cudaEventElapsedTime(&memDeallocTime, resultsCalculated, memDeallocStop);
    cudaEventElapsedTime(&totalTime, start, resultsCalculated);

    // Display timings in milliseconds
    std::cout << "\n **** EXECUTION TIMINGS: **** \n" << std::endl;
    std::cout << "Memory allocation time: " << memAllocTime << " ms" << std::endl;
    std::cout << "Kernel launch time: " << kernelLaunchTime << " ms" << std::endl;
    std::cout << "Kernel execution time: " << kernelExecTime << " ms" << std::endl;
    std::cout << "CPU results calculations: " << resultsExecTime << " ms" << std::endl;
    std::cout << "Memory deallocation time: " << memDeallocTime << " ms" << std::endl;
    std::cout << "TOTAL TIME: " << totalTime << " ms" << std::endl;
    float timePerGame = totalTime / (gGridSize * blockSize * gamesPerKernel);
    std::cout << "\nTime per game: " << timePerGame << " ms" << std::endl;

	float GamesPerSecond = totalGames / (totalTime / 1000.0f);
	std::cout << "Games per second: " << GamesPerSecond << std::endl;


    std::cout << "Press Enter to continue...";
    std::cin.get();

    return 0;
}
