#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <algorithm>
#include <chrono>
#include <memory>  // For std::unique_ptr
#include <iomanip>

using namespace std;

#define NUM_SUITS 4
#define NUM_CARDS 13
#define NUM_HANDS 9
#define MAX_NUM_PLAYERS 9

enum HandType { HighCard = 0, OnePair, TwoPair, ThreeOfAKind, Straight, Flush, FullHouse, FourOfAKind, StraightFlush, MaxHand };
enum CardType { c2 = 0, c3, c4, c5, c6, c7, c8, c9, cT, cJ, cQ, cK, cA, MaxCard };
enum SuitType { Club = 0, Diamond, Heart, Spade, MaxSuit };

const char* pHands[NUM_HANDS] = { "HighCard", "OnePair", "TwoPair", "ThreeOfAKind", "Straight", "Flush", "FullHouse", "FourOfAKind", "StraightFlush" };
const char pCards[NUM_CARDS] = { '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A' };
const char pSuits[NUM_SUITS] = { 'C', 'D', 'H', 'S' };

std::atomic<unsigned long> gGameCounter(0);

struct Card {
    CardType mCard;
    SuitType mSuit;

    Card() : mCard(MaxCard), mSuit(MaxSuit) {}

    void set(SuitType suit, CardType card) {
        mSuit = suit;
        mCard = card;
    }

    bool operator > (const Card& c) const {
        return mCard > c.mCard;
    }

    bool operator == (const Card& c) const {
        return mCard == c.mCard;
    }

    bool operator != (const Card& c) const {
        return !(*this == c);
    }
};

struct Player {
    Card mCards[2];

    Player() {}
};

class Deck {
public:
    Deck() : rng(std::random_device{}()) {
        reset();
    }

    void reset() {
        int index = 0;
        for (int suit = 0; suit < NUM_SUITS; ++suit)
            for (int card = 0; card < NUM_CARDS; ++card)
                mCards[index++].set((SuitType)suit, (CardType)card);
        mNumCards = NUM_SUITS * NUM_CARDS;
    }

    void shuffle() {
        std::shuffle(mCards, mCards + mNumCards, rng);
    }

    Card dealCard() {
        return mCards[--mNumCards];
    }

private:
    Card mCards[NUM_SUITS * NUM_CARDS];
    int mNumCards;
    std::mt19937 rng;
};

struct Hand {
    HandType mType;
    int mValue; // A numerical value representing the hand's strength for easy comparison

    Hand() : mType(MaxHand), mValue(0) {}

    // Simplified hand evaluator using counts
    void evaluate(const std::vector<Card>& cards) {
        int counts[13] = { 0 };
        int suits[4] = { 0 };

        for (const auto& card : cards) {
            counts[card.mCard]++;
            suits[card.mSuit]++;
        }

        // Check for flush
        bool isFlush = false;
        for (int i = 0; i < 4; ++i) {
            if (suits[i] >= 5) {
                isFlush = true;
                break;
            }
        }

        // Check for straight
        int straightHigh = -1;
        int consec = 0;
        for (int i = 12; i >= 0; --i) {
            if (counts[i]) {
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

        if (isFlush && straightHigh != -1) {
            mType = StraightFlush;
            mValue = straightHigh;
            return;
        }

        // Count duplicates
        int four = -1, three = -1;
        std::vector<int> pairs;
        for (int i = 12; i >= 0; --i) {
            if (counts[i] == 4) four = i;
            else if (counts[i] == 3 && three == -1) three = i;
            else if (counts[i] == 2) {
                pairs.push_back(i);
            }
        }

        if (four != -1) {
            mType = FourOfAKind;
            mValue = four;
            return;
        }

        if (three != -1 && !pairs.empty()) {
            mType = FullHouse;
            mValue = three * 13 + pairs[0];
            return;
        }

        if (isFlush) {
            mType = Flush;
            return;
        }

        if (straightHigh != -1) {
            mType = Straight;
            mValue = straightHigh;
            return;
        }

        if (three != -1) {
            mType = ThreeOfAKind;
            mValue = three;
            return;
        }

        if (pairs.size() >= 2) {
            mType = TwoPair;
            mValue = pairs[0] * 13 + pairs[1];
            return;
        }

        if (pairs.size() == 1) {
            mType = OnePair;
            mValue = pairs[0];
            return;
        }

        mType = HighCard;
        for (int i = 12; i >= 0; --i) {
            if (counts[i]) {
                mValue = i;
                break;
            }
        }
    }

    // Compare hands
    bool operator > (const Hand& h) const {
        if (mType != h.mType)
            return mType > h.mType;
        return mValue > h.mValue;
    }
};

struct TableStats {
    unsigned int mHands[NUM_HANDS] = { 0 };
    unsigned int mWinnerCount[MAX_NUM_PLAYERS] = { 0 };
    unsigned int mWinners = 0;
    unsigned int mHandsPlayed = 0;
};

struct Table {
    Deck mDeck;
    Card mBoard[5];
    Player mPlayers[MAX_NUM_PLAYERS];
    Hand mHands[MAX_NUM_PLAYERS];
    TableStats mTableStats;
    int mNumPlayers;
    std::mutex mStatsMutex;

    Table(int numPlayers) : mNumPlayers(numPlayers) {}

    void deal() {
        mDeck.reset();
        mDeck.shuffle();

        for (int i = 0; i < mNumPlayers; ++i) {
            mPlayers[i].mCards[0] = mDeck.dealCard();
            mPlayers[i].mCards[1] = mDeck.dealCard();
        }

        for (int i = 0; i < 5; ++i) {
            mBoard[i] = mDeck.dealCard();
        }
    }

    void evaluate() {
        std::vector<Card> communityCards(mBoard, mBoard + 5);
        for (int i = 0; i < mNumPlayers; ++i) {
            std::vector<Card> playerCards = communityCards;
            playerCards.push_back(mPlayers[i].mCards[0]);
            playerCards.push_back(mPlayers[i].mCards[1]);
            mHands[i].evaluate(playerCards);
        }

        Hand bestHand = mHands[0];
        std::vector<int> winners = { 0 };
        for (int i = 1; i < mNumPlayers; ++i) {
            if (mHands[i] > bestHand) {
                bestHand = mHands[i];
                winners = { i };
            }
            else if (!(bestHand > mHands[i]) && !(mHands[i] > bestHand)) {
                winners.push_back(i);
            }
        }

        // Update stats
        {
            // Since each thread works with its own Table, we can remove the mutex here
            for (int i = 0; i < mNumPlayers; ++i) {
                mTableStats.mHands[mHands[i].mType]++;
                mTableStats.mHandsPlayed++;
            }
            for (int winner : winners) {
                mTableStats.mWinnerCount[winner]++;
                mTableStats.mWinners++;
            }
        }
    }

    void updateGlobalStats(std::vector<unsigned int>& globalHands, std::vector<unsigned int>& globalWinnerCount, std::atomic<unsigned int>& globalWinners, std::atomic<unsigned int>& globalHandsPlayed, std::mutex& globalMutex) {
        std::lock_guard<std::mutex> lock(globalMutex);
        for (int i = 0; i < NUM_HANDS; ++i) {
            globalHands[i] += mTableStats.mHands[i];
            mTableStats.mHands[i] = 0;
        }
        for (int i = 0; i < MAX_NUM_PLAYERS; ++i) {
            globalWinnerCount[i] += mTableStats.mWinnerCount[i];
            mTableStats.mWinnerCount[i] = 0;
        }
        globalWinners += mTableStats.mWinners;
        globalHandsPlayed += mTableStats.mHandsPlayed;
        mTableStats.mWinners = 0;
        mTableStats.mHandsPlayed = 0;
    }
};

void printStats(const std::vector<unsigned int>& globalHands, const std::vector<unsigned int>& globalWinnerCount, unsigned int globalWinners, unsigned int globalHandsPlayed, int numPlayers) {
    for (int i = 0; i < numPlayers; ++i)
        printf("Player %d has won %d times (%.2f percent)\n", i, globalWinnerCount[i], ((float)globalWinnerCount[i] / (float)globalWinners) * 100.f);

    printf("\n");

    float total = 0;
    for (int i = 0; i < NUM_HANDS; ++i) {
        const float percentage = ((float)globalHands[i] / (float)globalHandsPlayed) * 100.f;
        printf("%15s: %8.4f%% hit %d times\n", pHands[i], percentage, globalHands[i]);
        total += percentage;
    }

    printf("\n%.2f%% total | %d hands played\n", total, globalHandsPlayed);
}

void simWorker(Table* table, std::atomic<bool>& stopFlag, std::vector<unsigned int>& globalHands, std::vector<unsigned int>& globalWinnerCount, std::atomic<unsigned int>& globalWinners, std::atomic<unsigned int>& globalHandsPlayed, std::mutex& globalMutex) {
    while (!stopFlag.load()) {
        for (int i = 0; i < 1000; ++i) {
            table->deal();
            table->evaluate();
        }
        table->updateGlobalStats(globalHands, globalWinnerCount, globalWinners, globalHandsPlayed, globalMutex);
        gGameCounter += 1000;
    }
}

int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(10);

    int numPlayers = 9;
    int numThreads = std::thread::hardware_concurrency();

    std::vector<std::unique_ptr<Table>> tables;
    std::vector<std::thread> threads;
    std::atomic<bool> stopFlag(false);

    std::vector<unsigned int> globalHands(NUM_HANDS, 0);
    std::vector<unsigned int> globalWinnerCount(MAX_NUM_PLAYERS, 0);
    std::atomic<unsigned int> globalWinners(0);
    std::atomic<unsigned int> globalHandsPlayed(0);
    std::mutex globalMutex;

    for (int i = 0; i < numThreads; ++i) {
        tables.emplace_back(std::make_unique<Table>(numPlayers));
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(simWorker, tables[i].get(), std::ref(stopFlag), std::ref(globalHands), std::ref(globalWinnerCount), std::ref(globalWinners), std::ref(globalHandsPlayed), std::ref(globalMutex));
    }

    std::this_thread::sleep_for(std::chrono::seconds(10));
    stopFlag = true;

    for (auto& thread : threads) {
        thread.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();

    // Collect any remaining stats
    for (int i = 0; i < numThreads; ++i) {
        tables[i]->updateGlobalStats(globalHands, globalWinnerCount, globalWinners, globalHandsPlayed, globalMutex);
    }

    printStats(globalHands, globalWinnerCount, globalWinners.load(), globalHandsPlayed.load(), numPlayers);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    unsigned long gamesPlayed = gGameCounter.load();

    printf("\nSimulation complete.\n");
    printf("Total execution time: %lld ms\n", duration);
    printf("Games per second: %f\n", gamesPlayed / (duration / 1000.0));

    std::cout << "Press Enter to continue...";
    std::cin.get();

    return 0;
}
