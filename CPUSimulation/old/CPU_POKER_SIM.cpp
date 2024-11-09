
#include <stdio.h>
#include <tchar.h>
#include <time.h>
#include <string.h>
#include "windows.h"
#define NUM_SUITS 4
#define NUM_CARDS 13
#define NUM_HANDS 9
#define MAX_NUM_PLAYERS 9
// #define ASSERT(EXP) if (!(EXP)) { Assert* a = NULL; a->Break(); }
#define ASSERT(EXP)

struct Assert { int i; void Break() { i = 0; ++i; } };

enum HandType { HighCard = 0, OnePair, TwoPair, ThreeOfAKind, Straight, Flush, FullHouse, FourOfAKind, StraightFlush, MaxHand };
enum CardType { c2 = 0, c3, c4, c5, c6, c7, c8, c9, cT, cJ, cQ, cK, cA, MaxCard };
enum SuitType { Club = 0, Diamond, Heart, Spade, MaxSuit };

const char* pHands[NUM_HANDS] = { "HighCard", "OnePair", "TwoPair", "ThreeOfAKind", "Straight", "Flush", "FullHouse", "FourOfAKind", "StraightFlush" };
const char pCards[NUM_CARDS] = { '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A' };
const char pSuits[NUM_SUITS] = { 5, 4, 3, 6 };

unsigned int gHands[NUM_HANDS] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
unsigned int gWinnerCount[MAX_NUM_PLAYERS] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
unsigned int gWinners = 0;
unsigned int gHandsPlayed = 0;
volatile unsigned long	gGameCounter = 0;
bool	gKillThreads = false;

struct Card
{
	CardType mCard;
	SuitType mSuit;

	Card() : mCard(MaxCard), mSuit(MaxSuit)
	{
	}

	void set(SuitType suit, CardType card)
	{
		mSuit = suit;
		mCard = card;
	}

	void print()
	{
		printf("%c%c ", pCards[mCard], pSuits[mSuit]);
	}

	bool operator > (const Card& c) const
	{
		return mCard > c.mCard; // Suit doesn't matter in holdem. Don't consider it.
	}

	bool operator == (const Card& c) const
	{
		return mCard == c.mCard; // Suit doesn't matter in holdem. Don't consider it.
	}

	bool operator != (const Card& c) const
	{
		return !(*this == c);
	}
};

struct Player
{
	int mStack;
	Card mCards[2];
	bool bFolded;

	Player() : mStack(0), bFolded(false)
	{
	}

	void print()
	{
		mCards[0].print();
		mCards[1].print();
	}
};

class Deck
{
	Card mCards[NUM_SUITS * NUM_CARDS];
	int mNumCards;

	void reset()
	{
		for (int suit = 0; suit < NUM_SUITS; ++suit)
			for (int card = 0; card < NUM_CARDS; ++card)
				mCards[(suit*NUM_CARDS) + card].set((SuitType)suit, (CardType)card);

		mNumCards = NUM_SUITS * NUM_CARDS;
	}

public:

	Card remove(int i)
	{
		ASSERT(mNumCards > 0);

		Card removed = mCards[i];

		for (int j = i; j < mNumCards - 1; ++j)
			mCards[j] = mCards[j + 1];

		--mNumCards;
		return removed;
	}

	void print()
	{
		for (int suit = 0; suit < NUM_SUITS; ++suit)
			for (int card = 0; card < NUM_CARDS; ++card)
				mCards[(suit*NUM_CARDS) + card].print();

		printf("Count: %i\n", mNumCards);
	}

	void shuffle()
	{
		Deck tmpDeck;
		int i = 0;

		while (tmpDeck.mNumCards - 1)
		{
			mCards[i] = tmpDeck.remove(rand() % (tmpDeck.mNumCards));
			++i;
		}

		mCards[i] = tmpDeck.remove(0);
		mNumCards = NUM_SUITS * NUM_CARDS;
	}

	Deck()
	{
		reset();
	}
};

struct Hand
{
	HandType mType;
	Card mCards[5];
	int mKind;
	int mKicker;
	int mKicker2;
	int mKicker3;
	bool bFolded;

	// Return 0 for equal hands, 1 if we are stronger than h, -1 if we are weaker than h.
	int isStronger(const Hand& h) const
	{
		ASSERT(mType < MaxHand);

		if (mType != h.mType)
			return mType > h.mType ? 1 : -1;

		ASSERT(mKind != -1 && mType < MaxHand); // Sanity check...

		if (mType == StraightFlush)
		{
			ASSERT(mKind == 0 && mKicker == -1 && mKicker2 == -1 && mKicker3 == -1);
			ASSERT(h.mKind == 0 && h.mKicker == -1 && h.mKicker2 == -1 && h.mKicker3 == -1);

			if (mCards[mKind] == h.mCards[h.mKind])
				return 0; // Equal straight flush (must be on the board by definition)
			else
				return mCards[mKind] > h.mCards[h.mKind] ? 1 : -1; // Evaluate who has the highest straight flush
		}
		else if (mType == FourOfAKind)
		{
			ASSERT(mKind > -1 && mKind < 2 && mKicker > -1 && mKicker2 == -1 && mKicker3 == -1);
			ASSERT(h.mKind > -1 && h.mKind < 2 && h.mKicker > -1 && h.mKicker2 == -1 && h.mKicker3 == -1);

			if (mCards[mKind] == h.mCards[h.mKind]) // Equal 4 of a kind
				if (mCards[mKicker] == h.mCards[h.mKicker]) // Check the kicker...
					return 0; // Equal four of a kinds, same kicker
				else
					return mCards[mKicker] > h.mCards[h.mKicker] ? 1 : -1; // Best kicker wins...
			else
				return mCards[mKind] == h.mCards[h.mKind] ? 1 : -1; // Best four of a kind wins
		}
		else if (mType == Flush)
		{
			ASSERT(mKind == 0 && mKicker == -1 && mKicker2 == -1 && mKicker3 == -1);
			ASSERT(h.mKind == 0 && h.mKicker == -1 && h.mKicker2 == -1 && h.mKicker3 == -1);

			// Both flushes MUST be of the same suit by definition 
			// (only one suit type can ever be held at once since it requires at least 3 board cards to pull off)
			ASSERT(mCards[mKind].mSuit == h.mCards[h.mKind].mSuit);

			// Check who has the best flush...
			if (mCards[0] == h.mCards[0] &&
				mCards[1] == h.mCards[1] &&
				mCards[2] == h.mCards[2] &&
				mCards[3] == h.mCards[3] &&
				mCards[4] == h.mCards[4])
			{
				return 0; // Equal flushes
			}
			else if (mCards[0] > h.mCards[0] ||
				mCards[1] > h.mCards[1] ||
				mCards[2] > h.mCards[2] ||
				mCards[3] > h.mCards[3] ||
				mCards[4] > h.mCards[4])
			{
				return 1; // h is the weak flush
			}
			else
			{
				return -1; // h is the strong flush
			}
		}
		else if (mType == Straight)
		{
			ASSERT(mKind == 0 && mKicker == -1 && mKicker2 == -1 && mKicker3 == -1);
			ASSERT(h.mKind == 0 && h.mKicker == -1 && h.mKicker2 == -1 && h.mKicker3 == -1);

			if (mCards[mKind] == h.mCards[h.mKind])
				return 0; // Equal straight
			else
				return mCards[mKind] > h.mCards[h.mKind] ? 1 : -1; // Highest straight is the strongest
		}
		else if (mType == FullHouse) // Evaluate best full house
		{
			ASSERT(mKind > -1 && mKind < 3 && mKicker > -1 && mKicker2 == -1 && mKicker3 == -1);
			ASSERT(h.mKind > -1 && h.mKind < 3 && h.mKicker > -1 && h.mKicker2 == -1 && h.mKicker3 == -1);

			if (mCards[mKind] == h.mCards[h.mKind]) // Do we have equal trips?
				if (mCards[mKicker] == h.mCards[h.mKicker]) // Do we also have equal pair?
					return 0; // Equal full houses
				else
					return mCards[mKicker] > h.mCards[h.mKicker] ? 1 : -1; // Hand with the greater pair wins
			else
				return mCards[mKind] > h.mCards[h.mKind] ? 1 : -1; // Hand with the greater trips wins
		}
		else if (mType == ThreeOfAKind) // Evaluate best 3 of a kind, consider first and second kickers
		{
			ASSERT(mKind > -1 && mKind < 3 && mKicker > -1 && mKicker2 > -1 && mKicker3 == -1);
			ASSERT(h.mKind > -1 && h.mKind < 3 && h.mKicker > -1 && h.mKicker2 > -1 && h.mKicker3 == -1);

			if (mCards[mKind] == h.mCards[h.mKind]) // Equal trips, go off the kickers
				if (mCards[mKicker] == h.mCards[h.mKicker]) // Same first kicker
					if (mCards[mKicker2] == h.mCards[h.mKicker2]) // Same second kicker
						return 0; // Exact same three of a kind
					else
						return mCards[mKicker2] > h.mCards[h.mKicker2] ? 1 : -1; // Decide who has the best hand based on the second kicker
				else
					return mCards[mKicker] > h.mCards[h.mKicker] ? 1 : -1; // Decide who has the best hand based on the first kicker
			else
				return mCards[mKind] > h.mCards[h.mKind] ? 1 : -1; // Decide who has the best hand based on who has the highest set
		}
		else if (mType == TwoPair)
		{
			ASSERT(mKind > -1 && mKicker > -1 && mKicker2 > -1 && mKicker3 == -1);
			ASSERT(h.mKind > -1 && h.mKicker > -1 && h.mKicker2 > -1 && h.mKicker3 == -1);

			if (mCards[mKind] == h.mCards[h.mKind]) // Same high pair
				if (mCards[mKicker] == h.mCards[h.mKicker]) // Same low pair
					if (mCards[mKicker2] == h.mCards[h.mKicker2]) // Same kicker
						return 0; // Exact same two-pair
					else
						return mCards[mKicker2] > h.mCards[h.mKicker2] ? 1 : -1; // Best kicker wins
				else
					return mCards[mKicker] > h.mCards[h.mKicker] ? 1 : -1; // Best low pair wins
			else
				return mCards[mKind] > h.mCards[h.mKind] ? 1 : -1; // Best high pair wins
		}
		else if (mType == OnePair)
		{
			ASSERT(mKind > -1 && mKicker > -1 && mKicker2 > -1 && mKicker3 > -1);
			ASSERT(h.mKind > -1 && h.mKicker > -1 && h.mKicker2 > -1 && h.mKicker3 > -1);

			if (mCards[mKind] == h.mCards[h.mKind]) // Same pair
				if (mCards[mKicker] == h.mCards[h.mKicker]) // Same kicker
					if (mCards[mKicker2] == h.mCards[h.mKicker2]) // Same second kicker
						if (mCards[mKicker3] == h.mCards[h.mKicker3]) // Same third kicker
							return 0; // Identical hands
						else
							return mCards[mKicker3] > h.mCards[h.mKicker3] ? 1 : -1; // Third kicker wins
					else
						return mCards[mKicker2] > h.mCards[h.mKicker2] ? 1 : -1; // Second kicker wins
				else
					return mCards[mKicker] > h.mCards[h.mKicker] ? 1 : -1; // First kicker wins
			else
				return mCards[mKind] > h.mCards[h.mKind] ? 1 : -1; // Best pair wins
		}
		else if (mType == HighCard)
		{
			ASSERT(mKind == 0 && mKicker == -1 && mKicker2 == -1 && mKicker3 == -1);
			ASSERT(h.mKind == 0 && h.mKicker == -1 && h.mKicker2 == -1 && h.mKicker3 == -1);

			if (mCards[mKind] == h.mCards[h.mKind]) // equal high cards
				if (mCards[1] == h.mCards[1]) // equal first kicker
					if (mCards[2] == h.mCards[2]) // equal second kicker
						if (mCards[3] == h.mCards[3]) // equal third kicker
							if (mCards[4] == h.mCards[4]) // equal forth kicker
								return 0; // Identical hands
							else
								return mCards[4] > h.mCards[4] ? 1 : -1; // Rely on the final kicker
						else
							return mCards[3] > h.mCards[3] ? 1 : -1; // Rely on the third kicker
					else
						return mCards[2] > h.mCards[2] ? 1 : -1; // Rely on the second kicker
				else
					return mCards[1] > h.mCards[1] ? 1 : -1; // Rely on the first kicker
			else
				return mCards[mKind] > h.mCards[h.mKind] ? 1 : -1; // Rely on the high card
		}

		ASSERT(false); // We should never get down to here without already have returned a result...
		return 0;
	}

	void print()
	{
		for (int i = 0; i < 5; ++i)
			mCards[i].print();

		printf(" %s\n", pHands[mType]);
	}

	Hand() : mType(MaxHand), mKind(-1), mKicker(-1), mKicker2(-1), mKicker3(-1), bFolded(false)
	{
	}

	Hand(Card& card1, Card& card2, Card& card3, Card& card4, Card& card5) : mType(MaxHand), mKind(-1), mKicker(-1), mKicker2(-1), mKicker3(-1), bFolded(false)
	{
		// Ensure all hands are initialized
		ASSERT(card1.mCard < MaxCard && card1.mSuit < MaxSuit &&
			card2.mCard < MaxCard && card2.mSuit < MaxSuit &&
			card3.mCard < MaxCard && card3.mSuit < MaxSuit &&
			card4.mCard < MaxCard && card4.mSuit < MaxSuit &&
			card5.mCard < MaxCard && card5.mSuit < MaxSuit);

		// Rank the cards based on value in our mCards array (this helps figure out the hand we have)
		Card tmpCard1, tmpCard2, weakCard1, weakCard2, weakCard3, weakCard4;

		if (card1 > card2) // first two cards
		{
			tmpCard1 = card1;
			weakCard1 = card2;
		}
		else
		{
			tmpCard1 = card2;
			weakCard1 = card1;
		}

		if (card3 > card4) // next two cards
		{
			tmpCard2 = card3;
			weakCard2 = card4;
		}
		else
		{
			tmpCard2 = card4;
			weakCard2 = card3;
		}

		if (tmpCard2 > tmpCard1) // determine the strongest out of the first 4 cards
		{
			weakCard3 = tmpCard1;
			tmpCard1 = tmpCard2;
		}
		else
		{
			weakCard3 = tmpCard2;
		}

		if (tmpCard1 > card5) // check against 5th card
		{
			mCards[0] = tmpCard1;
			weakCard4 = card5;
		}
		else
		{
			mCards[0] = card5;
			weakCard4 = tmpCard1;
		}

		if (weakCard1 > weakCard2) // First 2 weak cards
		{
			tmpCard1 = weakCard1;
			weakCard1 = weakCard2;
		}
		else
		{
			tmpCard1 = weakCard2;
		}

		if (weakCard3 > weakCard4) // Final 2 weak cards
		{
			tmpCard2 = weakCard3;
			weakCard2 = weakCard4;
		}
		else
		{
			tmpCard2 = weakCard4;
			weakCard2 = weakCard3;
		}

		if (tmpCard1 > tmpCard2) // set second strongest card
		{
			mCards[1] = tmpCard1;
			weakCard3 = tmpCard2;
		}
		else
		{
			mCards[1] = tmpCard2;
			weakCard3 = tmpCard1;
		}

		if (weakCard1 > weakCard2) // check the first two weak cards out of the remaining 3
		{
			tmpCard1 = weakCard1;
			weakCard1 = weakCard2;
		}
		else
		{
			tmpCard1 = weakCard2;
		}

		if (tmpCard1 > weakCard3) // check the stronger of the first two weak cards against the 3rd weak card
		{
			mCards[2] = tmpCard1; // third strongest card found

			if (weakCard1 > weakCard3) // slot the final 2 cards into position
			{
				mCards[3] = weakCard1;
				mCards[4] = weakCard3;
			}
			else
			{
				mCards[3] = weakCard3;
				mCards[4] = weakCard1;
			}
		}
		else
		{
			mCards[2] = weakCard3; // third strongest card found

			if (weakCard1 > tmpCard1) // slot the final 2 cards into position
			{
				mCards[3] = weakCard1;
				mCards[4] = tmpCard1;
			}
			else
			{
				mCards[3] = tmpCard1;
				mCards[4] = weakCard1;
			}
		}

		// 1. Check for straight
		if (mCards[0].mCard == mCards[1].mCard + 1 &&
			mCards[1].mCard == mCards[2].mCard + 1 &&
			mCards[2].mCard == mCards[3].mCard + 1 &&
			mCards[3].mCard == mCards[4].mCard + 1)
		{
			mType = Straight;
			mKind = 0;
		}

		// 2. Check for flush
		if (mCards[0].mSuit == mCards[1].mSuit &&
			mCards[1].mSuit == mCards[2].mSuit &&
			mCards[2].mSuit == mCards[3].mSuit &&
			mCards[3].mSuit == mCards[4].mSuit)
		{
			// Check for the straight flush
			if (mType == Straight)
				mType = StraightFlush;
			else
				mType = Flush;

			mKind = 0;
		}

		if (mType == Straight || mType == Flush || mType == StraightFlush)
			return;

		// 3. Check for 4 of a kind
		if (mCards[0] == mCards[1] &&
			mCards[1] == mCards[2] &&
			mCards[2] == mCards[3])
		{
			mType = FourOfAKind;
			mKind = 0;
			mKicker = 4;
			return;
		}
		else if (mCards[1] == mCards[2] &&
			mCards[2] == mCards[3] &&
			mCards[3] == mCards[4])
		{
			mType = FourOfAKind;
			mKind = 1;
			mKicker = 0;
			return;
		}

		// 4. Check for three of a kind and full houses
		if (mCards[0] == mCards[1] &&
			mCards[1] == mCards[2])
		{
			mType = ThreeOfAKind;
			mKind = 0;
			mKicker = 3;

			if (mCards[3] == mCards[4])
			{
				mType = FullHouse;
				return;
			}

			mKicker2 = 4;
			return;
		}
		else if (mCards[1] == mCards[2] &&
			mCards[2] == mCards[3])
		{
			// No possibility of a full house here...
			mType = ThreeOfAKind;
			mKind = 1;
			mKicker = 0;
			mKicker2 = 4;
			return;
		}
		else if (mCards[2] == mCards[3] &&
			mCards[3] == mCards[4])
		{
			mType = ThreeOfAKind;
			mKind = 2;
			mKicker = 0;

			if (mCards[0] == mCards[1])
			{
				mType = FullHouse;
				return;
			}

			mKicker2 = 1;
			return;
		}

		// 5. Check for two-pairs
		if (mCards[0] == mCards[1] &&
			mCards[2] == mCards[3])
		{
			mType = TwoPair;
			mKind = 0;
			mKicker = 2;
			mKicker2 = 4;
			return;
		}
		else if (mCards[1] == mCards[2] &&
			mCards[3] == mCards[4])
		{
			mType = TwoPair;
			mKind = 1;
			mKicker = 3;
			mKicker2 = 0;
			return;
		}
		else if (mCards[0] == mCards[1] &&
			mCards[3] == mCards[4])
		{
			mType = TwoPair;
			mKind = 0;
			mKicker = 3;
			mKicker2 = 2;
			return;
		}

		// 6. Check for pairs
		if (mCards[0] == mCards[1])
		{
			mType = OnePair;
			mKind = 0;
			mKicker = 2;
			mKicker2 = 3;
			mKicker3 = 4;
			return;
		}
		else if (mCards[1] == mCards[2])
		{
			mType = OnePair;
			mKind = 1;
			mKicker = 0;
			mKicker2 = 3;
			mKicker3 = 4;
			return;
		}
		else if (mCards[2] == mCards[3])
		{
			mType = OnePair;
			mKind = 2;
			mKicker = 0;
			mKicker2 = 1;
			mKicker3 = 4;
			return;
		}
		else if (mCards[3] == mCards[4])
		{
			mType = OnePair;
			mKind = 3;
			mKicker = 0;
			mKicker2 = 1;
			mKicker3 = 2;
			return;
		}

		// 7. We have just a high card...
		ASSERT(mType == MaxHand && mKind == -1); // Sanity check
		mType = HighCard;
		mKind = 0;
	}
};

struct TableStats
{
	unsigned int mHands[NUM_HANDS];
	unsigned int mWinnerCount[MAX_NUM_PLAYERS];
	unsigned int mWinners;
	unsigned int mHandsPlayed;

	TableStats()
	{
		reset();
	}

	void reset()
	{
		mWinners = 0;
		mHandsPlayed = 0;
		memset(mHands, 0, sizeof(mHands));
		memset(mWinnerCount, 0, sizeof(mWinnerCount));
	}
};

struct Table
{
	Deck mDeck;
	Card mBoard[5];
	Player mPlayers[MAX_NUM_PLAYERS];
	Hand mHands[MAX_NUM_PLAYERS];
	TableStats mTableStats;
	int mBoardCount;
	int mNumPlayers;
	int mSmallBlind;
	int mPot;
	int mButtonPos;
	bool bWinners[MAX_NUM_PLAYERS];
	CRITICAL_SECTION mCriticalSection;

	Table(int numPlayers, int startingStackSize, int smallBlind) : mBoardCount(0), mNumPlayers(numPlayers), mSmallBlind(smallBlind), mPot(0), mButtonPos(-1)
	{
		ASSERT(numPlayers >= 2 && numPlayers <= MAX_NUM_PLAYERS);
		ASSERT(startingStackSize >= 50);
		ASSERT(smallBlind >= 1);

		InitializeCriticalSectionAndSpinCount(&mCriticalSection, 0x80000400);

		for (int i = 0; i < mNumPlayers; ++i)
		{
			mPlayers[i].mStack = startingStackSize;
			bWinners[i] = false;
		}
	}

	~Table()
	{
		DeleteCriticalSection(&mCriticalSection);
	}

	void resetWinners()
	{
		for (int i = 0; i < mNumPlayers; ++i)
			bWinners[i] = false;
	}

	void deal()
	{
		mBoardCount = 0;

		// update button position
		++mButtonPos;
		if (mButtonPos >= mNumPlayers)
			mButtonPos = 0;

		// post blinds
		if (mButtonPos == mNumPlayers - 2)
		{
			mPlayers[mNumPlayers - 1].mStack -= mSmallBlind;
			mPlayers[0].mStack -= mSmallBlind * 2;
		}
		else if (mButtonPos == mNumPlayers - 1)
		{
			mPlayers[0].mStack -= mSmallBlind;
			mPlayers[1].mStack -= mSmallBlind * 2;
		}
		else
		{
			mPlayers[mButtonPos + 1].mStack -= mSmallBlind;
			mPlayers[mButtonPos + 2].mStack -= mSmallBlind * 2;
		}

		// add blinds to pot
		mPot = mSmallBlind * 3;

		// shuffle and deal
		mDeck.shuffle();
		int count = 0;
		int i = mButtonPos;
		int totalToDeal = mNumPlayers * 2;
		while (count < totalToDeal)
		{
			if (count < mNumPlayers)
				mPlayers[i].mCards[0] = mDeck.remove(0);
			else
				mPlayers[i].mCards[1] = mDeck.remove(0);

			++i;
			if (i >= mNumPlayers)
				i = 0;
			++count;
		}
	}

	void flop()
	{
		ASSERT(mBoardCount == 0);

		Card burn = mDeck.remove(0);

		for (int i = 0; i < 3; ++i)
			mBoard[mBoardCount++] = mDeck.remove(0);
	}

	void turn()
	{
		ASSERT(mBoardCount == 3);
		Card burn = mDeck.remove(0);
		mBoard[mBoardCount++] = mDeck.remove(0);
	}

	void river()
	{
		ASSERT(mBoardCount == 4);
		Card burn = mDeck.remove(0);
		mBoard[mBoardCount++] = mDeck.remove(0);
	}

	void evaluate()
	{
		// Get best hand per player then eval best hand per player against all other hands per player
		Hand tableHand(mBoard[0], mBoard[1], mBoard[2], mBoard[3], mBoard[4]);

		for (int i = 0; i < mNumPlayers; ++i)
		{
			Card& card1 = mPlayers[i].mCards[0];
			Card& card2 = mPlayers[i].mCards[1];

			Hand bestHand = tableHand;

			// Try both cards (10 combinations)
			Hand tmpHand(card1, card2, mBoard[0], mBoard[1], mBoard[2]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[0], mBoard[1], mBoard[3]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[0], mBoard[1], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[0], mBoard[2], mBoard[3]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[0], mBoard[2], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[0], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[1], mBoard[2], mBoard[3]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[1], mBoard[2], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[1], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, card2, mBoard[2], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			// Try card 1 (5 combinations)
			tmpHand = Hand(card1, mBoard[0], mBoard[1], mBoard[2], mBoard[3]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, mBoard[0], mBoard[1], mBoard[2], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, mBoard[0], mBoard[1], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, mBoard[0], mBoard[2], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card1, mBoard[1], mBoard[2], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			// Try card 2 (5 combinations)
			tmpHand = Hand(card2, mBoard[0], mBoard[1], mBoard[2], mBoard[3]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card2, mBoard[0], mBoard[1], mBoard[2], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card2, mBoard[0], mBoard[1], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card2, mBoard[0], mBoard[2], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			tmpHand = Hand(card2, mBoard[1], mBoard[2], mBoard[3], mBoard[4]);
			if (bestHand.isStronger(tmpHand) == -1)
				bestHand = tmpHand;

			// Store the best hand for this player in the hands array
			mHands[i] = bestHand;
		}

		// Compare all hands against all other hands, find the winner, or winners in the case of a split pot
		resetWinners();
		int strongestHandIdx = -1;
		for (int i = 0; i < mNumPlayers; ++i)
		{
			for (int j = 0; j < mNumPlayers; ++j)
			{
				if (i < j)
				{
					int handStrength = mHands[i].isStronger(mHands[j]);
					int currStrongest = -1;

					if (handStrength == -1)
						currStrongest = j;
					else if (handStrength == 1)
						currStrongest = i;

					if (currStrongest != -1) // either i or j is stronger
					{
						if (strongestHandIdx == -1)
						{
							// we have a first strongest hand
							strongestHandIdx = currStrongest;
							bWinners[strongestHandIdx] = true;
						}
						else
						{
							handStrength = mHands[currStrongest].isStronger(mHands[strongestHandIdx]);

							if (handStrength == 0)
							{
								// currStrongest gets to share in the pot with the strongest hand
								bWinners[currStrongest] = true;
							}
							else if (handStrength == 1)
							{
								// currStrongest beats the strongest hand, currStrongest is now the strongest hand
								resetWinners();
								strongestHandIdx = currStrongest;
								bWinners[strongestHandIdx] = true;
							}
						}
					}
					else if (handStrength == 0) // i and j are equal strength
					{
						if (strongestHandIdx == -1)
						{
							// i and j are the first strongest hands, they now share the pot
							strongestHandIdx = i;
							bWinners[i] = true;
							bWinners[j] = true;
							break;
						}

						handStrength = mHands[i].isStronger(mHands[strongestHandIdx]);

						if (handStrength == 0)
						{
							// add i and j to share the pot with the stongest hand
							bWinners[i] = true;
							bWinners[j] = true;
						}
						else if (handStrength == 1)
						{
							// remove the strongest hands from the pot, add i and j who both beat it
							resetWinners();
							strongestHandIdx = i;
							bWinners[i] = true;
							bWinners[j] = true;
						}
					}
				}
			}
		}

		EnterCriticalSection(&mCriticalSection);
		{
			// Accumulate statistics
			for (int i = 0; i < mNumPlayers; ++i)
			{
				++mTableStats.mHands[mHands[i].mType];
				if (bWinners[i])
				{
					++mTableStats.mWinnerCount[i];
					++mTableStats.mWinners;
				}
			}
			mTableStats.mHandsPlayed += mNumPlayers;
		}
		LeaveCriticalSection(&mCriticalSection);
	}

	void print()
	{
		printf("Board: ");

		for (int i = 0; i < mBoardCount; ++i)
			mBoard[i].print();

		printf("\n");

		for (int i = 0; i < mNumPlayers; ++i)
		{
			mPlayers[i].print();
			printf(" - Player %i %s: ", i, bWinners[i] ? "(winner)" : "(loser)");
			mHands[i].print();
		}
	}

	void UpdateGlobalStats()
	{
		EnterCriticalSection(&mCriticalSection);
		{
			for (unsigned int i = 0; i < NUM_HANDS; i++)
				gHands[i] += mTableStats.mHands[i];

			for (unsigned int i = 0; i < MAX_NUM_PLAYERS; i++)
				gWinnerCount[i] += mTableStats.mWinnerCount[i];

			gWinners += mTableStats.mWinners;
			gHandsPlayed += mTableStats.mHandsPlayed;

			unsigned long numGames = (unsigned long)((float)mTableStats.mHandsPlayed / (float)mNumPlayers);
			InterlockedExchangeAdd(&gGameCounter, numGames);

			// Now the stats have been accumulated, reset them...
			mTableStats.reset();
		}
		LeaveCriticalSection(&mCriticalSection);
	}

	static void printStats(unsigned int numPlayers)
	{
		for (unsigned int i = 0; i < numPlayers; ++i)
			printf("Player %d has won %d the times (%.2f percent)\n", i, gWinnerCount[i], ((float)gWinnerCount[i] / (float)gWinners) * 100.f);

		printf("\n");

		float total = 0;
		for (int i = 0; i < NUM_HANDS; ++i)
		{
			const float percentage = ((float)gHands[i] / (float)gHandsPlayed) * 100.f;
			printf("%15s: %8.4f hit %d times\n", pHands[i], percentage, gHands[i]);
			total += percentage;
		}

		printf("\n%.2f percent | %d hands played | %d games played\n", total, gHandsPlayed, gHandsPlayed / numPlayers);
	}
};

static LARGE_INTEGER gTicksPerSecond;

void InitTimer()
{
	QueryPerformanceFrequency(&gTicksPerSecond);
}

double GetTime()
{
	LARGE_INTEGER tick;
	QueryPerformanceCounter(&tick);
	return (double)tick.QuadPart / gTicksPerSecond.QuadPart;
}

DWORD WINAPI SimWorker(void* obj)
{
	Table &t = (Table&)*(Table*)obj;

	static const unsigned int workPerInterval = 300;

	while (1)
	{
		for (int i = 0; i < workPerInterval; ++i)
		{
			t.deal();
			t.flop();
			t.turn();
			t.river();
			t.evaluate();
		}

		if (gKillThreads)
			return 0;

		Sleep(1);
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	srand(time_t(NULL));

	const unsigned int totalThreads = 16;
	const unsigned int totalPlayers = 9;
	const float totalSimulationSeconds = 1.f;

	Table *t[totalThreads];

	for (unsigned int i = 0; i < totalThreads; i++)
	{
		t[i] = new Table(9, 500, 5);
		CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)SimWorker, t[i], 0, 0);
	}

	InitTimer();
	const double startTime = GetTime();
	double timer = startTime;

	while (1)
	{
		double elapsedTime = GetTime() - timer;
		if (elapsedTime > 1.0)
		{
			system("CLS");

			for (unsigned int i = 0; i < totalThreads; i++)
				t[i]->UpdateGlobalStats();

			printf("\n");
			Table::printStats(totalPlayers);
			timer = GetTime();

			float timeSinceStart = (float)timer - (float)startTime;
			printf("\nPlayed %d games this second\n", gGameCounter);

			InterlockedExchangeSubtract(&gGameCounter, gGameCounter);

			if (timeSinceStart >= totalSimulationSeconds)
				break;
		}

		Sleep(1);
	}

	gKillThreads = true;
	Sleep(100); // Give the threads a moment to bail... (This is not fully thread safe, but it will do)

	for (unsigned int i = 0; i < totalThreads; i++)
		delete t[i];

	printf("\nSimulation of %.2f seconds complete", totalSimulationSeconds);

	char c = 0;
	scanf_s("%c", c);

	return 0;
}