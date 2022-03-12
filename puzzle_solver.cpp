#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <bit>

constexpr int SZ=15;
constexpr float INF = 9999999999999999.9;
constexpr float EPS = 0.000001;

std::vector<std::array<float, SZ>> calculateIntraFrame(float cost[SZ][SZ])
{
	// USED Fragments; First; Last; 
	float costSoFar[1<<SZ][SZ][SZ]; // dont need the amount - count(used) is one.
	for(int x=0; x<1<<SZ; x++)
	for(int a=0; a<SZ; a++)
	for(int b=0; b<SZ; b++)
		costSoFar[x][a][b] = INF;


	for(int a=0; a<SZ; a++) /// each beg separately
	{
	// init a-b
		for(int b=0; b<SZ; b++)
		{
			if(a==b) continue;
			costSoFar[1<<a | 1 << b][a][b] = cost[a][b];
		}
		//extend by 1 step
		//

		for(int already = 2; already < SZ; already++)
		{
			for(int c=0; c<SZ; c++) // new end
			{
				if(c==a) continue;
				//all possib prev state:
				for(int b=0; b<SZ; b++)
				{
					if(b==a or b==c)continue;
					for(size_t X=3; X<(1<<SZ); X++) // what if that was the external loop?
					{
						if(std::popcount(X) != already) continue;
						if((X & (1<<a)) == 0 or (X & (1<<b)) == 0 or
						   (X & (1<<c)) != 0) continue;
						auto nX = X|(1<<c);
						costSoFar[nX][a][c] = std::min(costSoFar[nX][a][c], costSoFar[X][a][b] + cost[b][c]);
					}
				}
			}
		}
		printf(".");
		fflush(stdout);
	}

	std::vector<std::array<float, SZ>> res(SZ);
	for(int i=0; i<SZ; i++)
	for(int j=0; j<SZ; j++)
	{
		if(i==j) continue;
		//result in costSoFar[0x7FFF]
		res[i][j] = costSoFar[0x7FFF][i][j];
	}
	printf("\nintraCalc done. Cost[0][14] = %f\n", res[0][14]);
	return res;
}

std::vector<int> calculatePermut(float cost[SZ][SZ], int a, int B, int resOffset)
{
	std::vector<int> res;

        float costSoFar[1<<SZ][SZ][SZ]; // dont need the amount - count(used) is one.
        int prev[1<<SZ][SZ][SZ];
        for(int x=0; x<1<<SZ; x++)
        for(int a=0; a<SZ; a++)
        for(int b=0; b<SZ; b++)
	{
                costSoFar[x][a][b] = INF;
		prev[x][a][b] = -1;
	}


        for(int b=0; b<SZ; b++)
        {
                 if(a==b) continue;
                 costSoFar[1<<a | 1 << b][a][b] = cost[a][b];
        }

        for(int already = 2; already < SZ; already++)
        {
                for(int c=0; c<SZ; c++) // new end
                {
                         if(c==a) continue;
                         //all possib prev state:
                         for(int b=0; b<SZ; b++)
                         {
                                 if(b==a or b==c)continue;
                                 for(size_t X=3; X<(1<<SZ); X++) // what if that was the external loop?
                                 {
                                         if(std::popcount(X) != already) continue;
                                         if((X & (1<<a)) == 0 or (X & (1<<b)) == 0 or
                                            (X & (1<<c)) != 0) continue;
                                         auto nX = X|(1<<c);
					 auto newCost = costSoFar[X][a][b] + cost[b][c];
					 if(costSoFar[nX][a][c] > newCost)
					 {
                                         	costSoFar[nX][a][c] = newCost;
						prev[nX][a][c] = b;
					 }
                                 }
                          }
                 }
        }
        printf(".");
        fflush(stdout);

	res.resize(SZ);
	res[0]=a;
	int now = B;
	res[SZ-1]=B;
	size_t X=0x7FFF;
	for(int i=SZ-2; i>0; i--)
	{
		res[i] = prev[X][a][res[i+1]];
		X ^= 1<<res[i+1];
	}

   
	for(int i=0; i<SZ; i++)
		res[i] += resOffset;

	return res;
}

int main()
{
	int N, M;
	scanf("%i %i", &N, &M);
	float cost[N][N];
	for(int n=0; n<N; n++)
		cost[n][n] = 0.0;
	while(M--)
	{
		int A, B;
		scanf("%i %i", &A, &B);
		float tmp;
		scanf("%f",  &tmp);
	       	if(A<N && B<N)
			cost[A][B] = tmp;

	}
	int numFrames = N/SZ;
	float minCostFromTo_inFrame[SZ][SZ][numFrames];
	for(int i=0; i<SZ; i++)
		for(int j=0; j<SZ; j++)
			for(int f=0; f<numFrames; f++)
				minCostFromTo_inFrame[i][j][f] = 0.0;

	std::vector<std::vector<std::array<float, SZ>>> stepMinCosts;
	for(int step=0; step<numFrames; step++)
	{
		float intraFrCost[SZ][SZ];
		int base = step*SZ;
		for(int i=0; i<SZ; i++)
			for(int j=0; j<SZ; j++)
				intraFrCost[i][j] = cost[base+i][base+j];
		auto res = calculateIntraFrame(intraFrCost);
		stepMinCosts.push_back(res);
		printf("Step %i/%i done.\n", step+1, numFrames);

	}

	// find optimal costs
	int optimBeg[numFrames];
	int optimEnd[numFrames];


	//
	float minCostAfterFrame[numFrames][SZ];
	int prevEnd[numFrames][SZ];
	int thatBeg[numFrames][SZ];
	for(int i=0; i<numFrames; i++)
		for(int a=0; a<SZ; a++)
			minCostAfterFrame[i][a] = INF;
	//init
	for(int b=0; b<SZ; b++)
	{
		for(int a=0; a<SZ; a++)
		{
			if(a==b) continue;
			auto newCost = stepMinCosts[0][a][b];
			if(minCostAfterFrame[0][b] > newCost)
			{
				minCostAfterFrame[0][b] = newCost;
				prevEnd[0][b] = -1;
				thatBeg[0][b] = a;
			}
		}
	}

	for(int s=1; s<numFrames; s++)
	{
		for(int b=0; b<SZ; b++)
		{
			for(int a=0; a<SZ; a++)
			{
				if(a==b)continue;
				for(int e=0; e<SZ; e++)
				{
					auto newCost = minCostAfterFrame[s-1][e] + cost[SZ*(s-1)+e][SZ*s+a] + stepMinCosts[s][a][b];
					if(minCostAfterFrame[s][b] > newCost)
					{
						minCostAfterFrame[s][b] = newCost;
						prevEnd[s][b] = e;
						thatBeg[s][b] = a;
					}
				}
			}
		}
	}
	int bestEnd = 0;
	for(int b=0; b<SZ; b++)
	{
		if(minCostAfterFrame[numFrames-1][b] < minCostAfterFrame[numFrames-1][bestEnd])
			bestEnd = b;
	}
	printf("Total min cost: %f\n", minCostAfterFrame[numFrames-1][bestEnd]);
	int now = bestEnd;
	optimEnd[numFrames-1] = now;
	optimBeg[numFrames-1] = thatBeg[numFrames-1][now];
	for(int s=numFrames-2; s>=0; s--)
	{
		now = prevEnd[s+1][now];
		optimEnd[s] = now;
		optimBeg[s] = thatBeg[s][now];
	}


	for(int i=0; i<numFrames; i++)
		printf("frame %i. Optim beg: %i end %i\n", i, optimBeg[i], optimEnd[i]);


	//DO THEFINDINGDDS
	//
	//
	//quick hack for now: just smallest cost
/*	for(int i=0; i<numFrames; i++)
	{
		int ba=0;
		int bb=0;
		float minCost = INF;
		for(int a=0; a<SZ; a++)
			for(int b=0; b<SZ; b++)
			{
				if(a==b) continue;
				if(stepMinCosts[i][a][b] < minCost)
				{
					minCost = stepMinCosts[i][a][b];
					ba=a;
					bb=b;
				}
			}
		optimBeg[i]=ba;
		optimEnd[i]=bb;
		printf("frame %i. Optim beg: %i end %i cost: %f\n", i, ba, bb, minCost);
	}*/
//	reconstruct the path
	std::vector<int> path;
	for(int step=0; step<numFrames; step++)
	{
                float intraFrCost[SZ][SZ];
                int base = step*SZ;
                for(int i=0; i<SZ; i++)
                        for(int j=0; j<SZ; j++)
                                intraFrCost[i][j] = cost[base+i][base+j];

		auto frameP = calculatePermut(intraFrCost, optimBeg[step], optimEnd[step], base);
		path.insert(path.end(), frameP.begin(), frameP.end());
	}
	printf("PERMUT:\n");
	for(auto a: path)
		printf("%i, ", a);
	printf("\n");

}
