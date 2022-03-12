#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <bit>
#include <numeric>

constexpr int SZ = 15;
constexpr float INF = 9999999999999999.9;
constexpr float EPS = 0.000001;

std::vector<std::array<float, SZ>> calculateIntraFrame(float cost[SZ][SZ]) {
  // USED Fragments; First; Last;
  std::array<std::array<std::vector<float>, SZ>, SZ> costSoFar;
  for (int a = 0; a < SZ; a++)
    for (int b = 0; b < SZ; b++) costSoFar[a][b].resize(1 << SZ, INF);

  for (int a = 0; a < SZ; a++)  /// each beg separately
  {
    // init a-b
    for (int b = 0; b < SZ; b++) {
      if (a == b) continue;
      costSoFar[a][b][1 << a | 1 << b] = cost[a][b];
    }
    // extend by 1 step
    //

    for (int already = 2; already < SZ; already++) {
      for (int c = 0; c < SZ; c++)  // new end
      {
        if (c == a) continue;
        // all possib prev state:
        for (int b = 0; b < SZ; b++) {
          if (b == a or b == c) continue;
          for (size_t X = 3; X < (1 << SZ);
               X++)  // what if that was the external loop?
          {
            if (std::popcount(X) != already) continue;
            if ((X & (1 << a)) == 0 or (X & (1 << b)) == 0 or
                (X & (1 << c)) != 0)
              continue;
            auto nX = X | (1 << c);
            costSoFar[a][c][nX] =
                std::min(costSoFar[a][c][nX], costSoFar[a][b][X] + cost[b][c]);
          }
        }
      }
    }
    printf(".");
    fflush(stdout);
  }

  std::vector<std::array<float, SZ>> res(SZ);
  for (int i = 0; i < SZ; i++)
    for (int j = 0; j < SZ; j++) {
      if (i == j) continue;
      // result in costSoFar[0x7FFF]
      res[i][j] = costSoFar[i][j][0x7FFF];
    }
  printf("\nintraCalc done. Cost[0][14] = %f\n", res[0][14]);
  return res;
}

std::vector<int> calculatePermut(float cost[SZ][SZ], int a, int B,
                                 int resOffset) {
  std::vector<int> res;
  std::array<std::array<std::vector<float>, SZ>, SZ> costSoFar;
  std::array<std::array<std::vector<int>, SZ>, SZ> prev;

  for (int i = 0; i < SZ; i++)
    for (int j = 0; j < SZ; j++) {
      costSoFar[i][j].resize(1 << SZ, INF);
      prev[i][j].resize(1 << SZ, -1);
    }

  for (int b = 0; b < SZ; b++) {
    if (a == b) continue;
    costSoFar[a][b][1 << a | 1 << b] = cost[a][b];
  }

  for (int already = 2; already < SZ; already++) {
    for (int c = 0; c < SZ; c++)  // new end
    {
      if (c == a) continue;
      // all possib prev state:
      for (int b = 0; b < SZ; b++) {
        if (b == a or b == c) continue;
        for (size_t X = 3; X < (1 << SZ);
             X++)  // what if that was the external loop?
        {
          if (std::popcount(X) != already) continue;
          if ((X & (1 << a)) == 0 or (X & (1 << b)) == 0 or (X & (1 << c)) != 0)
            continue;
          auto nX = X | (1 << c);
          auto newCost = costSoFar[a][b][X] + cost[b][c];
          if (costSoFar[a][c][nX] > newCost) {
            costSoFar[a][c][nX] = newCost;
            prev[a][c][nX] = b;
          }
        }
      }
    }
  }
  printf(".");
  fflush(stdout);

  res.resize(SZ);
  res[0] = a;
  int now = B;
  res[SZ - 1] = B;
  size_t X = 0x7FFF;
  for (int i = SZ - 2; i > 0; i--) {
    res[i] = prev[a][res[i + 1]][X];
    X ^= 1 << res[i + 1];
  }

  for (int i = 0; i < SZ; i++) res[i] += resOffset;

  return res;
}

int main() {
  int N, M;
  scanf("%i %i", &N, &M);
  float cost[N][N];
  for (int n = 0; n < N; n++) cost[n][n] = 0.0;
  while (M--) {
    int A, B;
    scanf("%i %i", &A, &B);
    float tmp;
    scanf("%f", &tmp);
    if (A < N && B < N) cost[A][B] = tmp;
  }
  int numFrames = N / SZ;

  std::vector<std::vector<std::array<float, SZ>>> stepMinCosts;
  for (int step = 0; step < numFrames; step++) {
    float intraFrCost[SZ][SZ];
    int base = step * SZ;
    for (int i = 0; i < SZ; i++)
      for (int j = 0; j < SZ; j++) intraFrCost[i][j] = cost[base + i][base + j];
    stepMinCosts.push_back(calculateIntraFrame(intraFrCost));
    printf("Step %i/%i done.\n", step + 1, numFrames);
  }

  // find optimal costs
  int optimBeg[numFrames];
  int optimEnd[numFrames];

  //
  float minCostAfterFrame[numFrames][SZ];
  int prevEnd[numFrames][SZ];
  int thatBeg[numFrames][SZ];
  for (int i = 0; i < numFrames; i++)
    for (int a = 0; a < SZ; a++) minCostAfterFrame[i][a] = INF;
  // init
  for (int b = 0; b < SZ; b++) {
    for (int a = 0; a < SZ; a++) {
      if (a == b) continue;
      auto newCost = stepMinCosts[0][a][b];
      if (minCostAfterFrame[0][b] > newCost) {
        minCostAfterFrame[0][b] = newCost;
        prevEnd[0][b] = -1;
        thatBeg[0][b] = a;
      }
    }
  }

  for (int s = 1; s < numFrames; s++) {
    for (int b = 0; b < SZ; b++) {
      for (int a = 0; a < SZ; a++) {
        if (a == b) continue;
        for (int e = 0; e < SZ; e++) {
          auto newCost = minCostAfterFrame[s - 1][e] +
                         cost[SZ * (s - 1) + e][SZ * s + a] +
                         stepMinCosts[s][a][b];
          if (minCostAfterFrame[s][b] > newCost) {
            minCostAfterFrame[s][b] = newCost;
            prevEnd[s][b] = e;
            thatBeg[s][b] = a;
          }
        }
      }
    }
  }
  int bestEnd = 0;
  for (int b = 0; b < SZ; b++) {
    if (minCostAfterFrame[numFrames - 1][b] <
        minCostAfterFrame[numFrames - 1][bestEnd])
      bestEnd = b;
  }
  printf("Total min cost: %f\n", minCostAfterFrame[numFrames - 1][bestEnd]);
  int now = bestEnd;
  optimEnd[numFrames - 1] = now;
  optimBeg[numFrames - 1] = thatBeg[numFrames - 1][now];
  for (int s = numFrames - 2; s >= 0; s--) {
    now = prevEnd[s + 1][now];
    optimEnd[s] = now;
    optimBeg[s] = thatBeg[s][now];
  }

  for (int i = 0; i < numFrames; i++)
    printf("frame %i. Optim beg: %i end %i\n", i, optimBeg[i], optimEnd[i]);

  //	reconstruct the path
  std::vector<int> path;
  for (int step = 0; step < numFrames; step++) {
    float intraFrCost[SZ][SZ];
    int base = step * SZ;
    for (int i = 0; i < SZ; i++)
      for (int j = 0; j < SZ; j++) intraFrCost[i][j] = cost[base + i][base + j];

    auto frameP =
        calculatePermut(intraFrCost, optimBeg[step], optimEnd[step], base);
    path.insert(path.end(), frameP.begin(), frameP.end());
  }
  printf("PERMUT:\n");
  for (auto a : path) printf("%i, ", a);
  printf("\n");
}
