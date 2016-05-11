#include "Imagematting.h"
#include <time.h>

int main()
{
  Imagematting sm;
  clock_t start, finish;
  start = clock();
  sm.loadImage("lowpic.png");
  sm.loadTrimap("lowtrimap.png");
  sm.solveAlpha();
  sm.save("result.jpg");
  finish = clock();
  cout << double(finish - start) / CLOCKS_PER_SEC << endl;

  // // get array preAlpha
  // fstream f("A.txt", ios::in);
  // int N = 200 * 142 + 2;
  // for (int i = 0; i < N; i++)
  //   {
  //     for (int j = 0; j < N; j++)
  //       {
  //         f >> preAlpha[i][j];
  //       }
  //   }

  //	system("pause");
  return 0;
}

