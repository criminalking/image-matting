#include "Imagematting.h"
#include <time.h>

int main()
{
        Imagematting sm;
	clock_t start, finish;
	start = clock();
	sm.loadImage("abc.png");
	sm.loadTrimap("abc2.png");
	sm.solveAlpha();
	sm.save("result.jpg");
	finish = clock();
	cout << double(finish - start) / CLOCKS_PER_SEC << endl;

//	system("pause");
	return 0;
}

