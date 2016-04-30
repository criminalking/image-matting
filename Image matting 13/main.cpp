#include "Imagematting.h"
#include <time.h>

int main()
{

	//while(n--){
	Imagematting sm;
	clock_t start, finish;
	start = clock();
	sm.loadImage("aaa.jpg");
	sm.loadTrimap("bbb.jpg");
	sm.solveAlpha();
	sm.save("result.jpg");
	finish = clock();
	cout << double(finish - start) / CLOCKS_PER_SEC << endl;
//	sum += double(finish - start) / CLOCKS_PER_SEC;
//	}
//	cout << sum / 15 << endl;

	system("pause");
	return 0;
}
