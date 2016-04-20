#include "imagematting.h"
#include <time.h>

int main()
{

	//while(n--){
	Imagematting sm;
	clock_t start, finish;
	start = clock();
	sm.loadImage("abc.png");
	sm.loadTrimap("abc2.png");
	sm.solveAlpha();
	finish = clock();
	cout << double(finish - start) / CLOCKS_PER_SEC << endl;
	//sum += double(finish - start) / CLOCKS_PER_SEC;
	//}
	//cout << sum / 15 << endl;


	// test!!!!!!!!!!!!!!
	//int M = 30000;
	//SpMat A(M, M);
	//SpMat AtA(M, M);
	//std::vector<Triplet<double> > triplets;
	//for (int i = 0; i < M; i++)
	//{
	//	for (int j = 0; j < 150; j++) // only 1500 values are non-zero
	//	{
	//		triplets.push_back(T(i, j, rand() + 0.783));
	//	}
	//}
	//A.setFromTriplets(triplets.begin(), triplets.end());
	//VectorXd b = VectorXd::Random(M);
	//VectorXd AtB = A.transpose() * b;
	//AtA = A.transpose() * A;
	//VectorXd x(A.cols()); 
	//start = clock();
	//ConjugateGradient<SpMat> cg(AtA);
	//x.setZero();
	//x = cg.solve(AtB);
	//finish = clock();
	//cout << double(finish - start) / CLOCKS_PER_SEC << endl;




	system("pause");
	return 0;
}
