#include "Imagematting.h"

Imagematting::Imagematting()
{
	bsize = 0;
	fsize = 0;
	usize = 0;
	covarienceOfMat = cvCreateMat(3, 3, CV_64FC1); // n * n (channel * channel) 
	avgOfMat = cvCreateMat(1, 3, CV_64FC1); // 1 * n (1 * channel)
}

Imagematting::~Imagematting()
{
	cvReleaseImage(&img);
	cvReleaseImage(&trimap);
	cvReleaseImage(&matte);
	cvReleaseMat(&covarienceOfMat);
	cvReleaseMat(&avgOfMat);
}

void Imagematting::loadImage(char * filename)
{
	img = cvLoadImage(filename, -1);
	if (!img)
	{
		cout << "Loading Image Failed!" << endl;
		exit(-1);
	}
	height = img->height;
	width = img->width;
	step = img->widthStep;
	channels = img->nChannels;
	N = height * width + 2;
	data = (uchar *)img->imageData;
	tri = new int*[height];
	preAlpha = new double*[height];
	for (int i = 0; i < height; ++i) 
	{
		tri[i] = new int[width];
		preAlpha[i] = new double[width];
	}
	W1.resize(N, N);
	W2.resize(N, N);
	W3.resize(N, N);
	L.resize(N, N);
	I.resize(N, N);
	Alpha = VectorXd::Zero(N, 1);
	G = VectorXd::Zero(N, 1);

	matte = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
}

void Imagematting::loadTrimap(char * filename)
{
	trimap = cvLoadImage(filename, -1);
	g_step = trimap->widthStep;
	if (!trimap)
	{
		cout << "Loading Trimap Failed!" << endl;
		exit(-1);
	}
	//get uszie, bsiez, fsize
	uchar *udata = (uchar *)trimap->imageData;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int gray = udata[i * g_step + j];
			if (gray < 5) // c is a background     ////////////////////可以修改使用mat的push_back！！！！！
			{
				bsize++;
				tri[i][j] = 0;
				preAlpha[i][j] = 0;
			}
			else if (gray > 250) // c is a foreground
			{
				fsize++;
				tri[i][j] = 1;
				preAlpha[i][j] = 1;
			}
			else // c is a unknown pixel
			{
				usize++;
				tri[i][j] = 2;
			}
		}
	}
	bmat = Mat::zeros(bsize, 5, CV_32FC1);
	fmat = Mat::zeros(fsize, 5, CV_32FC1);
	umat = Mat::zeros(usize, 5, CV_32FC1);
	allmat = Mat::zeros(height * width, 5, CV_32FC1);
	createMat(); // create three mats which kd-trees and knnsearch need
	cout << "loadtrimap ok " << endl;
}

void Imagematting::addInMat(Mat &mat, int n, int i, int j, int b, int g, int r)
{
	mat.at<int>(n, 0) = i;
	mat.at<int>(n, 1) = j;
	mat.at<int>(n, 2) = b;
	mat.at<int>(n, 3) = g;
	mat.at<int>(n, 4) = r;
}

void Imagematting::createMat()
{
	uchar *udata = (uchar *)trimap->imageData;

	int bn = 0, fn = 0, un = 0, n = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int bc = data[i * step + j * channels];
			int gc = data[i * step + j * channels + 1];
			int rc = data[i * step + j * channels + 2];
			int gray = udata[i * g_step + j];
			if (gray < 5) // gray is background (gray == 4!!!!!!!!!!!!!!)
			{
				// add (x,y,r,g,b) to bmat
				addInMat(bmat, bn, i, j, bc, gc, rc);
				addInMat(allmat, n, i, j, bc, gc, rc);
				bn++;
				n++;
			}
			else if (gray > 250) // gray is foreground (gray == 252!!!!!!!!!!)
			{
				// add (x,y,r,g,b) to fmat
				addInMat(fmat, fn, i, j, bc, gc, rc);
				addInMat(allmat, n, i, j, bc, gc, rc);
				fn++;
				n++;
			}
			else // otherwise
			{
				// add (x,y,r,g,b) to umat
				addInMat(umat, un, i, j, bc, gc, rc);
				addInMat(allmat, n, i, j, bc, gc, rc);
				un++;
				n++;
			}
		}
	}
	cout << "createmat ok " << endl;
}

void Imagematting::findKnearest()// build 2 KD-trees
{
	flann::Index tree1(bmat, flann::KDTreeIndexParams(4));// create kd-tree for background
	tree1.knnSearch(umat, bresult.indices, bresult.dists, K); // search kd-tree

	flann::Index tree2(fmat, flann::KDTreeIndexParams(4));// create kd-tree for foreground
	tree2.knnSearch(umat, fresult.indices, fresult.dists, K); // search kd-tree

	flann::Index tree3(allmat, flann::KDTreeIndexParams(4));// create kd-tree for all pixels
	tree3.knnSearch(allmat, w3result.indices, w3result.dists, K); // search kd-tree

	//FileStorage fs("K2.xml", FileStorage::WRITE); // save the data
	//fs << "bindices" << bresult.indices;
	//fs << "findices" << fresult.indices;
	//fs << "w3indices" << w3result.indices;
	//fs.release();

	cout << "get kdtree ok " << endl;
}
 
double Imagematting::geteveryAlpha(int c, int f, int b) //f is the fth-nearest pixel of C, b is the bth-nearest pixel of C
{
	int findex = fresult.indices.at<int>(c, f);
	int bindex = bresult.indices.at<int>(c, b);

	double alpha = ((umat.at<int>(c, 2) - bmat.at<int>(bindex, 2)) * (fmat.at<int>(findex, 2) - bmat.at<int>(bindex, 2)) +
		(umat.at<int>(c, 3) - bmat.at<int>(bindex, 3)) * (fmat.at<int>(findex, 3) - bmat.at<int>(bindex, 3)) +
		(umat.at<int>(c, 4) - bmat.at<int>(bindex, 4)) * (fmat.at<int>(findex, 4) - bmat.at<int>(bindex, 4)))
		/ (((fmat.at<int>(findex, 2) - bmat.at<int>(bindex, 2)) * (fmat.at<int>(findex, 2) - bmat.at<int>(bindex, 2)) +
		(fmat.at<int>(findex, 3) - bmat.at<int>(bindex, 3)) * (fmat.at<int>(findex, 3) - bmat.at<int>(bindex, 3)) +
		(fmat.at<int>(findex, 4) - bmat.at<int>(bindex, 4)) * (fmat.at<int>(findex, 4) - bmat.at<int>(bindex, 4))) + 0.0000001);
	return min(1.0, max(0.0, alpha));
}

double Imagematting::getRd(int c, int f, int b) //f is the fth-nearest pixel of C, b is the bth-nearest pixel of C
{
	double alpha = geteveryAlpha(c, f, b);

	int findex = fresult.indices.at<int>(c, f);
	int bindex = bresult.indices.at<int>(c, b);

	double result = sqrt(((umat.at<int>(c, 2) - alpha * fmat.at<int>(findex, 2) - (1 - alpha) * bmat.at<int>(bindex, 2)) * (umat.at<int>(c, 2) - alpha *  fmat.at<int>(findex, 2) - (1 - alpha) * bmat.at<int>(bindex, 2)) +
		(umat.at<int>(c, 3) - alpha * fmat.at<int>(findex, 3) - (1 - alpha) * bmat.at<int>(bindex, 3)) * (umat.at<int>(c, 3) - alpha * fmat.at<int>(findex, 3) - (1 - alpha) * bmat.at<int>(bindex, 3)) +
		(umat.at<int>(c, 4) - alpha * fmat.at<int>(findex, 4) - (1 - alpha) * bmat.at<int>(bindex, 4)) * (umat.at<int>(c, 4) - alpha * fmat.at<int>(findex, 4) - (1 - alpha) * bmat.at<int>(bindex, 4))) /
		(((fmat.at<int>(findex, 2) - bmat.at<int>(bindex, 2)) * (fmat.at<int>(findex, 2) - bmat.at<int>(bindex, 2)) +
		(fmat.at<int>(findex, 3) - bmat.at<int>(bindex, 3)) * (fmat.at<int>(findex, 3) - bmat.at<int>(bindex, 3)) +
		(fmat.at<int>(findex, 4) - bmat.at<int>(bindex, 4)) * (fmat.at<int>(findex, 4) - bmat.at<int>(bindex, 4))) + 0.0000001));
//	return result / 255.0;  //?????????????????????
	return result;
}

void Imagematting::getD() // correspond to umat
{
	// get db, df of every pixel
	dB = new int[usize];
	dF = new int[usize];
	for (int i = 0; i < usize; i++)
	{
		// calculate d2
		int bindex = bresult.indices.at<int>(i, 0); // get the nearest background of C
		dB[i] = 0;
		for (int j = 2; j < 5; j++)
			dB[i] += (umat.at<int>(i, j) - bmat.at<int>(bindex, j)) * (umat.at<int>(i, j) - bmat.at<int>(bindex, j));

		int findex = fresult.indices.at<int>(i, 0); // get the nearest foreground of C
		dF[i] = 0;
		for (int j = 2; j < 5; j++)
			dF[i] += (umat.at<int>(i, j) - fmat.at<int>(findex, j)) * (umat.at<int>(i, j) - fmat.at<int>(findex, j));
	}
	cout << "getD ok " << endl;
}

double Imagematting::getW(int c, int fb, bool flag) // flag == 1, f; flag == 0, b; fb is the fbth-nearest fore- or background pixel of C
{
	double w;

	if (flag == 0) // b
	{
		int index = bresult.indices.at<int>(c, fb);
		w = exp(-((umat.at<int>(c, 2) - bmat.at<int>(index, 2)) * (umat.at<int>(c, 2) - bmat.at<int>(index, 2)) +
			(umat.at<int>(c, 3) - bmat.at<int>(index, 3)) * (umat.at<int>(c, 3) - bmat.at<int>(index, 3)) +
			(umat.at<int>(c, 4) - bmat.at<int>(index, 4)) * (umat.at<int>(c, 4) - bmat.at<int>(index, 4))) / (dB[c] + 0.0000001));
	}
	else // f
	{
		int index = fresult.indices.at<int>(c, fb);
		w = exp(-((umat.at<int>(c, 2) - fmat.at<int>(index, 2)) * (umat.at<int>(c, 2) - fmat.at<int>(index, 2)) +
			(umat.at<int>(c, 3) - fmat.at<int>(index, 3)) * (umat.at<int>(c, 3) - fmat.at<int>(index, 3)) +
			(umat.at<int>(c, 4) - fmat.at<int>(index, 4)) * (umat.at<int>(c, 4) - fmat.at<int>(index, 4))) / (dF[c] + 0.0000001));
	}
	return w;
}

double Imagematting::getConfidence(int c, int f, int b)  //f is the fth-nearest foreground pixel of C, b is the bth-nearest background pixel of C
{		
	double confi;
	confi = exp(-(getRd(c, f, b) * getRd(c, f, b) * getW(c, f, 1) * getW(c, b, 0)) / (sigma * sigma)); //////////////getW 部分有重复可简化
	return confi;
}

void Imagematting::getPreAlpha()
{
	getD(); 
	// calculate confidence of every unknown pixel
	for (int i = 0; i < usize; i++)
	{
		int Ci = umat.at<int>(i, 0);
		int Cj = umat.at<int>(i, 1);
		// choose three pairs which have the biggest confidence of every unknown pixel and mean their alpha as the predicted alpha --- fAlpha 
		double alpha1 = 0, alpha2 = 0, alpha3 = 0; 
		double confi1 = 0, confi2 = 0, confi3 = 0; // 1 > 2 > 3
		for (int f = 0; f < K; f++)
		{
			for (int b = 0; b < K; b++)
			{
				double confi = getConfidence(i, f, b);
				if (confi > confi1)
				{
					alpha3 = alpha2; confi3 = confi2;
					alpha2 = alpha1; confi2 = confi1;
					alpha1 = geteveryAlpha(i, f, b); confi1 = confi;
				}
				else if (confi < confi1 && confi > confi2)
				{
					alpha3 = alpha2; confi3 = confi2;
					alpha2 = geteveryAlpha(i, f, b); confi2 = confi;
				}
				else if (confi < confi2 && confi > confi3)
				{
					alpha3 = geteveryAlpha(i, f, b); confi3 = confi;
				}
			}
		}
		//// get fAlpha
		preAlpha[Ci][Cj] = (alpha1 + alpha2 + alpha3) / 3.0;
	}

	// save preAlpha in file "Array.xml"
	//FileStorage fs("Array.xml", FileStorage::WRITE); // save the data
	//fs << "preAlpha" << preAlpha;
	//fs.release();

	// save in ".txt"
	fstream f("a2.txt", ios::out);
	if (!f) cout << "Error!" << endl;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			f << preAlpha[i][j] << endl;
		}
	}
	f.close();
	cout << "getPreAlpha ok " << endl;
}

void   Imagematting::getCovarianceMatrix(int x, int y) // x & y are the middle points in one 3*3 window (require: not the edge) 
{
	int M = winSize * winSize; // the number of pixel in the window
	int N = 3; // channels
	CvMat* mat = cvCreateMat(M, N, CV_64FC1);
	
	// set the original mat
	for (int i = 0; i < N; i++) /////////////////////////////////进行改进！！！！因为该矩阵是对称的
	{
		for (int j = 0; j < N; j++)
		{
			cvmSet(mat, i * 3 + j, 0, data[(x + i - 1) * step + (y + j - 1) * channels]);
			cvmSet(mat, i * 3 + j, 1, data[(x + i - 1) * step + (y + j - 1) * channels + 1]);
			cvmSet(mat, i * 3 + j, 2, data[(x + i - 1) * step + (y + j - 1) * channels + 2]);
			//CV_MAT_ELEM(*mat, double, i * 3 + j, 0) = data[(x + i - 1) * step + (y + j - 1) * channels];
			//CV_MAT_ELEM(*mat, double, i * 3 + j, 1) = data[(x + i - 1) * step + (y + j - 1) * channels + 1];
			//CV_MAT_ELEM(*mat, double, i * 3 + j, 2) = data[(x + i - 1) * step + (y + j - 1) * channels + 2];
	//		cout << CV_MAT_ELEM(*mat, double, i * 3 + j, 0) << "," << CV_MAT_ELEM(*mat, double, i * 3 + j, 1) << "," << CV_MAT_ELEM(*mat, double, i * 3 + j, 2) << endl;
		}
	}
	
	// compute the covariance Matrix
	cvZero(covarienceOfMat);
	cvZero(avgOfMat);
	cvCalcCovarMatrix((const void **)&mat, 1, covarienceOfMat, avgOfMat, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	if (M > 1) cvConvertScale(covarienceOfMat, covarienceOfMat, 1.0 / (M - 1)); // normalization
	cvReleaseMat(&mat);
}

void   Imagematting::getCiCj(CvMat *mat, int i, int j)
{
	cvmSet(mat, 0, 0, data[i * step + j * channels]);
	cvmSet(mat, 0, 1, data[i * step + j * channels + 1]);
	cvmSet(mat, 0, 2, data[i * step + j * channels + 2]);
}

void   Imagematting::getWeight1() // get data term W(i, F) & W(i, B) 
{
	std::vector<T> triplets;
	triplets.push_back(T(0, 0, gamma)); //W1(0, 0) = gamma; W1(0, 1) = 0;
	triplets.push_back(T(1, 1, gamma)); //W1(1, 1) = gamma; W1(1, 0) = 0; 
	for (int i = 2; i < N; i++) // 0, 1 are two virtue nodes
	{
		int x = (i - 2) / width;
		int y = (i - 2) - x * width;
		if (tri[x][y] == 0 || tri[x][y] == 1)
		{
			triplets.push_back(T(i, 0, gamma * tri[x][y]));
			triplets.push_back(T(i, 1, gamma * (1 - tri[x][y])));
			triplets.push_back(T(i, i, gamma * tri[x][y] + gamma * (1 - tri[x][y]))); // add to L(i, i)
		}
		else
		{
			triplets.push_back(T(i, 0, gamma * preAlpha[x][y]));
			triplets.push_back(T(i, 1, gamma * (1 - preAlpha[x][y])));
			triplets.push_back(T(i, i, gamma * preAlpha[x][y] + gamma * (1 - preAlpha[x][y]))); // add to L(i, i)
		}
	}
	W1.setFromTriplets(triplets.begin(), triplets.end());
	cout << "getWeight1 ok" << endl;
}

void   Imagematting::getWeight2() // get local smooth term Wlap(ij)
{
	std::vector<T> triplets;
	double w;
	CvMat* reverseMat = cvCreateMat(3, 3, CV_64FC1); // save the matrix which needs to be reverse
	CvMat* CiMat = cvCreateMat(1, 3, CV_64FC1); // for Ci
	CvMat* CjMat = cvCreateMat(1, 3, CV_64FC1); // for Cj
	CvMat* stoMat = cvCreateMat(1, 3, CV_64FC1); // store mat
	CvMat* stoMat2 = cvCreateMat(3, 1, CV_64FC1); // store mat
	CvMat* result = cvCreateMat(1, 1, CV_64FC1); // the result mat
	CvMat* IdenMat = cvCreateMat(3, 3, CV_64FC1); // the identity mat
	cvSetIdentity(IdenMat); // get I
	for (int i = 0; i < usize; i++) 
	{
		w = 0;
		int Ci = umat.at<int>(i, 0);
		int Cj = umat.at<int>(i, 1);
		getCiCj(CiMat, Ci, Cj);
		// get 4 weights: up, down, left, right
		//left (Ci, Cj - 1)
		if (Cj - 1 >= 0)
		{
			getCiCj(CjMat, Ci, Cj - 1);
			for (int m = 0; m < 3; m++)
			{
				for (int n = 0; n < 2; n++)
				{
					if (Ci + m - 1 > 0 && Ci + m - 1 < height - 1 && Cj + n - 1 > 0 && Cj + n - 1 < width - 1)
					{
						getCovarianceMatrix(Ci + m - 1, Cj + n - 1); // get covarienceOfMat & avgOfMat in 3*3 window
						cvAddWeighted(covarienceOfMat, 1, IdenMat, RC / 9, 0, reverseMat); // covarienceOfMat + RC/9 * I
						cvInvert(reverseMat, reverseMat, CV_SVD); // Mat = (Mat)-1
						cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
						cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + RC/9 * I)-1
						cvSub(CjMat, avgOfMat, avgOfMat); // Cj - uk
						cvTranspose(avgOfMat, stoMat2); // T(Cj - uk)const
						cvMatMul(stoMat, stoMat2, result); // (Ci - uk) * (covarienceOfMat + RC/9 * I)-1 * T(Cj - uk)
						w = w + 1 + cvmGet(result, 0, 0);
					}
				}
			}
			//	Pixel[i].space[0] = delta / 9 * w;
			triplets.push_back(T(Ci * width + Cj + 2, Ci * width + Cj + 1, delta / 9 * w));
			triplets.push_back(T(Ci * width + Cj + 2, Ci * width + Cj + 2, delta / 9 * w)); // add to L(i, i)
		}
		w = 0;
		//right (Ci, Cj + 1)
		if (Cj + 1 < width)
		{
			getCiCj(CjMat, Ci, Cj + 1);
			for (int m = 0; m < 3; m++)
			{
				for (int n = 0; n < 2; n++)
				{
					if (Ci + m - 1 > 0 && Ci + m - 1 < height - 1 && Cj + n > 0 && Cj + n < width - 1)
					{
						getCovarianceMatrix(Ci + m - 1, Cj + n); // get covarienceOfMat & avgOfMat in 3*3 window
						cvAddWeighted(covarienceOfMat, 1, IdenMat, RC / 9, 0, reverseMat); // covarienceOfMat + RC/9 * I
						cvInvert(reverseMat, reverseMat, CV_SVD); // Mat = (Mat)-1
						cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
						cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + RC/9 * I)-1
						cvSub(CjMat, avgOfMat, avgOfMat); // Cj - uk
						cvTranspose(avgOfMat, stoMat2); // T(Cj - uk)const
						cvMatMul(stoMat, stoMat2, result); // (Ci - uk) * (covarienceOfMat + RC/9 * I)-1 * T(Cj - uk)
						w = w + 1 + cvmGet(result, 0, 0);
					}
				}
			}
			//	Pixel[i].space[0] = delta / 9 * w;
			triplets.push_back(T(Ci * width + Cj + 2, Ci * width + Cj + 3, delta / 9 * w));
			triplets.push_back(T(Ci * width + Cj + 2, Ci * width + Cj + 2, delta / 9 * w)); // add to L(i, i)
		}
		w = 0;
		//up (Ci - 1, Cj)
		if (Ci - 1 >= 0)
		{
			getCiCj(CjMat, Ci - 1, Cj);
			for (int m = 0; m < 2; m++)
			{
				for (int n = 0; n < 3; n++)
				{
					if (Ci + m - 1 > 0 && Ci + m - 1 < height - 1 && Cj + n - 1 > 0 && Cj + n - 1 < width - 1)
					{
						getCovarianceMatrix(Ci + m - 1, Cj + n - 1); // get covarienceOfMat & avgOfMat in 3*3 window
						cvAddWeighted(covarienceOfMat, 1, IdenMat, RC / 9, 0, reverseMat); // covarienceOfMat + RC/9 * I
						cvInvert(reverseMat, reverseMat, CV_SVD); // Mat = (Mat)-1
						cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
						cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + RC/9 * I)-1
						cvSub(CjMat, avgOfMat, avgOfMat); // Cj - uk
						cvTranspose(avgOfMat, stoMat2); // T(Cj - uk)const
						cvMatMul(stoMat, stoMat2, result); // (Ci - uk) * (covarienceOfMat + RC/9 * I)-1 * T(Cj - uk)
						w = w + 1 + cvmGet(result, 0, 0);
					}
				}
			}
			//	Pixel[i].space[0] = delta / 9 * w;
			triplets.push_back(T(Ci * width + Cj + 2, (Ci - 1) * width + Cj + 2, delta / 9 * w));
			triplets.push_back(T(Ci * width + Cj + 2, Ci * width + Cj + 2, delta / 9 * w)); // add to L(i, i)
		}
		w = 0;
		//down (Ci + 1, Cj)
		if (Ci + 1 < height)
		{
			getCiCj(CjMat, Ci + 1, Cj);
			for (int m = 0; m < 2; m++)
			{
				for (int n = 0; n < 3; n++)
				{
					if (Ci + m > 0 && Ci + m < height - 1 && Cj + n - 1 > 0 && Cj + n - 1 < width - 1)
					{
						getCovarianceMatrix(Ci + m, Cj + n - 1); // get covarienceOfMat & avgOfMat in 3*3 window
						cvAddWeighted(covarienceOfMat, 1, IdenMat, RC / 9, 0, reverseMat); // covarienceOfMat + RC/9 * I
						cvInvert(reverseMat, reverseMat, CV_SVD); // Mat = (Mat)-1
						cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
						cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + RC/9 * I)-1
						cvSub(CjMat, avgOfMat, avgOfMat); // Cj - uk
						cvTranspose(avgOfMat, stoMat2); // T(Cj - uk)const
						cvMatMul(stoMat, stoMat2, result); // (Ci - uk) * (covarienceOfMat + RC/9 * I)-1 * T(Cj - uk)
						w = w + 1 + cvmGet(result, 0, 0);
					}
				}
			}
			triplets.push_back(T(Ci * width + Cj + 2, (Ci + 1) * width + Cj + 2, delta / 9 * w));
			triplets.push_back(T(Ci * width + Cj + 2, Ci * width + Cj + 2, delta / 9 * w)); // add to L(i, i)
		}
	}
	W2.setFromTriplets(triplets.begin(), triplets.end());
	cvReleaseMat(&reverseMat);
	cvReleaseMat(&CiMat);
	cvReleaseMat(&CjMat);
	cvReleaseMat(&stoMat);
	cvReleaseMat(&stoMat2);
	cvReleaseMat(&result);
	cvReleaseMat(&IdenMat);
	cout << "getWeight2 ok" << endl;
}

void   Imagematting::getWeight3() // get unlocal smooth term Wlle(ij)
{
	std::vector<T> triplets;
	for (int i = 2; i < N; i++)
	{
		VectorXd Y(5, 1);
		MatrixXd X(K, 5);
		VectorXd W(K, 1);
		for (int j = 0; j < 5; j++) Y(j) = allmat.at<int>(i - 2, j);
		for (int j = 0; j < K; j++) // search the K-nearest pixels in RGBXY
		{
			int index = w3result.indices.at<int>(i - 2, j); // index + 2 is the index in N ///////////////////////下面要修改
			X.row(j) << allmat.at<int>(index, 0), allmat.at<int>(index, 1), allmat.at<int>(index, 2), allmat.at<int>(index, 3), allmat.at<int>(index, 4); 
		}
		// W = (X * T(X))^-1 * X * Y
	//	W = (X * X.transpose()).inverse() * X * Y;
		W = (X.transpose()).colPivHouseholderQr().solve(Y);
		
		for (int j = 0; j < K; j++) // search the K-nearest pixels in RGBXY
		{
			int Knearest = w3result.indices.at<int>(i - 2, j) + 2; // Knearest is the index of K-nearest neighbors of i
			triplets.push_back(T(i, Knearest, W(j)));
			triplets.push_back(T(i, i, W(j))); // add to L(i, i)
		}
	}
	W3.setFromTriplets(triplets.begin(), triplets.end());
	cout << "getWeight3 ok" << endl;
}

void   Imagematting::getG()
{
	// Gi is set to 1 if i belongs to foreground and 0 otherwise
	for (int i = 2; i < N; i++) // 0, 1 are two virtue nodes
	{
		int x = (i - 2) / width;
		int y = (i - 2) - x * width;
		if (tri[x][y] == 1)
			G(i) = 1;
	}
	cout << "getG ok" << endl;
}

void   Imagematting::getI()
{
	// Iii = 1000 if i belongs to S(S = f + b + u(preAlpha > 0.85))
	std::vector<T> triplets;
	for (int i = 2; i < N; i++) // let two virtue nodes be 0 (I00 = I11 = 0)
	{
		int x = (i - 2) / width;
		int y = (i - 2) - x * width;
		if (tri[x][y] == 0 || tri[x][y] == 1 || preAlpha[x][y] > 0.85)
			triplets.push_back(T(i, i, 1000));
	}
	I.setFromTriplets(triplets.begin(), triplets.end());
	cout << "getI ok" << endl;
}

void   Imagematting::getL() // get unlocal smooth term Wlle(ij)
{
	// L = -W1 - W2 - W3 (L(i, i) is already computed.)
	getWeight1();
	getWeight2();
	getWeight3();
	L = -W1 - W2 - W3;
	cout << "getL ok" << endl;
}

void   Imagematting::getFinalAlpha()
{
	getI();
	getG();
	getL();
	// (I + T(L) * L) * alpha = I * G
	SpMat A = I + (L.transpose() * L);
	A.prune(0.0, 1e-10);
	saveMarket(A, "A2.mtx");
	
	VectorXd b = I * G;

	clock_t start, finish;
	start = clock();

	//ConjugateGradient<SpMat> cg(A.transpose() * A); // use CG method
	//Alpha.setZero();
	//Alpha = cg.solve(A.transpose() * b);

	SimplicialLDLT<SpMat> ldlt(A);
	Alpha = ldlt.solve(b);

	finish = clock();
	cout << double(finish - start) / CLOCKS_PER_SEC << endl;

	//let Alpha between 0-1
//	Alpha = Alpha.cwiseAbs(); // abs
	saveMarket(Alpha, "Alpha.mtx");
	cout << "getFinalAlpha ok" << endl;
}

void Imagematting::showMatte()
{
	int h = matte->height;
	int w = matte->width;
	uchar *udata = (uchar *)matte->imageData;
	//for (int i = 2; i < N; i++)
	//{
	//	int x = (i - 2) / width;
	//	int y = (i - 2) - x * width;
	//	udata[x * step + y] = int(Alpha(i) * 255);
	//	cout << int(udata[x * step + y]) << endl;
	//}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = i * width + j + 2;
			udata[i * g_step + j] = int(Alpha[index] * 255);
		}
	}
}

void Imagematting::save(char * filename)
{
	cvSaveImage(filename, matte);
}

void   Imagematting::solveAlpha()
{
	clock_t start, finish;
	
	findKnearest(); // get K nearest backgrounds(indices + dists) 
	
	//// read four mats in "Kdatas.xml"
	//FileStorage fs("K1.xml", FileStorage::READ);
	//fs["findices"] >> fresult.indices;
	//fs["bindices"] >> bresult.indices;
	//fs["w3indices"] >> w3result.indices;
	//fs.release(); 

	getPreAlpha(); // get predicted alpha of every pixel

	//// get array preAlpha
	//fstream f("a1.txt", ios::in);
	//for (int i = 0; i < height; i++)
	//{
	//	for (int j = 0; j < width; j++)
	//	{
	//		f >> preAlpha[i][j];
	//	}
	//}

	getFinalAlpha();

	showMatte();
}