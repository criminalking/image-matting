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
        confidence = new double*[height];
	for (int i = 0; i < height; ++i)
	{
		tri[i] = new int[width];
		preAlpha[i] = new double[width];
                confidence[i] = new double[width];
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
			if (gray < 5) // c is a background     ////////////////////¿ÉÒÔÐÞ¸ÄÊ¹ÓÃmatµÄpush_back£¡£¡£¡£¡£¡
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
	bmat = Mat::zeros(bsize, 2, CV_32FC1);
	fmat = Mat::zeros(fsize, 2, CV_32FC1);
	umat = Mat::zeros(usize, 2, CV_32FC1);
	allmat = Mat::zeros(height * width, 5, CV_32FC1);
	createMat(); // create three mats which kd-trees and knnsearch need
	cout << "loadtrimap ok " << endl;
}



void Imagematting::addInMat(Mat &mat, int n, int i, int j, int b, int g, int r)
{
  AT(mat, n, 0) = i;
  AT(mat, n, 1) = j;
  AT(mat, n, 2) = b;
  AT(mat, n, 3) = g;
  AT(mat, n, 4) = r;
}

void Imagematting::addInMat(Mat &mat, int n, int x, int y)
{
  AT(mat, n, 0) = x;
  AT(mat, n, 1) = y;
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
				// add (x,y,r,g,b) to allmat
				addInMat(bmat, bn, i, j);
				addInMat(allmat, n, i, j, bc, gc, rc);
				bn++;
				n++;
			}
			else if (gray > 250) // gray is foreground (gray == 252!!!!!!!!!!)
			{
				// add (x,y,r,g,b) to allmat
				addInMat(fmat, fn, i, j);
				addInMat(allmat, n, i, j, bc, gc, rc);
				fn++;
				n++;
			}
			else // otherwise
			{
				// add (x,y,r,g,b) to allmat
				addInMat(umat, un, i, j);
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
	tree3.knnSearch(allmat, allresult.indices, allresult.dists, K); // search kd-tree

	FileStorage fs("K2.xml", FileStorage::WRITE); // save the data
	fs << "bindices" << bresult.indices;
	fs << "findices" << fresult.indices;
	fs << "allindices" << allresult.indices;
	fs.release();

	cout << "get kdtree ok " << endl;
}

int  Imagematting::BC(Mat &mat, int index)
{
  return data[mat.at<int>(index, 0) * step + mat.at<int>(index, 1) * channels];
}

int  Imagematting::GC(Mat &mat, int index)
{
  return data[mat.at<int>(index, 0) * step + mat.at<int>(index, 1) * channels + 1];
}

int  Imagematting::RC(Mat &mat, int index)
{
  return data[mat.at<int>(index, 0) * step + mat.at<int>(index, 1) * channels + 2];
}

double Imagematting::geteveryAlpha(int c, int f, int b) //f is the fth-nearest pixel of C, b is the bth-nearest pixel of C
{
  int findex = AT(fresult.indices, c, f);
  int bindex = AT(bresult.indices, c, b);

  // double alpha = ((AT(umat, c, 2) - AT(bmat, bindex, 2)) * (AT(fmat, findex, 2) - AT(bmat, bindex, 2)) +
  //                 (AT(umat, c, 3) - AT(bmat, bindex, 3)) * (AT(fmat, findex, 3) - AT(bmat, bindex, 3)) +
  //                 (AT(umat, c, 4) - AT(bmat, bindex, 4)) * (AT(fmat, findex, 4) - AT(bmat, bindex, 4)))
  //   / ((AT(fmat, findex, 2) - AT(bmat, bindex, 2)) * (AT(fmat, findex, 2) - AT(bmat, bindex, 2)) +
  //       	(AT(fmat, findex, 3) - AT(bmat, bindex, 3)) * (AT(fmat, findex, 3) - AT(bmat, bindex, 3)) +
  //       	(AT(fmat, findex, 4) - AT(bmat, bindex, 4)) * (AT(fmat, findex, 4) - AT(bmat, bindex, 4)) + 0.0000001);
  double alpha = ((BC(umat, c) - BC(bmat, bindex)) * (BC(fmat, findex) - BC(bmat, bindex)) +
                  (GC(umat, c) - GC(bmat, bindex)) * (GC(fmat, findex) - GC(bmat, bindex)) +
                  (RC(umat, c) - RC(bmat, bindex)) * (RC(fmat, findex) - RC(bmat, bindex)))
    / ((BC(fmat, findex) - BC(bmat, bindex)) * (BC(fmat, findex) - BC(bmat, bindex)) +
        	(GC(fmat, findex) - GC(bmat, bindex)) * (GC(fmat, findex) - GC(bmat, bindex)) +
        	(RC(fmat, findex) - RC(bmat, bindex)) * (RC(fmat, findex) - RC(bmat, bindex)) + 0.0000001);
	return min(1.0, max(0.0, alpha));
}

double Imagematting::getRd(int c, int f, int b) //f is the fth-nearest pixel of C, b is the bth-nearest pixel of C
{
	double alpha = geteveryAlpha(c, f, b);
        int findex = AT(fresult.indices, c, f);
        int bindex = AT(bresult.indices, c, b);

	double result = sqrt(((BC(umat, c) - alpha * BC(fmat, findex) - (1 - alpha) * BC(bmat, bindex)) * (BC(umat, c) - alpha * BC(fmat, findex) - (1 - alpha) * BC(bmat, bindex)) +
		(GC(umat, c) - alpha * GC(fmat, findex) - (1 - alpha) * GC(bmat, bindex)) * (GC(umat, c) - alpha * GC(fmat, findex) - (1 - alpha) * GC(bmat, bindex)) +
		(RC(umat, c) - alpha * RC(fmat, findex) - (1 - alpha) * RC(bmat, bindex)) * (RC(umat, c) - alpha * RC(fmat, findex) - (1 - alpha) * RC(bmat, bindex))) /
		(((BC(fmat, findex) - BC(bmat, bindex)) * (BC(fmat, findex) - BC(bmat, bindex)) +
		(GC(fmat, findex) - GC(bmat, bindex)) * (GC(fmat, findex) - GC(bmat, bindex)) +
		(RC(fmat, findex) - RC(bmat, bindex)) * (RC(fmat, findex) - RC(bmat, bindex))) + 0.0000001));
	//return result / 255.0;  //?????????????????????
        return result;
}

void Imagematting::getD() // correspond to umat
{
	// get db, df of every pixel
	dB = new int[usize];
	dF = new int[usize];
        double min1 = 0, min2 = 0;
	for (int i = 0; i < usize; i++)
	{
		// calculate d2
          dB[i] = 200000;
          dF[i] = 200000;
          for (int k = 0; k < K; k++)
            {
		int bindex = AT(bresult.indices, i, k); // get the nearest background of C
                min1 = (BC(umat, i) - BC(bmat, bindex)) * (BC(umat, i) - BC(bmat, bindex)) + (GC(umat, i) - GC(bmat, bindex)) * (GC(umat, i) - GC(bmat, bindex)) + (RC(umat, i) - RC(bmat, bindex)) * (RC(umat, i) - RC(bmat, bindex));
                if (min1 < dB[i]) dB[i] = min1;

		int findex = AT(fresult.indices, i, k); // get the nearest foreground of C
                min2 = (BC(umat, i) - BC(fmat, findex)) * (BC(umat, i) - BC(fmat, findex)) + (GC(umat, i) - GC(fmat, findex)) * (GC(umat, i) - GC(fmat, findex)) + (RC(umat, i) - RC(fmat, findex)) * (RC(umat, i) - RC(fmat, findex));
                if (min2 < dF[i]) dF[i] = min2;
            }
	}
	cout << "getD ok " << endl;
}

double Imagematting::getW(int c, int fb, bool flag) // flag == 1, f; flag == 0, b; fb is the fbth-nearest fore- or background pixel of C
{
	double w;

	if (flag == 0) // b
	{
		int index = AT(bresult.indices, c, fb);
		w = exp(-((BC(umat, c) - BC(bmat, index)) * (BC(umat, c) - BC(bmat, index)) +
			(GC(umat, c) - GC(bmat, index)) * (GC(umat, c) - GC(bmat, index)) +
			(RC(umat, c) - RC(bmat, index)) * (RC(umat, c) - RC(bmat, index))) / (dB[c] + 0.0000001));
	}
	else // f
	{
                int index = AT(fresult.indices, c, fb);
		w = exp(-((BC(umat, c) - BC(fmat, index)) * (BC(umat, c) - BC(fmat, index)) +
			(GC(umat, c) - GC(fmat, index)) * (GC(umat, c) - GC(fmat, index)) +
			(RC(umat, c) - RC(fmat, index)) * (RC(umat, c) - RC(fmat, index))) / (dF[c] + 0.0000001));
	}
	return w;
}

double Imagematting::getConfidence(int c, int f, int b)  //f is the fth-nearest foreground pixel of C, b is the bth-nearest background pixel of C
{
	double confi;
	confi = exp(-(getRd(c, f, b) * getRd(c, f, b) * getW(c, f, 1) * getW(c, b, 0)) / (sigma * sigma)); //////////////getW ²¿·ÖÓÐÖØ¸´¿É¼ò»¯
	return confi;
}

void Imagematting::getPreAlpha()
{
	getD();
	// calculate confidence of every unknown pixel
	for (int i = 0; i < usize; i++)
	{
		int Ci = AT(umat, i, 0);
		int Cj = AT(umat, i, 1);
		// choose three pairs which have the biggest confidence of every unknown pixel and mean their alphas as the predicted alpha --- fAlpha
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
                confidence[Ci][Cj] = (confi1 + confi2 + confi3) / 3.0;
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

void Imagematting::TEST(SpMat A)
{
  for (int k=0; k < A.outerSize(); ++k)
{
    for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
    {
      //  std::cout << "(" << it.row() << ","; // row index
      //    std::cout << it.col() << ")\t"; // col index (here it is equal to k)
      cout<< it.value() << "   ";
    }
}
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
			triplets.push_back(T(i, 0, -gamma * tri[x][y]));
			triplets.push_back(T(i, 1, -gamma * (1 - tri[x][y])));
                        triplets.push_back(T(i, i, gamma)); // add to L(i, i)
		}
		else
		{
			triplets.push_back(T(i, 0, -gamma * preAlpha[x][y]));
			triplets.push_back(T(i, 1, -gamma * (1 - preAlpha[x][y])));
                        triplets.push_back(T(i, i, gamma)); // add to L(i, i)
		}
	}
	W1.setFromTriplets(triplets.begin(), triplets.end());
        W1.prune(0.0);
	cout << "getWeight1 ok" << endl;
}

void   Imagematting::testgetWeight2() // get local smooth term Wlap(ij)
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
		int Ci = AT(umat, i, 0);
		int Cj = AT(umat, i, 1);
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
						cvAddWeighted(covarienceOfMat, 1, IdenMat, REG / 9, 0, reverseMat); // covarienceOfMat + REG/9 * I
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
			triplets.push_back(T(Ci * width + Cj + 2, Ci * width + Cj + 1,-abs(delta / 9 * w)));
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
						cvAddWeighted(covarienceOfMat, 1, IdenMat, REG / 9, 0, reverseMat); // covarienceOfMat + RC/9 * I
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
			triplets.push_back(T(Ci * width + Cj + 2, Ci * width + Cj + 3,-abs(delta / 9 * w)));
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
						cvAddWeighted(covarienceOfMat, 1, IdenMat, REG / 9, 0, reverseMat); // covarienceOfMat + RC/9 * I
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
			triplets.push_back(T(Ci * width + Cj + 2, (Ci - 1) * width + Cj + 2, -abs(delta / 9 * w)));
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
						cvAddWeighted(covarienceOfMat, 1, IdenMat, REG / 9, 0, reverseMat); // covarienceOfMat + REG/9 * I
						cvInvert(reverseMat, reverseMat, CV_SVD); // Mat = (Mat)-1
						cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
						cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1
						cvSub(CjMat, avgOfMat, avgOfMat); // Cj - uk
						cvTranspose(avgOfMat, stoMat2); // T(Cj - uk)const
						cvMatMul(stoMat, stoMat2, result); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1 * T(Cj - uk)
						w = w + 1 + cvmGet(result, 0, 0);
					}
				}
			}
			triplets.push_back(T(Ci * width + Cj + 2, (Ci + 1) * width + Cj + 2, -abs(delta / 9 * w)));
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
        W2.prune(0.0);
	cout << "getWeight2 ok" << endl;
}


void   Imagematting::getCovarianceMatrix(int x, int y) // x & y are the middle points in one 3*3 window (require: not the edge)
{
	int M = winSize * winSize; // the number of pixel in the window
	int n = 3; // channels
	CvMat* mat = cvCreateMat(M, n, CV_64FC1);

	// set the original mat
	for (int i = 0; i < winSize; i++)
	{
		for (int j = 0; j < winSize; j++)
		{
			cvmSet(mat, i * 3 + j, 0, data[(x + i - 1) * step + (y + j - 1) * channels]);
			cvmSet(mat, i * 3 + j, 1, data[(x + i - 1) * step + (y + j - 1) * channels + 1]);
			cvmSet(mat, i * 3 + j, 2, data[(x + i - 1) * step + (y + j - 1) * channels + 2]);
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


void   Imagematting::getWeight2() // get local smooth term Wlap(ij)
{
  // get Ci, Cj, uk, sigmaK of every 3*3 window
  	std::vector<T> triplets;
	double w;
	CvMat* reverseMat = cvCreateMat(winSize, winSize, CV_64FC1); // save the matrix which needs to be reverse
	CvMat* CiMat = cvCreateMat(1, 3, CV_64FC1); // for Ci
	CvMat* CjMat = cvCreateMat(1, 3, CV_64FC1); // for Cj
	CvMat* stoMat = cvCreateMat(1, winSize, CV_64FC1); // store mat
	CvMat* stoMat2 = cvCreateMat(1, winSize, CV_64FC1); // store mat
        CvMat* stoMattr = cvCreateMat(winSize, 1, CV_64FC1); // store mat
	CvMat* result = cvCreateMat(1, 1, CV_64FC1); // the result mat
	CvMat* IdenMat = cvCreateMat(winSize, winSize, CV_64FC1); // the identity mat
	cvSetIdentity(IdenMat); // get I
        for (int i = 0; i < height - 2; i++) // (i, j) is the up left point of every window
          {
            for (int j = 0; j < width - 2; j++)
              {
                //compute uk and sigmaK(update covarienceOfMat and avgOfMat)
                getCovarianceMatrix(i + 1, j + 1); // send the center point
                cvAddWeighted(covarienceOfMat, 1, IdenMat, REG / 9, 0, reverseMat); // covarienceOfMat + REG/9 * I
                cvInvert(reverseMat, reverseMat, CV_SVD_SYM); // Mat = (Mat)-1

                // Wij = Wji
                // get W2 in horizontal direction
                for (int l1 = 0; l1 < winSize; l1++)
                  {
                    for (int l2 = 0; l2 < winSize - 1; l2++)
                      {
                        // (l1, l2) & (l1, l2 + 1)
                         getCiCj(CiMat, l1, l2);
                         getCiCj(CjMat, l1, l2 + 1);
                         cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
                         cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1
                         cvSub(CjMat, avgOfMat, stoMat2); // Cj - uk
                         cvTranspose(stoMat2, stoMattr); // T(Cj - uk)const
                         cvMatMul(stoMat, stoMattr, result); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1 * T(Cj - uk)
                         w = 1 + cvmGet(result, 0, 0);
                         triplets.push_back(T(l1 * width + l2 + 2, l1 * width + (l2 + 1) + 2, -delta / 9 * w)); // Wij
                         triplets.push_back(T(l1 * width + (l2 + 1) + 2, l1 * width + l2 + 2, -delta / 9 * w)); // Wji
                         triplets.push_back(T(l1 * width + l2 + 2, l1 * width + l2 + 2, delta / 9 * w)); // add to L(i, i)
                         triplets.push_back(T(l1 * width + (l2 + 1) + 2, l1 * width + (l2 + 1) + 2, delta / 9 * w)); // add to L(i, i)
                      }
                  }

                // get W2 in vertical direction
                for (int l1 = 0; l1 < winSize - 1; l1++)
                  {
                    for (int l2 = 0; l2 < winSize; l2++)
                      {
                        // (l1, l2) & (l1 + 1, l2)
                         getCiCj(CiMat, l1, l2);
                         getCiCj(CjMat, l1 + 1, l2);
                         cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
                         cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1
                         cvSub(CjMat, avgOfMat, stoMat2); // Cj - uk
                         cvTranspose(stoMat2, stoMattr); // T(Cj - uk)const
                         cvMatMul(stoMat, stoMattr, result); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1 * T(Cj - uk)
                         w = 1 + cvmGet(result, 0, 0);
                         triplets.push_back(T(l1 * width + l2 + 2, (l1 + 1) * width + l2 + 2, -delta / 9 * w)); // Wij
                         triplets.push_back(T((l1 + 1) * width + l2 + 2, l1 * width + l2 + 2, -delta / 9 * w)); // Wji
                         triplets.push_back(T(l1 * width + l2 + 2, l1 * width + l2 + 2, delta / 9 * w)); // add to L(i, i)
                         triplets.push_back(T((l1 + 1) * width + l2 + 2, (l1 + 1) * width + l2 + 2, delta / 9 * w)); // add to L(i, i)
                      }
                  }
              }
          }
        W2.setFromTriplets(triplets.begin(), triplets.end());
	cvReleaseMat(&reverseMat);
	cvReleaseMat(&CiMat);
	cvReleaseMat(&CjMat);
	cvReleaseMat(&stoMat);
	cvReleaseMat(&stoMat2);
        cvReleaseMat(&stoMattr);
	cvReleaseMat(&result);
	cvReleaseMat(&IdenMat);
        W2.prune(0.0);
	cout << "getWeight2 ok" << endl;
}

void   Imagematting::getWeight3() // get unlocal smooth term Wlle(ij), use LLE
{
	std::vector<T> triplets;
        double del = 0.1 * 0.1 / K; // add a very small number to eigenvalue of XtX in order to insurance invertibility of XtX

	for (int i = 2; i < N; i++)
	{
		VectorXd Y(5, 1);
		MatrixXd X(K, 5);
                MatrixXd XtX(K, K); // Xt is the transposition of X
                MatrixXd I(K, K); // identity matrix
		VectorXd W(K, 1); // need to compute
                I = MatrixXd::Identity(K, K);
                for (int j = 0; j < 5; j++) Y(j) = AT(allmat, i - 2, j);
                //    if (i == 1000) cout << Y << "   ";
		for (int j = 0; j < K; j++) // search the K-nearest pixels in RGBXY
		{
			int index = AT(allresult.indices, i - 2, j); // index + 2 is the index in N ///////////////////////ÏÂÃæÒªÐÞ¸Ä
			X.row(j) << AT(allmat, index, 0), AT(allmat, index, 1),  AT(allmat, index, 2), AT(allmat, index, 3),  AT(allmat, index, 4);
                        //       if (i == 1000) cout << X.row(j) << endl;
                        X.row(j) = Y.transpose() - X.row(j);

		}

                XtX = X * X.transpose();
                double tr = XtX.trace();
                XtX  = XtX + del * I * tr;
                XtX = XtX.inverse();

		for (int j = 0; j < K; j++) // search the K-nearest pixels in RGBXY
		{
                  W(j) = XtX.row(j).sum() / XtX.sum();
                  int Knearest =AT(allresult.indices, i - 2, j) + 2; // Knearest is the index of K-nearest neighbors of i
                  triplets.push_back(T(i, Knearest, -W(j)));
                  triplets.push_back(T(i, i, W(j))); // add to L(i, i)
		}
	}
	W3.setFromTriplets(triplets.begin(), triplets.end());
        W3.prune(0.0);
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
	// Iii = 1000 if i belongs to S(S = f + b + u(confidence > 0.85))
        int coe = 1000;
	std::vector<T> triplets;
	for (int i = 2; i < N; i++) // let two virtue nodes be 0 (I00 = I11 = 0)
	{
		int x = (i - 2) / width;
		int y = (i - 2) - x * width;
		if (tri[x][y] == 0 || tri[x][y] == 1 || confidence[x][y] > 0.85)
                  triplets.push_back(T(i, i, coe));
	}
	I.setFromTriplets(triplets.begin(), triplets.end());
	cout << "getI ok" << endl;
}

void   Imagematting::getL() // get unlocal smooth term Wlle(ij)
{
	// L = W1 + W2 + W3 (L(i, i) is already computed.)
	getWeight1();
	getWeight2();
	getWeight3();
	L = W1 + W2 + W3;
        L.prune(0.0);
	cout << "getL ok" << endl;
}

void   Imagematting::getFinalAlpha()
{
	getI();
	getG();
	getL();
        //    TEST(L);
	// (I + T(L) * L) * alpha = I * G
	SpMat A = I + (L.transpose() * L);
	A.prune(0.0); // delete zero

	VectorXd b = I * G;

	clock_t start, finish;
	start = clock();

	//ConjugateGradient<SpMat> cg(A); // use CG method
	//Alpha.setZero();
	//Alpha = cg.solve(b);

	SimplicialLDLT<SpMat> ldlt(A);
	Alpha = ldlt.solve(b);

	finish = clock();
	cout << double(finish - start) / CLOCKS_PER_SEC << endl;

	//let Alpha between 0-1
	Alpha = Alpha.cwiseAbs(); // abs
	saveMarket(Alpha, "Alpha.mtx");
	cout << "getFinalAlpha ok" << endl;
}

void Imagematting::showMatte()
{
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
                        //          udata[i * g_step + j] = int(preAlpha[i][j] * 255);
                        //    cout << Alpha[index] * 255 <<endl;
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
//fs["allindices"] >> allresult.indices;
//fs.release();

getPreAlpha(); // get predicted alpha of every pixel

//// get array preAlpha
//fstream f("a1.txt", ios::in);
//for (int i = 0; i < height; i++)
//{
//for (int j = 0; j < width; j++)
//{
//f >> preAlpha[i][j];
//}
//}

getFinalAlpha();

showMatte();
}
